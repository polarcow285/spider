# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run IK for the given hand type and mode. Using mink so that the joints don't "bend backwards"

Input data format: npz file which contains qpos for key frames.

TODO: for enable collision, first use non collision as initial guess

Author: Chaoyi Pan
Date: 2025-07-06

Increased posture regularization from a tiny fixed 1e-3 to a CLI parameter: posture_cost, default 1.0.
Each frame now uses the current solved pose as a posture prior before solving the next frame, so the IK prefers the same joint branch instead of jumping/flipping.
Added per-joint velocity clipping with max_joint_velocity, default 8.0 rad/s, while leaving free bodies unconstrained.
Added ik_substeps_per_frame, default 20, so the solver takes smaller smoother steps between keyframes.
"""

import os

import loguru
import mujoco
import mujoco.viewer
import numpy as np
import tyro
from mink import Configuration, FrameTask, PostureTask, solve_ik
from mink.lie import SE3
from loop_rate_limiters import RateLimiter
from mujoco import MjSpec
from omegaconf import DictConfig, OmegaConf
from scipy import signal

from spider import ROOT
from spider.io import get_processed_data_dir
from spider.mujoco_utils import get_viewer


LEAP_NO_BACKBEND_JOINTS = {
    "if_mcp",
    "if_pip",
    "if_dip",
    "mf_mcp",
    "mf_pip",
    "mf_dip",
    "rf_mcp",
    "rf_pip",
    "rf_dip",
    "th_mcp",
    "th_ipl",
}


def is_no_backbend_flexion_joint(robot_type: str, joint_name: str | None) -> bool:
    if joint_name is None:
        return False

    if robot_type == "leap":
        return joint_name in LEAP_NO_BACKBEND_JOINTS

    if robot_type == "wuji":
        if "_finger" not in joint_name or "_joint" not in joint_name:
            return False
        joint_number = joint_name.rsplit("_joint", maxsplit=1)[-1]
        return joint_number in {"1", "3", "4"}

    return False


def tighten_no_backbend_joint_limits(
    model: mujoco.MjModel,
    robot_type: str,
    min_flexion: float,
) -> list[str]:
    """Remove negative flexion from finger joints that can hyperextend."""
    tightened = []
    for jid in range(model.njnt):
        joint_type = model.jnt_type[jid]
        if joint_type != mujoco.mjtJoint.mjJNT_HINGE:
            continue

        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not is_no_backbend_flexion_joint(robot_type, joint_name):
            continue

        lower, upper = model.jnt_range[jid]
        if lower >= min_flexion or upper <= min_flexion:
            continue

        model.jnt_limited[jid] = 1
        model.jnt_range[jid, 0] = min_flexion
        tightened.append(joint_name)

    return tightened


def clamp_no_backbend_joint_positions(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    robot_type: str,
    min_flexion: float,
) -> bool:
    changed = False
    for jid in range(model.njnt):
        joint_type = model.jnt_type[jid]
        if joint_type != mujoco.mjtJoint.mjJNT_HINGE:
            continue

        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not is_no_backbend_flexion_joint(robot_type, joint_name):
            continue

        qpos_addr = model.jnt_qposadr[jid]
        lower = max(model.jnt_range[jid, 0], min_flexion)
        if qpos[qpos_addr] < lower:
            qpos[qpos_addr] = lower
            changed = True

    return changed


def add_mocap_bodies(
    mjspec: MjSpec,
    sites_for_mimic: list[str],
    mocap_bodies: list[str],
    robot_conf: DictConfig = None,
    add_equality_constraint: bool = True,
):
    """Add mocap bodies to the model specification.
    Source: https://github.com/robfiras/loco-mujoco

    Args:
        mjspec (MjSpec): The model specification.
        sites_for_mimic (List[str]): The sites to mimic.
        mocap_bodies (List[str]): The names of the mocap bodies to be added to the model specification.
        mocap_bodies_init_pos: The initial positions of the mocap bodies.
        add_equality_constraint (bool): Whether to add equality constraints between the sites and the mocap bodies.

    """
    if robot_conf is not None and robot_conf.optimization_params.disable_joint_limits:
        for j in mjspec.joints:
            j.limited = False

    for j in mjspec.joints:
        j.actfrclimited = 0

    if robot_conf is not None and robot_conf.optimization_params.disable_collisions:
        for g in mjspec.geoms:
            g.contype = 0
            g.conaffinity = 0

    for mb_name in mocap_bodies:
        b_handle = mjspec.worldbody.add_body(name=mb_name, mocap=True)
        if "wrist" in mb_name or "object" in mb_name:
            b_handle.add_site(
                name=mb_name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.01, 0.02, 0.03],
                rgba=[0.0, 1.0, 0.0, 0.5],
                group=1,
            )
        else:
            b_handle.add_site(
                name=mb_name,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0.01, 0.01],
                rgba=[0.0, 1.0, 0.0, 0.5],
                group=1,
            )

    if add_equality_constraint:
        for b1, b2 in zip(sites_for_mimic, mocap_bodies, strict=False):
            if robot_conf is not None:
                eq_type = getattr(
                    mujoco.mjtEq,
                    robot_conf.site_joint_matches[b1].equality_constraint_type,
                )
                torque_scale = robot_conf.site_joint_matches[b1].torque_scale
            else:
                eq_type = mujoco.mjtEq.mjEQ_CONNECT
                torque_scale = 1.0

            constraint_data = np.zeros(11)
            if eq_type == mujoco.mjtEq.mjEQ_WELD:
                constraint_data[3] = 1.0
            constraint_data[10] = torque_scale
            e = mjspec.add_equality(
                name=f"{b1}_{b2}_equality_constraint",
                type=eq_type,
                name1=b1,
                name2=b2,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                data=constraint_data,
            )

            if robot_conf is not None:
                if hasattr(robot_conf.site_joint_matches[b1], "solref"):
                    test = len(robot_conf.site_joint_matches[b1].solref)
                    assert len(robot_conf.site_joint_matches[b1].solref) == 2, (
                        "solref must be a list of length 2"
                    )
                    e.solref = robot_conf.site_joint_matches[b1].solref
                if hasattr(robot_conf.site_joint_matches[b1], "solimp"):
                    assert len(robot_conf.site_joint_matches[b1].solimp) == 5, (
                        "solimp must be a list of length 5"
                    )
                    e.solimp = robot_conf.site_joint_matches[b1].solimp

    return mjspec


def get_robot_sites(robot_type: str, embodiment_type: str):
    if robot_type in ["allegro", "metahand", "leap"]:
        sites_in_robot = [
            "right_wrist",
            "right_index_tip",
            "right_middle_tip",
            "right_ring_tip",
            "right_thumb_tip",
            "left_palm",
            "left_ring_tip",
            "left_middle_tip",
            "left_index_tip",
            "left_thumb_tip",
            "right_object",
            "left_object",
        ]
    else:
        sites_in_robot = [
            "right_wrist",
            "right_thumb_tip",
            "right_index_tip",
            "right_middle_tip",
            "right_ring_tip",
            "right_pinky_tip",
            "left_palm",
            "left_thumb_tip",
            "left_index_tip",
            "left_middle_tip",
            "left_ring_tip",
            "left_pinky_tip",
            "right_object",
            "left_object",
        ]
    if embodiment_type == "right":
        sites_in_robot = [s for s in sites_in_robot if "right" in s]
    elif embodiment_type == "left":
        sites_in_robot = [s for s in sites_in_robot if "left" in s]
    return sites_in_robot

def print_mj_qpos_layout(model):
    print("\n========== MuJoCo qpos layout ==========")
    qpos_index = 0

    for jid in range(model.njnt):
        jnt_type = model.jnt_type[jid]
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jnt_name == None:
            jnt_name = ""
        qpos_addr = model.jnt_qposadr[jid]

        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            size = 7
        elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
            size = 4
        else:
            size = 1

        print(f"[qpos {qpos_addr:03d}:{qpos_addr+size:03d}] {jnt_name:30s} type={jnt_type}")

        qpos_index += size

    print("Total qpos size:", model.nq)
    print("========================================\n")

def print_script_qpos_order(index_map):
    print("\n========== Script index_map qpos order ==========")

    ordered = sorted(index_map.items(), key=lambda x: x[1]["qpos_idx"])

    for name, info in ordered:
        print(f"qpos_idx={info['qpos_idx']:02d}  name={name}")

    print("=================================================\n")


def qpos_pose_to_se3(qpos_pose: np.ndarray) -> SE3:
    """Convert [x, y, z, qw, qx, qy, qz] target data to a Mink SE3."""
    mat = np.eye(4)
    rot = np.zeros(9)
    quat = qpos_pose[3:].copy()
    quat /= np.linalg.norm(quat)
    mujoco.mju_quat2Mat(rot, quat)
    mat[:3, :3] = rot.reshape(3, 3)
    mat[:3, 3] = qpos_pose[:3]
    return SE3.from_matrix(mat)


def make_mink_tasks(
    sites_for_mimic: list[str],
    model: mujoco.MjModel,
    posture_cost: float,
    wrist_position_cost: float,
    wrist_orientation_cost: float,
    object_position_cost: float,
    object_orientation_cost: float,
    finger_position_cost: float,
):
    tasks = {}
    for site_name in sites_for_mimic:
        if "wrist" in site_name:
            tasks[site_name] = FrameTask(
                frame_name=site_name,
                frame_type="site",
                position_cost=wrist_position_cost,
                orientation_cost=wrist_orientation_cost,
            )
        elif "object" in site_name:
            tasks[site_name] = FrameTask(
                frame_name=site_name,
                frame_type="site",
                position_cost=object_position_cost,
                orientation_cost=object_orientation_cost,
            )
        else:
            tasks[site_name] = FrameTask(
                frame_name=site_name,
                frame_type="site",
                position_cost=finger_position_cost,
                orientation_cost=0.0,
            )
    posture_task = PostureTask(model, cost=posture_cost)
    return tasks, posture_task


def set_mink_targets(
    tasks: dict[str, FrameTask],
    index_map: dict,
    qpos_ref: np.ndarray,
    frame_idx: int,
):
    for site_name, task in tasks.items():
        qpos_idx = index_map[site_name]["qpos_idx"]
        task.set_target(qpos_pose_to_se3(qpos_ref[frame_idx, qpos_idx]))


def solve_mink_velocity(configuration: Configuration, tasks: list, dt: float):
    try:
        return solve_ik(
            configuration,
            tasks,
            dt=dt,
            damping=1e-4,
            safety_break=False,
        )
    except TypeError:
        last_error = None
        for solver in ("quadprog", "proxqp", "osqp"):
            try:
                return solve_ik(
                    configuration,
                    tasks,
                    dt=dt,
                    solver=solver,
                    damping=1e-4,
                    safety_break=False,
                )
            except Exception as exc:
                last_error = exc
        raise last_error


def clip_joint_velocity(
    model: mujoco.MjModel,
    velocity: np.ndarray,
    max_joint_velocity: float,
):
    """Limit hinge/slide/ball joint speed while leaving free bodies unconstrained."""
    velocity = velocity.copy()
    for jid in range(model.njnt):
        jnt_type = model.jnt_type[jid]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            continue

        dof_addr = model.jnt_dofadr[jid]
        if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
            dof_size = 3
        else:
            dof_size = 1
        velocity[dof_addr : dof_addr + dof_size] = np.clip(
            velocity[dof_addr : dof_addr + dof_size],
            -max_joint_velocity,
            max_joint_velocity,
        )
    return velocity


# parameters
def main(
    dataset_dir: str = f"{ROOT}/../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "allegro",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    show_viewer: bool = True,
    save_video: bool = False,
    enable_collision: bool = False,
    start_idx: int = 0,
    end_idx: int = -1,
    sim_dt: float = 0.002, # 0.01,
    ref_dt: float = 0.02,
    data_id: int = 0,
    keypoint_path: str | None = None,
    keypoint_variant: str | None = None,
    open_hand: bool = False,
    contact_detection_step_threshold: int = 3,
    finger_solimp_width: float = 0.01,
    wrist_solimp_width: float = 10.0,
    wrist_torque_scale: float = 10.0,
    object_solimp_width: float = 0.001,
    max_num_initial_guess: int = 8,
    average_frame_size: int = 3,
    aggregate_contact: bool = True,
    z_offset: float = 0.0,
    posture_cost: float = 1.0,
    wrist_position_cost: float = 50.0,
    wrist_orientation_cost: float = 10.0,
    object_position_cost: float = 50.0,
    object_orientation_cost: float = 10.0,
    finger_position_cost: float = 20.0,
    max_joint_velocity: float = 8.0,
    ik_substeps_per_frame: int = 20,
    prevent_backward_finger_bending: bool = True,
    no_backbend_min_flexion: float = 0.0,
    visualization_fps: int = 120,
    articulated: bool = False,
    articulated_joint_name: str = "scissors_joint",
    articulated_body_name: str = "bottom",
):
    # resolved processed directories
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir_robot = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    processed_dir_mano = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(processed_dir_robot, exist_ok=True)
    # load model from processed scene
    model_path = f"{processed_dir_robot}/../scene.xml"
    # NOTE: sites in robot should follow the order of the xml file
    sites_in_robot = get_robot_sites(robot_type, embodiment_type)

    file_path = f"{processed_dir_mano}/trajectory_keypoints_{robot_type}.npz"
    loaded_data = np.load(file_path)
    frame_slice = slice(start_idx, None if end_idx < 0 else end_idx)
    qpos_finger_right = loaded_data["qpos_finger_right"][frame_slice]
    qpos_finger_left = loaded_data["qpos_finger_left"][frame_slice]
    qpos_wrist_right = loaded_data["qpos_wrist_right"][frame_slice]
    qpos_wrist_left = loaded_data["qpos_wrist_left"][frame_slice]
    qpos_obj_right = loaded_data["qpos_obj_right"][frame_slice]
    qpos_obj_left = loaded_data["qpos_obj_left"][frame_slice]
    obj_arti = None
    if articulated:
        if "obj_arti" not in loaded_data:
            raise ValueError(
                f"articulated=True requires `obj_arti` in {file_path}"
            )
        obj_arti = loaded_data["obj_arti"][frame_slice]
        if obj_arti.ndim == 1:
            obj_arti = obj_arti[:, None]
    try:
        contact_left = loaded_data["contact_left"][frame_slice]
        contact_right = loaded_data["contact_right"][frame_slice]
    except:
        loguru.logger.warning("No contact data found, using all one")
        contact_left = np.ones((qpos_finger_right.shape[0], 5))
        contact_right = np.ones((qpos_finger_left.shape[0], 5))
    contact_ref = np.concatenate([contact_right, contact_left], axis=1)
    if aggregate_contact:
        contact_aggregated = np.any(contact_ref, axis=-1)
        for i in range(contact_ref.shape[1]):
            contact_ref[:, i] = contact_aggregated
    # get the first contact frame where contact_left turns to 1 (two 1s consecutive)
    first_contact_frame_left = np.zeros(5) + qpos_finger_right.shape[0]
    first_contact_frame_right = np.zeros(5) + qpos_finger_left.shape[0]
    for j in range(5):
        for i in range(contact_detection_step_threshold, len(contact_left)):
            if contact_left[i - contact_detection_step_threshold : i, j].all():
                first_contact_frame_left[j] = i
                break
        for i in range(contact_detection_step_threshold, len(contact_right)):
            if contact_right[i - contact_detection_step_threshold : i, j].all():
                first_contact_frame_right[j] = i
                break

    qpos_ref = np.concatenate(
        [
            qpos_wrist_right[:, None],
            qpos_finger_right,
            qpos_wrist_left[:, None],
            qpos_finger_left,
            qpos_obj_right[:, None],
            qpos_obj_left[:, None],
        ],
        axis=1,
    )
    qpos_ref[:, :, 2] += z_offset

    # load model
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_model.opt.timestep = sim_dt
    no_backbend_joint_names = []
    if prevent_backward_finger_bending:
        no_backbend_joint_names = tighten_no_backbend_joint_limits(
            mj_model,
            robot_type,
            no_backbend_min_flexion,
        )
        if no_backbend_joint_names:
            loguru.logger.info(
                "Preventing backward finger bending by setting lower limits to "
                f"{no_backbend_min_flexion} for: {no_backbend_joint_names}"
            )
    mj_data = mujoco.MjData(mj_model)
    print_mj_qpos_layout(mj_model)

    # NOTE: sites for mimic should follow the order of data
    index_map = {}
    cnt = 0
    for sides in ["right", "left"]:
        for body_name in [
            "wrist",
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "pinky_tip",
        ]:
            index_map[f"{sides}_{body_name}"] = {
                "qpos_idx": cnt,
                "mocap_idx": -1,
                "eq_constraint_idx": -1,
            }
            cnt += 1
    # add objects
    index_map["right_object"] = {
        "qpos_idx": cnt,
        "mocap_idx": -1,
        "eq_constraint_idx": -1,
    }
    cnt += 1
    index_map["left_object"] = {
        "qpos_idx": cnt,
        "mocap_idx": -1,
        "eq_constraint_idx": -1,
    }

    cnt += 1

    sites_for_mimic = [
        "right_wrist",
        "right_thumb_tip",
        "right_index_tip",
        "right_middle_tip",
        "right_ring_tip",
        "right_pinky_tip",
        "left_palm",
        "left_thumb_tip",
        "left_index_tip",
        "left_middle_tip",
        "left_ring_tip",
        "left_pinky_tip",
        "right_object",
        "left_object",
    ]

    # special case: allegro hand
    if robot_type in ["allegro", "metahand", "leap"]:
        sites_for_mimic.remove("right_pinky_tip")
        sites_for_mimic.remove("left_pinky_tip")

    if embodiment_type == "right":
        sites_for_mimic = [s for s in sites_for_mimic if "right" in s]
    elif embodiment_type == "left":
        sites_for_mimic = [s for s in sites_for_mimic if "left" in s]
    site_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, s)
        for s in sites_for_mimic
    ]
    # make sure all site_ids are valid, i.e. no -1
    assert all(site_id != -1 for site_id in site_ids), f"site_ids: {site_ids}"
    mano2mimic_site_idx = []
    for s in sites_for_mimic:
        # find robot site name
        for site_name in sites_in_robot:
            if site_name == s:
                mano2mimic_site_idx.append(sites_in_robot.index(site_name))

    # create mocap sites for retargeting
    site_joint_matches = {}
    for key in sites_for_mimic:
        if "wrist" in key:  # palm: strong rotation constraint, weak position constraint
            constraint_type = "mjEQ_WELD"
            solimp = [0.0, 0.95, wrist_solimp_width, 0.5, 2.0]
            torque_scale = wrist_torque_scale
            solref = [0.02, 1.0]
        elif "object" in key:  # object: strong position and rotation constraint
            constraint_type = "mjEQ_WELD"
            torque_scale = 10.0
            solimp = [0.9, 0.95, object_solimp_width, 0.5, 2.0]
            solref = [0.002, 1.0]
        else:  # finger: weak rotation constraint, strong position constraint (but weaker than object)
            constraint_type = "mjEQ_CONNECT"
            if "thumb" in key or "index" in key or "middle" in key:
                width_scale = 1.0
            else:
                width_scale = 3.0
            solimp = [0.0, 0.95, finger_solimp_width * width_scale, 0.5, 2.0]
            solref = [0.01, 1.0]
            torque_scale = 1.0
        site_joint_matches[key] = {
            "equality_constraint_type": constraint_type,
            "torque_scale": torque_scale,
            "solref": solref,
            "solimp": solimp,
        }

    robot_conf = OmegaConf.create(
        {
            "optimization_params": {
                "disable_joint_limits": False,
                "disable_collisions": not enable_collision,
            },
            "site_joint_matches": site_joint_matches,
        }
    )
    target_mocap_bodies = ["target_mocap_body_" + s for s in sites_for_mimic]
    mj_spec = mujoco.MjSpec.from_file(model_path)

    # ================================
    # add target mocap bodies for visualization
    # ================================
    mjspec = add_mocap_bodies(
        mj_spec,
        sites_for_mimic,
        target_mocap_bodies,
        robot_conf,
        add_equality_constraint=False,
    )

    # ================================
    # add constraints to relative bodies, i.e. stick the object to the finger
    # ================================
    finger_names = [
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
    ]
    if robot_type in ["allegro", "metahand", "leap"]:
        finger_names = finger_names[:4]

    sides = {
        "right": ["right"],
        "left": ["left"],
        "bimanual": ["right", "left"],
    }[embodiment_type]

    # add position sensor to sites_for_mimic
    for i in range(len(sites_for_mimic)):
        site_name = sites_for_mimic[i]
        mjspec.add_sensor(
            name=f"pos_{site_name}",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=site_name,
        )

    mj_model_ik = mj_spec.compile()
    mj_model_ik.opt.timestep = sim_dt
    if prevent_backward_finger_bending:
        tighten_no_backbend_joint_limits(
            mj_model_ik,
            robot_type,
            no_backbend_min_flexion,
        )
    mj_model_ik.opt.iterations = 20
    mj_model_ik.opt.ls_iterations = 50
    if not enable_collision:
        mj_model_ik.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mj_model_ik.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    configuration = Configuration(mj_model_ik)
    mj_data_ik = configuration.data
    mink_tasks, posture_task = make_mink_tasks(
        sites_for_mimic,
        mj_model_ik,
        posture_cost=posture_cost,
        wrist_position_cost=wrist_position_cost,
        wrist_orientation_cost=wrist_orientation_cost,
        object_position_cost=object_position_cost,
        object_orientation_cost=object_orientation_cost,
        finger_position_cost=finger_position_cost,
    )
    loguru.logger.info(
        "Mink task costs: "
        f"wrist_pos={wrist_position_cost}, "
        f"wrist_ori={wrist_orientation_cost}, "
        f"finger_pos={finger_position_cost}, "
        f"object_pos={object_position_cost}, "
        f"object_ori={object_orientation_cost}, "
        f"posture={posture_cost}"
    )
    articulated_qadr = None
    if articulated:
        articulated_jid = mujoco.mj_name2id(
            mj_model_ik,
            mujoco.mjtObj.mjOBJ_JOINT,
            articulated_joint_name,
        )
        if articulated_jid == -1:
            raise ValueError(
                f"Could not find articulated joint `{articulated_joint_name}`"
            )
        articulated_qadr = mj_model_ik.jnt_qposadr[articulated_jid]
    nq_obj = 14 if embodiment_type == "bimanual" else 7
    if articulated:
        nq_obj += obj_arti.shape[1]

    # update index_map
    for target_mocap_body in target_mocap_bodies:
        body_name = target_mocap_body[18:]
        body_id = mujoco.mj_name2id(
            mj_model_ik, mujoco.mjtObj.mjOBJ_BODY, target_mocap_body
        )
        mocap_id = mj_model_ik.body_mocapid[body_id]
        index_map[body_name]["mocap_idx"] = mocap_id
        # print(body_name, body_id, mocap_id)

    # set object position
    if embodiment_type == "bimanual":
        mj_data_ik.qpos[-14:-7] = qpos_obj_right[0]
        mj_data_ik.qpos[-7:] = qpos_obj_left[0]
    elif embodiment_type == "right":
        body_name = articulated_body_name if articulated else "bottom_visual"
        j_obj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if j_obj == -1 and not articulated:
            j_obj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "bottom")
        if j_obj == -1:
            raise ValueError(f"Could not find object body `{body_name}`")
        jnt_adr = mj_model.body_jntadr[j_obj]
        mj_data_ik.qpos[jnt_adr:jnt_adr+7] = qpos_obj_right[0]
    elif embodiment_type == "left":
        mj_data_ik.qpos[-7:] = qpos_obj_left[0]
    if articulated:
        mj_data_ik.qpos[
            articulated_qadr : articulated_qadr + obj_arti.shape[1]
        ] = obj_arti[0]
    configuration.update(mj_data_ik.qpos)
    posture_task.set_target(configuration.q)

    # set the mocap sites to the tip positions
    # for i, site_id in enumerate(site_ids):
    for i in range(len(sites_for_mimic)):
        site_id = site_ids[i]
        site_name = sites_for_mimic[i]
        mano_id = mano2mimic_site_idx[i]
        mocap_id = i
        mj_data_ik.mocap_pos[mocap_id] = qpos_ref[0, mano_id, :3]
        mj_data_ik.mocap_quat[mocap_id] = qpos_ref[0, mano_id, 3:]

    # rollout mujoco
    # Keep trajectory integration on ref_dt, but preview video/viewer at the
    # dataset visualization rate so it does not look artificially slowed down.
    rate_limiter = RateLimiter(visualization_fps)
    H = qpos_finger_right.shape[0]
    cnt = 0
    if save_video:
        import imageio

        mj_model_ik.vis.global_.offwidth = 720
        mj_model_ik.vis.global_.offheight = 480
        renderer = mujoco.Renderer(mj_model_ik, height=480, width=720)
    # TODO: move it to mujoco_utils
    run_viewer = get_viewer(show_viewer, mj_model_ik, mj_data_ik)

    # random initial guess to find a stable initial pose

    ref_mocap_ids = []
    ref_site_ids = []
    track_site_ids = []
    # get track site ids
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is not None and name.startswith("track"):
            track_site_ids.append(sid)

    for sid in track_site_ids:
        track_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        ref_name = track_name.replace("track", "ref")
        # get mocap id of ref site
        mocap_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, ref_name)
        mocap_id = mj_model.body_mocapid[mocap_body_id]
        ref_mocap_ids.append(mocap_id)
        # get site id of ref site
        ref_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, ref_name)
        ref_site_ids.append(ref_site_id)
    print("Ref mocap ids: ", ref_mocap_ids)

    with run_viewer() as gui:
        cnt = 0
        while cnt < H:
            if cnt == 0:
                # reset distance cost
                cost_sum = 0.0
                set_mink_targets(mink_tasks, index_map, qpos_ref, cnt)
                posture_task.set_target(configuration.q.copy())
                task_list = list(mink_tasks.values()) + [posture_task]
                init_steps = max(
                    ik_substeps_per_frame,
                    max(1, max_num_initial_guess) * 10,
                )
                for _ in range(init_steps):
                    vel = solve_mink_velocity(configuration, task_list, sim_dt)
                    vel = clip_joint_velocity(
                        mj_model_ik,
                        vel,
                        max_joint_velocity=max_joint_velocity,
                    )
                    configuration.integrate_inplace(vel, sim_dt)
                    if articulated:
                        mj_data_ik.qpos[
                            articulated_qadr : articulated_qadr + obj_arti.shape[1]
                        ] = obj_arti[cnt]
                        configuration.update(mj_data_ik.qpos)
                    if (
                        prevent_backward_finger_bending
                        and clamp_no_backbend_joint_positions(
                            mj_model_ik,
                            mj_data_ik.qpos,
                            robot_type,
                            no_backbend_min_flexion,
                        )
                    ):
                        configuration.update(mj_data_ik.qpos)
                    mj_data_ik.qvel[:] = vel
                mj_data_ik.qvel[:] = 0.0
                qpos_list = []
                contact_pos_list = []
                contact_list = []
                images = []

            for k, v in index_map.items():
                if v["mocap_idx"] != -1:
                    mj_data_ik.mocap_pos[v["mocap_idx"]] = qpos_ref[
                        cnt, v["qpos_idx"], :3
                    ]
                    mj_data_ik.mocap_quat[v["mocap_idx"]] = qpos_ref[
                        cnt, v["qpos_idx"], 3:
                    ]

            set_mink_targets(mink_tasks, index_map, qpos_ref, cnt)
            posture_task.set_target(configuration.q.copy())
            task_list = list(mink_tasks.values()) + [posture_task]
            num_ik_substeps = max(1, ik_substeps_per_frame, int(ref_dt / sim_dt))
            ik_dt = ref_dt / num_ik_substeps
            for _ in range(num_ik_substeps):
                vel = solve_mink_velocity(configuration, task_list, ik_dt)
                vel = clip_joint_velocity(
                    mj_model_ik,
                    vel,
                    max_joint_velocity=max_joint_velocity,
                )
                configuration.integrate_inplace(vel, ik_dt)
                if articulated:
                    mj_data_ik.qpos[
                        articulated_qadr : articulated_qadr + obj_arti.shape[1]
                    ] = obj_arti[cnt]
                    configuration.update(mj_data_ik.qpos)
                if (
                    prevent_backward_finger_bending
                    and clamp_no_backbend_joint_positions(
                        mj_model_ik,
                        mj_data_ik.qpos,
                        robot_type,
                        no_backbend_min_flexion,
                    )
                ):
                    configuration.update(mj_data_ik.qpos)
                mj_data_ik.qvel[:] = vel
            mujoco.mj_forward(mj_model_ik, mj_data_ik)

            # set site position and set it to ref mocap position (use original mj_model and mj_data)
            mj_data.qpos[:] = mj_data_ik.qpos.copy()
            if articulated:
                mj_data.qpos[
                    articulated_qadr : articulated_qadr + obj_arti.shape[1]
                ] = obj_arti[cnt]
            mj_data.qvel[:] = 0.0
            mj_data.ctrl[:] = mj_data_ik.qpos[:-nq_obj].copy()

            # override joint position according to contact state
            if open_hand:
                for side in ["right", "left"]:
                    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                        # get joint index
                        joint_ids = []
                        for jid in range(mj_model.njnt):
                            joint_name = mujoco.mj_id2name(
                                mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid
                            )
                            if side in joint_name and finger in joint_name:
                                joint_ids.append(jid)
                        if len(joint_ids) > 0:
                            for joint_idx in joint_ids:
                                current_joint_pos = mj_data.qpos[joint_idx]
                                zero_joint_pos = 0.0
                                # Map sides and fingers to their respective indices
                                side_map = {
                                    "right": first_contact_frame_right,
                                    "left": first_contact_frame_left,
                                }
                                finger_map = {
                                    "thumb": 0,
                                    "index": 1,
                                    "middle": 2,
                                    "ring": 3,
                                    "pinky": 4,
                                }

                                contact_frame_list = side_map[side]
                                finger_idx = finger_map[finger]
                                contact_frame = contact_frame_list[finger_idx]

                                # Use smooth transition with clipping
                                ratio = np.clip(cnt / max(contact_frame, 1), 0.0, 1.0)
                                ratio = 1.0 - np.cos(ratio * np.pi * 0.5)
                                joint_pos = (
                                    ratio * current_joint_pos
                                    + (1 - ratio) * zero_joint_pos
                                )
                                mj_data.qpos[joint_idx] = joint_pos

            mujoco.mj_kinematics(mj_model, mj_data)
            for i in range(len(ref_mocap_ids)):
                mocap_id = ref_mocap_ids[i]
                track_site_id = track_site_ids[i]
                track_site_name = mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_SITE, track_site_id
                )
                mj_data.mocap_pos[mocap_id] = mj_data.site_xpos[track_site_id].copy()

            contact = np.zeros(len(track_site_ids))
            contact_map = {
                "right_thumb": 0,
                "right_index": 1,
                "right_middle": 2,
                "right_ring": 3,
                "right_pinky": 4,
                "left_thumb": 5,
                "left_index": 6,
                "left_middle": 7,
                "left_ring": 8,
                "left_pinky": 9,
            }
            for i in range(len(track_site_ids)):
                track_site_name = mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_SITE, track_site_ids[i]
                )
                for k, v in contact_map.items():
                    if k in track_site_name and "object" in track_site_name:
                        contact[i] = contact_ref[cnt, v]
                        break

            mujoco.mj_forward(mj_model, mj_data)
            contact_pos_list.append(mj_data.mocap_pos.copy())
            # get contact list
            # logic: for each track_site, check its corresponding object site (e.g. track site named "track_hand_right_index_tip" should correspond to "track_object_right_index_tip")
            # similarly, "track_object_right_index_tip" should correspond to "track_hand_right_index_tip"
            # after find its corresponding object site, check if the distance between the two sites is less than 0.01, if so, set contact to 1, otherwise set contact to 0
            # contact order should follow track site definition order
            # contact size is equal to check sites number
            # for i in range(len(track_site_ids)):
            #     track_site_id = track_site_ids[i]
            #     track_site_pos = mj_data.site_xpos[track_site_id].copy()
            #     track_site_name = mujoco.mj_id2name(
            #         mj_model, mujoco.mjtObj.mjOBJ_SITE, track_site_id
            #     )
            #     if "hand" in track_site_name:
            #         match_site_name = track_site_name.replace("hand", "object")
            #     elif "object" in track_site_name:
            #         match_site_name = track_site_name.replace("object", "hand")
            #     else:
            #         raise ValueError(f"Invalid track site name: {track_site_name}")
            #     match_site_id = mujoco.mj_name2id(
            #         mj_model, mujoco.mjtObj.mjOBJ_SITE, match_site_name
            #     )
            #     match_site_pos = mj_data.site_xpos[match_site_id].copy()
            #     if np.linalg.norm(track_site_pos - match_site_pos) < 0.01:
            #         contact[i] = 1
            #     else:
            #         contact[i] = 0
            contact_list.append(contact)

            # get contact point distance
            for i in range(len(sites_for_mimic)):
                site_name = sites_for_mimic[i]

            qpos_list.append(mj_data.qpos.copy())
            if save_video:
                opt = mujoco.MjvOption()
                # opt.sitegroup[4] = 1
                renderer.update_scene(data=mj_data, camera="front", scene_option=opt)
                images.append(renderer.render())
            if show_viewer:
                gui.sync()
                rate_limiter.sleep()
            cnt += 1
            if cnt == H:
                cost_mean = cost_sum / H
                def decode_qpos(model):
                    mapping = []

                    for jid in range(model.njnt):
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                        jtype = model.jnt_type[jid]
                        addr = model.jnt_qposadr[jid]

                        if jtype == mujoco.mjtJoint.mjJNT_FREE:
                            size = 7
                            fields = ["x", "y", "z", "qw", "qx", "qy", "qz"]
                        elif jtype == mujoco.mjtJoint.mjJNT_BALL:
                            size = 4
                            fields = ["qw", "qx", "qy", "qz"]
                        else:
                            size = 1
                            fields = ["q"]

                        for i in range(size):
                            mapping.append((addr + i, f"{name}:{fields[i]}"))

                    return sorted(mapping, key=lambda x: x[0])


                print("\n========== FULL qpos decoding ==========\n")
                mapping = decode_qpos(mj_model)
                for idx, label in mapping:
                    print(f"qpos[{idx:02d}] -> {label}")
                if show_viewer:
                    # check if the rollout is good, if so, break
                    user_input = input("Is the rollout good? (y/n): ")
                    if user_input.lower() == "y":
                        break
                    else:
                        cnt = 0
                else:
                    break

        file_dir = processed_dir_robot
        os.makedirs(file_dir, exist_ok=True)
        if save_video:
            imageio.mimsave(
                f"{file_dir}/visualization_ik.mp4",
                images,
                fps=visualization_fps,
            )
            loguru.logger.info(
                f"Saved visualization video to {file_dir}/visualization_ik.mp4"
            )

        qpos_list = np.array(qpos_list)

                # ============================================================
        # Save EXACT trajectory that was visualized in MuJoCo viewer
        # ============================================================

        qpos_list = np.array(qpos_list)
        contact_pos_list = np.array(contact_pos_list)
        contact_list = np.array(contact_list)

        H = qpos_list.shape[0]

        # Compute qvel directly from the visualized trajectory. Keep frame 0 so
        # output length matches the input trajectory length.
        qvel_list = np.zeros((H, mj_model_ik.nv))

        for i in range(1, H):
            mujoco.mj_differentiatePos(
                mj_model_ik,
                qvel_list[i],
                ref_dt,
                qpos_list[i - 1],
                qpos_list[i],
            )

        qpos_save = qpos_list
        contact_pos_save = contact_pos_list
        contact_save = contact_list

        assert qpos_save.shape[0] == qvel_list.shape[0]

        out_npz = f"{file_dir}/trajectory_kinematic_mink.npz"

        np.savez(
            out_npz,
            qpos=qpos_save,
            qvel=qvel_list,
            contact=contact_save,
            contact_pos=contact_pos_save,
            frequency=1 / ref_dt,
        )

        # Save identical rollout copy for compatibility
        out_npz_rollout = f"{file_dir}/trajectory_ikrollout_mink.npz"

        np.savez(
            out_npz_rollout,
            qpos=qpos_save,
        )

        loguru.logger.info(f"Saved {out_npz}")
        loguru.logger.info(f"Saved {out_npz_rollout}")


if __name__ == "__main__":
    tyro.cli(main)
