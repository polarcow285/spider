"""
Convert a retargeted NPZ trajectory to an H5 trajectory for Isaac/DexRL.

Default input format: npz file 'trajectory_kinematic_mink.npz'
Keys:
    qpos: shape=(T, nq), MuJoCo qpos.
    frequency: scalar, optional metadata.

With --dexmachina, the input defaults to
'trajectory_dexmachina_minimal_{robot_type}.npz' and also contains:
    contact_pos: shape=(T, num_object_parts, num_hand_links, 3), float64
    contact_mask: shape=(T, num_object_parts, num_hand_links), bool
    contact_links: shape=(T, num_object_parts, num_hand_links, 4), float64
        xyz plus DexMachina-style part id (1=top, 2=bottom when available)
    contact_part_names: shape=(num_object_parts,), str
    contact_link_names: shape=(num_hand_links,), str

Output format: h5 file 'mink_retargeted_{robot_type}_{task}.h5' compatible with
replay_retargeted_traj.py / dexrl.data.arctic.load_retargeted_traj.
With --dexmachina, the output file is named
'dexmachina_retargeted_{robot_type}_{task}.h5'.
Saved by default in spider/postprocess/mink/.
Keys:
    object_pos: shape=(T, 3), float64
    object_quat: shape=(T, 4), float64, wxyz
    object_joint: shape=(T,), float64, optional when qpos contains one
    robot_pos: shape=(T, 3), float64
    robot_quat: shape=(T, 4), float64, wxyz
    robot_euler_XYZ: shape=(T, 3), float64, intrinsic XYZ
    robot_joints: shape=(T, n_hand_dof), float64
    robot_keypoints: shape=(T, 17, 3) for leap or (T, 21, 3) for wuji,
        float64; wrist keypoint followed by hand joint anchors in robot_joints order.
    contact: H5 group containing the DexMachina contact keys above when
        --dexmachina is enabled.

Expected qpos layout:
    [robot_pos_xyz(3), robot_euler_XYZ(3), robot_finger_joints,
     object_pos_xyz(3), object_quat_wxyz(4), optional_object_joint]

Example usage:
    python spider/postprocess/isaac.py \
        --task scissors --embodiment-type right \
        --dataset-dir example_datasets --dataset-name arctic --robot-type leap
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import loguru
import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R

from spider.io import get_processed_data_dir


DEFAULT_ROBOT_JOINT_COUNTS = {
    "leap": 16,
    "allegro": 16,
    "metahand": 16,
    "wuji": 20,
}

DEFAULT_ROBOT_KEYPOINT_COUNTS = {
    "leap": 17,
    "wuji": 21,
}

CONTACT_KEYS = (
    "contact_pos",
    "contact_mask",
    "contact_links",
    "contact_part_names",
    "contact_link_names",
)


def default_robot_joint_count(robot_type: str) -> int:
    try:
        return DEFAULT_ROBOT_JOINT_COUNTS[robot_type]
    except KeyError as exc:
        raise ValueError(
            f"No default robot_joint_count for `{robot_type}`. "
            "Pass --robot-joint-count explicitly."
        ) from exc


def split_mink_qpos(
    qpos: np.ndarray,
    robot_type: str,
    robot_joint_count: int | None,
) -> dict[str, np.ndarray]:
    if robot_joint_count is None:
        robot_joint_count = default_robot_joint_count(robot_type)

    robot_root_dim = 6
    object_pose_dim = 7
    min_qpos_dim = robot_root_dim + robot_joint_count + object_pose_dim
    if qpos.shape[1] < min_qpos_dim:
        raise ValueError(
            f"qpos has width {qpos.shape[1]}, but expected at least {min_qpos_dim} "
            f"for robot_root_dim={robot_root_dim}, "
            f"robot_joint_count={robot_joint_count}, object_pose_dim={object_pose_dim}."
        )

    robot_pos = qpos[:, 0:3]
    robot_euler = qpos[:, 3:6]
    robot_quat_xyzw = R.from_euler("XYZ", robot_euler).as_quat()
    robot_quat = robot_quat_xyzw[:, [3, 0, 1, 2]]

    robot_joint_start = robot_root_dim
    robot_joint_end = robot_joint_start + robot_joint_count
    robot_joints = qpos[:, robot_joint_start:robot_joint_end]

    object_start = robot_joint_end
    object_pos = qpos[:, object_start : object_start + 3]
    object_quat = qpos[:, object_start + 3 : object_start + 7]

    output = {
        "robot_pos": robot_pos,
        "robot_quat": robot_quat,
        "robot_euler_XYZ": robot_euler,
        "robot_joints": robot_joints,
        "object_pos": object_pos,
        "object_quat": object_quat,
    }

    remaining = qpos.shape[1] - (object_start + object_pose_dim)
    if remaining == 1:
        output["object_joint"] = qpos[:, object_start + object_pose_dim]
    elif remaining > 1:
        loguru.logger.warning(
            f"qpos has {remaining} extra values after object pose; "
            "only the first extra value is exported as object_joint."
        )
        output["object_joint"] = qpos[:, object_start + object_pose_dim]

    return output


def joint_qpos_width(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def find_site_or_body(
    model: mujoco.MjModel,
    names: list[str],
) -> tuple[str, int, str]:
    for name in names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id != -1:
            return ("site", site_id, name)
    for name in names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id != -1:
            return ("body", body_id, name)
    raise ValueError(f"Could not find any site/body named one of {names}.")


def wrist_keypoint_handle(
    model: mujoco.MjModel,
    embodiment_type: str,
) -> tuple[str, int, str]:
    if embodiment_type == "left":
        candidates = ["left_wrist", "left_palm", "left_hand"]
    else:
        candidates = ["right_wrist", "right_palm", "right_hand"]
    return find_site_or_body(model, candidates)


def hand_joint_keypoint_ids(
    model: mujoco.MjModel,
    robot_joint_count: int,
) -> tuple[np.ndarray, list[str]]:
    robot_root_dim = 6
    robot_joint_end = robot_root_dim + robot_joint_count
    joint_ids = []
    joint_names = []

    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        qpos_addr = int(model.jnt_qposadr[joint_id])
        qpos_end = qpos_addr + joint_qpos_width(joint_type)
        if qpos_addr < robot_root_dim or qpos_end > robot_joint_end:
            continue
        if joint_type not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue

        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_ids.append(joint_id)
        joint_names.append("" if name is None else name)

    order = np.argsort([int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids])
    sorted_joint_ids = np.array(joint_ids, dtype=np.int64)[order]
    sorted_joint_names = [joint_names[idx] for idx in order]
    if len(sorted_joint_ids) != robot_joint_count:
        raise ValueError(
            f"Expected {robot_joint_count} hand joints in qpos addresses "
            f"[{robot_root_dim}, {robot_joint_end}), but found "
            f"{len(sorted_joint_ids)}: {sorted_joint_names}."
        )
    return sorted_joint_ids, sorted_joint_names


def compute_robot_keypoints(
    qpos: np.ndarray,
    model_path: str,
    robot_type: str,
    embodiment_type: str,
    robot_joint_count: int | None,
) -> tuple[np.ndarray, list[str]]:
    if robot_type not in DEFAULT_ROBOT_KEYPOINT_COUNTS:
        raise ValueError(
            f"robot_keypoints export is only defined for "
            f"{sorted(DEFAULT_ROBOT_KEYPOINT_COUNTS)}; got `{robot_type}`."
        )
    if robot_joint_count is None:
        robot_joint_count = default_robot_joint_count(robot_type)

    model = mujoco.MjModel.from_xml_path(model_path)
    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"Trajectory qpos width {qpos.shape[1]} does not match MuJoCo "
            f"model.nq {model.nq} from {model_path}."
        )

    wrist_kind, wrist_id, wrist_name = wrist_keypoint_handle(model, embodiment_type)
    joint_ids, joint_names = hand_joint_keypoint_ids(
        model=model,
        robot_joint_count=robot_joint_count,
    )

    data = mujoco.MjData(model)
    keypoints = np.empty((qpos.shape[0], 1 + len(joint_ids), 3), dtype=np.float64)
    for frame_idx, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(model, data)
        if wrist_kind == "site":
            keypoints[frame_idx, 0] = data.site_xpos[wrist_id]
        else:
            keypoints[frame_idx, 0] = data.xpos[wrist_id]
        keypoints[frame_idx, 1:] = data.xanchor[joint_ids]

    expected_count = DEFAULT_ROBOT_KEYPOINT_COUNTS[robot_type]
    if keypoints.shape[1] != expected_count:
        raise ValueError(
            f"Expected robot_keypoints shape (T, {expected_count}, 3) for "
            f"`{robot_type}`, got {keypoints.shape}."
        )
    return keypoints, [wrist_name, *joint_names]


def load_contact_data(
    trajectory: np.lib.npyio.NpzFile,
    frame_slice: slice,
    num_frames: int,
) -> dict[str, np.ndarray]:
    missing = [key for key in CONTACT_KEYS if key not in trajectory]
    if missing:
        raise KeyError(f"DexMachina input is missing contact keys: {missing}.")

    contact = {
        "contact_pos": np.asarray(trajectory["contact_pos"], dtype=np.float64)[
            frame_slice
        ],
        "contact_mask": np.asarray(trajectory["contact_mask"], dtype=bool)[
            frame_slice
        ],
        "contact_links": np.asarray(trajectory["contact_links"], dtype=np.float64)[
            frame_slice
        ],
        "contact_part_names": np.asarray(trajectory["contact_part_names"]),
        "contact_link_names": np.asarray(trajectory["contact_link_names"]),
    }
    if contact["contact_pos"].shape != contact["contact_mask"].shape + (3,):
        raise ValueError("contact_pos must have shape contact_mask.shape + (3,).")
    if contact["contact_links"].shape != contact["contact_mask"].shape + (4,):
        raise ValueError("contact_links must have shape contact_mask.shape + (4,).")
    if contact["contact_pos"].shape[0] != num_frames:
        raise ValueError("Contact frame count does not match selected qpos frames.")
    if contact["contact_part_names"].shape[0] != contact["contact_pos"].shape[1]:
        raise ValueError("contact_part_names length does not match contact_pos.")
    if contact["contact_link_names"].shape[0] != contact["contact_pos"].shape[2]:
        raise ValueError("contact_link_names length does not match contact_pos.")
    return contact


def write_group(group: h5py.Group, data: dict[str, np.ndarray]) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    for key, value in data.items():
        if value.dtype.kind in {"U", "S", "O"}:
            group.create_dataset(key, data=value.astype(string_dtype))
        else:
            group.create_dataset(key, data=value)


def write_h5(
    path: str,
    data: dict[str, np.ndarray],
    attrs: dict[str, str | float],
    contact: dict[str, np.ndarray] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as h5_file:
        for key, value in data.items():
            h5_file.create_dataset(key, data=value.astype(np.float64))
        if contact is not None:
            write_group(h5_file.create_group("contact"), contact)
        for key, value in attrs.items():
            h5_file.attrs[key] = value


def main(
    dataset_dir: str = "../../example_datasets",
    dataset_name: str = "arctic",
    robot_type: str = "leap",
    embodiment_type: str = "right",
    task: str = "scissors",
    data_id: int = 0,
    trajectory_path: str | None = None,
    model_path: str | None = None,
    output_path: str | None = None,
    output_dir: str | None = None,
    robot_joint_count: int | None = None,
    dexmachina: bool = False,
    start_idx: int = 0,
    end_idx: int = -1,
):
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    if trajectory_path is None:
        trajectory_name = (
            f"trajectory_dexmachina_minimal_{robot_type}.npz"
            if dexmachina
            else "trajectory_kinematic_mink.npz"
        )
        trajectory_path = os.path.join(processed_dir, trajectory_name)
    if model_path is None:
        model_path = os.path.join(processed_dir, "..", "scene.xml")
    if output_dir is None:
        output_dir = os.path.join(Path(__file__).resolve().parent, "mink")
    if output_path is None:
        output_name = (
            f"dexmachina_retargeted_{robot_type}_{task}.h5"
            if dexmachina
            else f"mink_retargeted_{robot_type}_{task}.h5"
        )
        output_path = os.path.join(output_dir, output_name)

    trajectory_path = os.path.abspath(trajectory_path)
    model_path = os.path.abspath(model_path)
    output_path = os.path.abspath(output_path)

    source_name = "DexMachina" if dexmachina else "Mink"
    loguru.logger.info(f"Loading {source_name} trajectory from {trajectory_path}")
    with np.load(trajectory_path) as trajectory:
        qpos = trajectory["qpos"]
        frequency = (
            float(trajectory["frequency"])
            if "frequency" in trajectory
            else np.nan
        )
        qpos = qpos.reshape(-1, qpos.shape[-1])
        frame_slice = slice(start_idx, None if end_idx == -1 else end_idx)
        qpos = qpos[frame_slice]
        contact = (
            load_contact_data(trajectory, frame_slice, qpos.shape[0])
            if dexmachina
            else None
        )

    h5_data = split_mink_qpos(
        qpos=qpos,
        robot_type=robot_type,
        robot_joint_count=robot_joint_count,
    )
    if robot_type in DEFAULT_ROBOT_KEYPOINT_COUNTS:
        robot_keypoints, robot_keypoint_names = compute_robot_keypoints(
            qpos=qpos,
            model_path=model_path,
            robot_type=robot_type,
            embodiment_type=embodiment_type,
            robot_joint_count=robot_joint_count,
        )
        h5_data["robot_keypoints"] = robot_keypoints
        loguru.logger.info(
            f"Computed robot_keypoints with shape {robot_keypoints.shape} "
            f"from {model_path}."
        )
    else:
        robot_keypoint_names = []

    attrs = {
        "source_npz": trajectory_path,
        "model_path": model_path,
        "dataset_name": dataset_name,
        "robot_type": robot_type,
        "embodiment_type": embodiment_type,
        "task": task,
        "qpos_layout": (
            "robot_pos_xyz, robot_euler_XYZ, robot_joints, "
            "object_pos_xyz, object_quat_wxyz, optional_object_joint"
        ),
    }
    if not np.isnan(frequency):
        attrs["frequency"] = frequency
        attrs["dt"] = 1.0 / frequency
    if robot_keypoint_names:
        attrs["robot_keypoint_order"] = ",".join(robot_keypoint_names)
        attrs["robot_keypoint_source"] = (
            "MuJoCo wrist site/body position followed by hand joint anchors "
            "in qpos order."
        )

    write_h5(output_path, h5_data, attrs, contact=contact)
    loguru.logger.info(
        "Saved Isaac/DexRL retargeted H5 to "
        f"{output_path} with {qpos.shape[0]} frames and "
        f"{h5_data['robot_joints'].shape[1]} robot joints."
    )


if __name__ == "__main__":
    tyro.cli(main)
