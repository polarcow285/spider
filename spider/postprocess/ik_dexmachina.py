"""
Collision-aware MuJoCo postprocess pass for DexMachina-style retargeting.

Input format: npz file named like ``trajectory_kinematic_{robot_type}.npz``.
The qpos layout is expected to match ik_mink/isaac.py:

    [robot_pos_xyz(3), robot_euler_XYZ(3), robot_finger_joints,
     object_pos_xyz(3), object_quat_wxyz(4), optional_object_joint]

This script runs a DexMachina-style collision-aware retargeting pass: robot qpos
values are used as absolute position-actuator targets, the object state is
pinned to the IK/demo state, contacts are enabled, and every frame can be solved
independently. Like DexMachina, the default saved replay keys come from a second
single-environment controlled rollout, while the strict collision-repaired
achieved qpos is also stored for diagnostics and contact-aware rewards.

Output format: h5 file ``dexmachina_retargeted_{robot_type}_{task}.h5`` saved
by default in ``spider/postprocess/mink/`` and compatible with
``replay_retargeted_traj.py`` / ``dexrl.data.arctic.load_retargeted_traj``.

Required keys:
    object_pos: shape=(T, 3), float64
    object_quat: shape=(T, 4), float64, wxyz
    object_joint: shape=(T,), float64, optional when qpos contains one
    robot_pos: shape=(T, 3), float64
    robot_quat: shape=(T, 4), float64, wxyz
    robot_euler_XYZ: shape=(T, 3), float64, intrinsic XYZ
    robot_joints: shape=(T, n_hand_dof), float64

Additional DexMachina-style contact keys:
    contact_pos: shape=(T, num_object_parts, num_hand_links, 3), float64
    contact_mask: shape=(T, num_object_parts, num_hand_links), bool
    contact_links: shape=(T, num_object_parts, num_hand_links, 4), float64
        xyz plus DexMachina-style part id (1=top, 2=bottom when available)
    achieved_qpos: shape=(T, nq), float64, collision-aware achieved state
    target_qpos: shape=(T, nq), float64, absolute controller targets
    rollout_qpos: shape=(T, nq), float64, stage-2 controlled rollout state

The optional replay viewer can draw contact points from contact_pos/contact_mask
and active collision-pair lines in either a separate transparent contact view or
as an overlay on the regular MuJoCo viewer.

Example:
    python spider/postprocess/ik_dexmachina.py --task scissors --embodiment-type right --dataset-dir example_datasets --dataset-name arctic --robot-type leap
"""

from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import h5py
import loguru
import mujoco
import mujoco.viewer
import numpy as np
import tyro
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from spider.io import get_processed_data_dir
from spider.postprocess.isaac import split_mink_qpos


PART_ID_BY_NAME = {
    "top": 1,
    "bottom": 2,
}

CONTACT_PART_COLORS = np.array(
    [
        [1.0, 0.25, 0.05, 1.0],
        [0.05, 0.7, 1.0, 1.0],
        [0.7, 0.2, 1.0, 1.0],
        [0.2, 0.9, 0.35, 1.0],
    ],
    dtype=np.float32,
)

_WORKER_CONTEXT: "RetargetContext | None" = None


@dataclass(frozen=True)
class SceneContacts:
    part_names: list[str]
    part_geom_ids: list[np.ndarray]
    part_ids: np.ndarray
    hand_link_body_ids: np.ndarray
    hand_link_names: list[str]
    hand_geom_to_link: dict[int, int]
    object_geom_to_part: dict[int, int]


@dataclass(frozen=True)
class RetargetContext:
    model: mujoco.MjModel
    actuator_qpos_addr: np.ndarray
    actuator_ctrlrange: np.ndarray
    actuator_ctrllimited: np.ndarray
    object_qpos_start: int
    frozen_joint_slices: tuple[tuple[slice, slice], ...]
    contacts: SceneContacts


def joint_qpos_width(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def joint_qvel_width(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    return 1


def mj_name(
    model: mujoco.MjModel,
    obj_type: mujoco.mjtObj,
    obj_id: int,
) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return "" if name is None else name


def descendants_of(model: mujoco.MjModel, root_body_id: int) -> set[int]:
    descendants = {root_body_id}
    for body_id in range(root_body_id + 1, model.nbody):
        parent = int(model.body_parentid[body_id])
        while parent != 0 and parent != body_id:
            if parent == root_body_id:
                descendants.add(body_id)
                break
            parent = int(model.body_parentid[parent])
    return descendants


def nearest_ancestor_in_set(
    model: mujoco.MjModel,
    body_id: int,
    candidates: set[int],
) -> int | None:
    current = body_id
    while current != 0:
        if current in candidates:
            return current
        current = int(model.body_parentid[current])
    return 0 if 0 in candidates else None


def is_collision_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    return (
        int(model.geom_contype[geom_id]) != 0
        or int(model.geom_conaffinity[geom_id]) != 0
    )


def is_visual_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).lower()
    return int(model.geom_group[geom_id]) == 1 or "visual" in name


def build_actuator_qpos_map(model: mujoco.MjModel) -> np.ndarray:
    qpos_addr = np.full(model.nu, -1, dtype=np.int64)
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        qpos_addr[actuator_id] = int(model.jnt_qposadr[joint_id])
    return qpos_addr


def infer_object_qpos_start(
    model: mujoco.MjModel,
    actuator_qpos_addr: np.ndarray,
    object_qpos_start: int | None,
) -> int:
    if object_qpos_start is not None:
        return object_qpos_start
    valid_addr = actuator_qpos_addr[actuator_qpos_addr >= 0]
    if valid_addr.size == 0:
        raise ValueError("Could not infer robot qpos span because no joint actuators were found.")
    return int(valid_addr.max()) + 1


def apply_wrist_stiffness_scale(
    model: mujoco.MjModel,
    actuator_qpos_addr: np.ndarray,
    object_qpos_start: int,
    wrist_stiffness_scale: float,
) -> np.ndarray:
    """Scale MuJoCo position-servo gains for the wrist/root qpos block."""
    scale = float(wrist_stiffness_scale)
    if scale <= 0.0:
        raise ValueError("wrist_stiffness_scale must be positive.")
    if np.isclose(scale, 1.0):
        return np.array([], dtype=np.int64)

    wrist_qpos_end = min(6, object_qpos_start)
    if wrist_qpos_end <= 0:
        return np.array([], dtype=np.int64)

    actuator_ids = np.flatnonzero(
        (actuator_qpos_addr >= 0) & (actuator_qpos_addr < wrist_qpos_end)
    )
    servo_gain_scale = min(scale, 4.0)
    for actuator_id in actuator_ids:
        actuator_id = int(actuator_id)
        model.actuator_gainprm[actuator_id, 0] *= servo_gain_scale
        model.actuator_biasprm[actuator_id, 1] *= servo_gain_scale
        model.actuator_biasprm[actuator_id, 2] *= servo_gain_scale
    return actuator_ids.astype(np.int64)


def build_frozen_joint_slices(
    model: mujoco.MjModel,
    object_qpos_start: int,
) -> tuple[tuple[slice, slice], ...]:
    slices: list[tuple[slice, slice]] = []
    for joint_id in range(model.njnt):
        qadr = int(model.jnt_qposadr[joint_id])
        if qadr < object_qpos_start:
            continue
        joint_type = int(model.jnt_type[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])
        qwidth = joint_qpos_width(joint_type)
        dwidth = joint_qvel_width(joint_type)
        slices.append((slice(qadr, qadr + qwidth), slice(dadr, dadr + dwidth)))
    return tuple(slices)


def object_part_sort_key(item: tuple[int, str]) -> tuple[int, int]:
    body_id, name = item
    lname = name.lower()
    if "top" in lname:
        return (0, body_id)
    if "bottom" in lname:
        return (1, body_id)
    return (2, body_id)


def discover_object_part_roots(
    model: mujoco.MjModel,
    object_qpos_start: int,
    max_object_parts: int | None,
) -> list[int]:
    roots: list[int] = []
    for body_id in range(model.nbody):
        for offset in range(int(model.body_jntnum[body_id])):
            joint_id = int(model.body_jntadr[body_id]) + offset
            if int(model.jnt_qposadr[joint_id]) >= object_qpos_start:
                roots.append(body_id)
                break

    if not roots:
        return []

    named_roots = [(body_id, mj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)) for body_id in roots]
    named_roots.sort(key=object_part_sort_key)
    roots = [body_id for body_id, _ in named_roots]
    if max_object_parts is not None and max_object_parts > 0:
        roots = roots[:max_object_parts]
    return roots


def build_scene_contacts(
    model: mujoco.MjModel,
    object_qpos_start: int,
    max_object_parts: int | None,
) -> SceneContacts:
    part_roots = discover_object_part_roots(
        model,
        object_qpos_start=object_qpos_start,
        max_object_parts=max_object_parts,
    )
    part_root_set = set(part_roots)
    part_names = [mj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in part_roots]
    part_ids = np.array(
        [PART_ID_BY_NAME.get(name.lower(), idx + 1) for idx, name in enumerate(part_names)],
        dtype=np.float64,
    )

    part_geom_ids: list[list[int]] = [[] for _ in part_roots]
    object_geom_to_part: dict[int, int] = {}
    part_index_by_root = {body_id: idx for idx, body_id in enumerate(part_roots)}
    for geom_id in range(model.ngeom):
        if not is_collision_geom(model, geom_id):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        ancestor = nearest_ancestor_in_set(model, body_id, part_root_set)
        if ancestor is None:
            continue
        part_index = part_index_by_root[ancestor]
        part_geom_ids[part_index].append(geom_id)
        object_geom_to_part[geom_id] = part_index

    robot_body_ids: set[int] = set()
    for body_id in range(model.nbody):
        has_robot_joint = False
        for offset in range(int(model.body_jntnum[body_id])):
            joint_id = int(model.body_jntadr[body_id]) + offset
            if int(model.jnt_qposadr[joint_id]) < object_qpos_start:
                has_robot_joint = True
                break
        if has_robot_joint:
            robot_body_ids.update(descendants_of(model, body_id))

    hand_geom_to_link: dict[int, int] = {}
    hand_link_body_ids: list[int] = []
    link_index_by_body: dict[int, int] = {}
    for geom_id in range(model.ngeom):
        if geom_id in object_geom_to_part:
            continue
        if not is_collision_geom(model, geom_id):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        if body_id not in robot_body_ids:
            continue
        if body_id not in link_index_by_body:
            link_index_by_body[body_id] = len(hand_link_body_ids)
            hand_link_body_ids.append(body_id)
        hand_geom_to_link[geom_id] = link_index_by_body[body_id]

    hand_link_names = [
        mj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in hand_link_body_ids
    ]

    if not part_geom_ids:
        loguru.logger.warning("No object collision geoms were discovered; contact_pos will be empty.")
    if not hand_link_body_ids:
        loguru.logger.warning("No hand collision geoms were discovered; contact_pos will be empty.")

    return SceneContacts(
        part_names=part_names,
        part_geom_ids=[np.array(ids, dtype=np.int64) for ids in part_geom_ids],
        part_ids=part_ids,
        hand_link_body_ids=np.array(hand_link_body_ids, dtype=np.int64),
        hand_link_names=hand_link_names,
        hand_geom_to_link=hand_geom_to_link,
        object_geom_to_part=object_geom_to_part,
    )


def configure_runtime_collisions(
    model: mujoco.MjModel,
    enable_collision: bool,
    enable_self_collision: bool,
    force_collision_masks: bool,
    clear_contact_exclusions: bool,
    collide_parent_child: bool,
    collision_margin: float,
) -> tuple[int, int]:
    """Enable MuJoCo collision filtering for the retargeting runtime model.

    The generated scenes often contain explicit hand self-collision excludes.
    MuJoCo compiles those into ``exclude_signature``; clearing those signatures
    makes inter-finger/palm contacts visible without editing the source XML.
    Parent-child filtering is kept unless explicitly disabled because adjacent
    articulated links commonly have overlapping collision proxies at the joint.
    """
    if enable_collision:
        model.opt.disableflags = int(model.opt.disableflags) & ~int(
            mujoco.mjtDisableBit.mjDSBL_CONTACT
        )
        model.opt.disableflags = int(model.opt.disableflags) & ~int(
            mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        )
    else:
        model.opt.disableflags = int(model.opt.disableflags) | int(
            mujoco.mjtDisableBit.mjDSBL_CONTACT
        )

    if collide_parent_child:
        model.opt.disableflags = int(model.opt.disableflags) | int(
            mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
        )
    else:
        model.opt.disableflags = int(model.opt.disableflags) & ~int(
            mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
        )

    forced_geom_count = 0
    if enable_collision and force_collision_masks:
        for geom_id in range(model.ngeom):
            if is_visual_geom(model, geom_id):
                continue
            if int(model.geom_contype[geom_id]) != 1:
                model.geom_contype[geom_id] = 1
                forced_geom_count += 1
            if int(model.geom_conaffinity[geom_id]) != 1:
                model.geom_conaffinity[geom_id] = 1
                forced_geom_count += 1
            if collision_margin > 0:
                model.geom_margin[geom_id] = max(
                    float(model.geom_margin[geom_id]),
                    float(collision_margin),
                )

    cleared_exclude_count = 0
    if enable_collision and enable_self_collision and clear_contact_exclusions:
        cleared_exclude_count = int(model.nexclude)
        if cleared_exclude_count > 0:
            model.exclude_signature[:] = -1

    return forced_geom_count, cleared_exclude_count


def candidate_mesh_dirs(model_path: str, meshdir: str | None) -> list[Path]:
    model_dir = Path(model_path).resolve().parent
    candidates: list[Path] = []

    if meshdir:
        mesh_path = Path(meshdir)
        if not mesh_path.is_absolute():
            mesh_path = model_dir / mesh_path
        candidates.append(mesh_path.resolve())

    for parent in (model_dir, *model_dir.parents):
        candidates.append((parent / "assets").resolve())
        if parent.name == "processed":
            candidates.append((parent / "custom" / "assets").resolve())

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate))
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def load_mj_model(model_path: str) -> mujoco.MjModel:
    try:
        return mujoco.MjModel.from_xml_path(model_path)
    except ValueError as first_error:
        xml_text = Path(model_path).read_text(encoding="utf-8")
        root = ET.fromstring(xml_text)
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(root, "compiler")
        original_meshdir = compiler.get("meshdir")

        for mesh_dir in candidate_mesh_dirs(model_path, original_meshdir):
            compiler.set("meshdir", str(mesh_dir))
            try:
                model = mujoco.MjModel.from_xml_string(
                    ET.tostring(root, encoding="unicode")
                )
            except ValueError:
                continue
            loguru.logger.info(
                f"Loaded MuJoCo XML with fallback meshdir {mesh_dir}."
            )
            return model

        raise first_error


def load_model_for_retargeting(
    model_path: str,
    sim_dt: float,
    solver_iterations: int,
    solver_tolerance: float,
    integrator: str,
    enable_collision: bool,
    enable_self_collision: bool,
    force_collision_masks: bool,
    clear_contact_exclusions: bool,
    collide_parent_child: bool,
    collision_margin: float,
) -> mujoco.MjModel:
    model = load_mj_model(model_path)
    model.opt.timestep = sim_dt
    integrators = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }
    try:
        model.opt.integrator = integrators[integrator.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown integrator `{integrator}`. Choose one of {sorted(integrators)}."
        ) from exc
    model.opt.iterations = max(int(model.opt.iterations), int(solver_iterations))
    if solver_tolerance > 0:
        model.opt.tolerance = min(float(model.opt.tolerance), float(solver_tolerance))
    forced_geom_count, cleared_exclude_count = configure_runtime_collisions(
        model=model,
        enable_collision=enable_collision,
        enable_self_collision=enable_self_collision,
        force_collision_masks=force_collision_masks,
        clear_contact_exclusions=clear_contact_exclusions,
        collide_parent_child=collide_parent_child,
        collision_margin=collision_margin,
    )
    if enable_collision:
        loguru.logger.info(
            "MuJoCo collisions enabled "
            f"(forced {forced_geom_count} geom mask values, "
            f"cleared {cleared_exclude_count} contact excludes, "
            f"collide_parent_child={collide_parent_child}, "
            f"collision_margin={collision_margin})."
        )
    return model


def build_context(
    model_path: str,
    sim_dt: float,
    solver_iterations: int,
    solver_tolerance: float,
    integrator: str,
    object_qpos_start: int | None,
    max_object_parts: int | None,
    enable_collision: bool,
    enable_self_collision: bool,
    force_collision_masks: bool,
    clear_contact_exclusions: bool,
    collide_parent_child: bool,
    collision_margin: float,
    wrist_stiffness_scale: float,
) -> RetargetContext:
    model = load_model_for_retargeting(
        model_path=model_path,
        sim_dt=sim_dt,
        solver_iterations=solver_iterations,
        solver_tolerance=solver_tolerance,
        integrator=integrator,
        enable_collision=enable_collision,
        enable_self_collision=enable_self_collision,
        force_collision_masks=force_collision_masks,
        clear_contact_exclusions=clear_contact_exclusions,
        collide_parent_child=collide_parent_child,
        collision_margin=collision_margin,
    )
    actuator_qpos_addr = build_actuator_qpos_map(model)
    inferred_object_qpos_start = infer_object_qpos_start(
        model,
        actuator_qpos_addr=actuator_qpos_addr,
        object_qpos_start=object_qpos_start,
    )
    wrist_actuator_ids = apply_wrist_stiffness_scale(
        model=model,
        actuator_qpos_addr=actuator_qpos_addr,
        object_qpos_start=inferred_object_qpos_start,
        wrist_stiffness_scale=wrist_stiffness_scale,
    )
    if wrist_actuator_ids.size > 0:
        loguru.logger.info(
            "Enabled wrist/root tracking scale "
            f"{wrist_stiffness_scale:g} for {wrist_actuator_ids.size} actuators "
            f"(servo gains capped at {min(float(wrist_stiffness_scale), 4.0):g}x)."
        )
    frozen_joint_slices = build_frozen_joint_slices(
        model,
        object_qpos_start=inferred_object_qpos_start,
    )
    contacts = build_scene_contacts(
        model,
        object_qpos_start=inferred_object_qpos_start,
        max_object_parts=max_object_parts,
    )
    return RetargetContext(
        model=model,
        actuator_qpos_addr=actuator_qpos_addr,
        actuator_ctrlrange=model.actuator_ctrlrange.copy(),
        actuator_ctrllimited=model.actuator_ctrllimited.copy(),
        object_qpos_start=inferred_object_qpos_start,
        frozen_joint_slices=frozen_joint_slices,
        contacts=contacts,
    )


def set_position_targets(
    context: RetargetContext,
    data: mujoco.MjData,
    qpos_target: np.ndarray,
) -> None:
    ctrl = np.zeros(context.model.nu, dtype=np.float64)
    valid = context.actuator_qpos_addr >= 0
    ctrl[valid] = qpos_target[context.actuator_qpos_addr[valid]]
    limited = context.actuator_ctrllimited.astype(bool) & valid
    ctrl[limited] = np.clip(
        ctrl[limited],
        context.actuator_ctrlrange[limited, 0],
        context.actuator_ctrlrange[limited, 1],
    )
    data.ctrl[:] = ctrl


def wrist_tracking_alpha(wrist_stiffness_scale: float) -> float:
    if wrist_stiffness_scale <= 1.0:
        return 0.0
    return float(np.clip(1.0 - 1.0 / wrist_stiffness_scale, 0.0, 0.95))


def apply_wrist_tracking_correction(
    context: RetargetContext,
    data: mujoco.MjData,
    qpos_target: np.ndarray,
    wrist_stiffness_scale: float,
) -> None:
    """Make the virtual wrist/root track its target when servo gains are insufficient.

    The wrist/root joints are mocap-style virtual joints, not physical hand
    joints. For H5 qpos replay we care about the commanded wrist pose, so a
    scale above 1.0 applies a bounded post-step correction in addition to the
    MuJoCo position servo gain scaling.
    """
    alpha = wrist_tracking_alpha(wrist_stiffness_scale)
    if alpha <= 0.0:
        return
    wrist_qpos_end = min(6, context.object_qpos_start, data.qpos.size)
    if wrist_qpos_end <= 0:
        return
    data.qpos[:wrist_qpos_end] = (
        (1.0 - alpha) * data.qpos[:wrist_qpos_end]
        + alpha * qpos_target[:wrist_qpos_end]
    )
    wrist_qvel_end = min(6, data.qvel.size)
    data.qvel[:wrist_qvel_end] = 0.0


def pin_frozen_joints(
    context: RetargetContext,
    data: mujoco.MjData,
    qpos_target: np.ndarray,
) -> None:
    for qslice, vslice in context.frozen_joint_slices:
        data.qpos[qslice] = qpos_target[qslice]
        data.qvel[vslice] = 0.0


def clamp_limited_robot_joints(
    context: RetargetContext,
    data: mujoco.MjData,
) -> None:
    model = context.model
    for joint_id in range(model.njnt):
        qadr = int(model.jnt_qposadr[joint_id])
        if qadr >= context.object_qpos_start:
            continue
        if not bool(model.jnt_limited[joint_id]):
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        lower, upper = model.jnt_range[joint_id]
        data.qpos[qadr] = np.clip(data.qpos[qadr], lower, upper)


def clamp_robot_qpos_targets(
    context: RetargetContext,
    qpos: np.ndarray,
) -> np.ndarray:
    """Clamp absolute qpos targets the same way DexMachina clamps actions."""
    clipped = np.asarray(qpos, dtype=np.float64).copy()
    model = context.model

    valid = context.actuator_qpos_addr >= 0
    limited_actuators = context.actuator_ctrllimited.astype(bool) & valid
    for actuator_id in np.flatnonzero(limited_actuators):
        qadr = int(context.actuator_qpos_addr[actuator_id])
        if qadr >= context.object_qpos_start:
            continue
        lower, upper = context.actuator_ctrlrange[actuator_id]
        clipped[..., qadr] = np.clip(clipped[..., qadr], lower, upper)

    for joint_id in range(model.njnt):
        qadr = int(model.jnt_qposadr[joint_id])
        if qadr >= context.object_qpos_start:
            continue
        if not bool(model.jnt_limited[joint_id]):
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        lower, upper = model.jnt_range[joint_id]
        clipped[..., qadr] = np.clip(clipped[..., qadr], lower, upper)

    return clipped


def contact_arrays_empty(context: RetargetContext) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_parts = len(context.contacts.part_names)
    num_links = len(context.contacts.hand_link_names)
    contact_pos = np.zeros((num_parts, num_links, 3), dtype=np.float64)
    contact_mask = np.zeros((num_parts, num_links), dtype=bool)
    counts = np.zeros((num_parts, num_links), dtype=np.int64)
    return contact_pos, contact_mask, counts


def add_grouped_contact(
    contact_pos: np.ndarray,
    contact_mask: np.ndarray,
    counts: np.ndarray,
    part_index: int,
    link_index: int,
    pos: np.ndarray,
) -> None:
    count = counts[part_index, link_index]
    contact_pos[part_index, link_index] = (
        contact_pos[part_index, link_index] * count + pos
    ) / (count + 1)
    counts[part_index, link_index] = count + 1
    contact_mask[part_index, link_index] = True


def link_centers(context: RetargetContext, data: mujoco.MjData) -> np.ndarray:
    centers = np.zeros((len(context.contacts.hand_link_body_ids), 3), dtype=np.float64)
    counts = np.zeros(len(context.contacts.hand_link_body_ids), dtype=np.int64)
    for geom_id, link_index in context.contacts.hand_geom_to_link.items():
        centers[link_index] += data.geom_xpos[geom_id]
        counts[link_index] += 1
    valid = counts > 0
    centers[valid] /= counts[valid, None]
    return centers


def part_centers(context: RetargetContext, data: mujoco.MjData) -> np.ndarray:
    centers = np.zeros((len(context.contacts.part_geom_ids), 3), dtype=np.float64)
    for part_index, geom_ids in enumerate(context.contacts.part_geom_ids):
        if geom_ids.size == 0:
            continue
        centers[part_index] = data.geom_xpos[geom_ids].mean(axis=0)
    return centers


def extract_mujoco_contacts(
    context: RetargetContext,
    data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray]:
    contact_pos, contact_mask, counts = contact_arrays_empty(context)

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        part_index = context.contacts.object_geom_to_part.get(geom1)
        link_index = context.contacts.hand_geom_to_link.get(geom2)
        if part_index is None or link_index is None:
            part_index = context.contacts.object_geom_to_part.get(geom2)
            link_index = context.contacts.hand_geom_to_link.get(geom1)
        if part_index is None or link_index is None:
            continue

        add_grouped_contact(
            contact_pos,
            contact_mask,
            counts,
            part_index=part_index,
            link_index=link_index,
            pos=np.asarray(contact.pos, dtype=np.float64),
        )

    return contact_pos, contact_mask


def fill_contacts_from_ik_fallback(
    context: RetargetContext,
    data: mujoco.MjData,
    contact_pos: np.ndarray,
    contact_mask: np.ndarray,
    ik_contact: np.ndarray | None,
    ik_contact_pos: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if ik_contact is None or ik_contact_pos is None:
        return contact_pos, contact_mask
    if ik_contact_pos.size == 0 or len(context.contacts.hand_link_body_ids) == 0:
        return contact_pos, contact_mask
    if len(context.contacts.part_geom_ids) == 0:
        return contact_pos, contact_mask

    flat_mask = np.asarray(ik_contact).reshape(-1).astype(bool)
    flat_pos = np.asarray(ik_contact_pos, dtype=np.float64).reshape(-1, 3)
    if flat_mask.size != flat_pos.shape[0]:
        return contact_pos, contact_mask

    centers_link = link_centers(context, data)
    centers_part = part_centers(context, data)
    counts = contact_mask.astype(np.int64)

    for pos in flat_pos[flat_mask]:
        if centers_link.shape[0] == 0 or centers_part.shape[0] == 0:
            continue
        link_index = int(np.argmin(np.linalg.norm(centers_link - pos[None], axis=1)))
        part_index = int(np.argmin(np.linalg.norm(centers_part - pos[None], axis=1)))
        if contact_mask[part_index, link_index]:
            continue
        add_grouped_contact(
            contact_pos,
            contact_mask,
            counts,
            part_index=part_index,
            link_index=link_index,
            pos=pos,
        )

    return contact_pos, contact_mask


def is_relevant_collision_pair(
    context: RetargetContext,
    geom1: int,
    geom2: int,
) -> bool:
    geom1_is_hand = geom1 in context.contacts.hand_geom_to_link
    geom2_is_hand = geom2 in context.contacts.hand_geom_to_link
    geom1_is_object = geom1 in context.contacts.object_geom_to_part
    geom2_is_object = geom2 in context.contacts.object_geom_to_part
    return (
        (geom1_is_hand and geom2_is_object)
        or (geom2_is_hand and geom1_is_object)
        or (geom1_is_hand and geom2_is_hand)
    )


def robot_qpos_bounds(context: RetargetContext) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(context.object_qpos_start, -np.inf, dtype=np.float64)
    upper = np.full(context.object_qpos_start, np.inf, dtype=np.float64)
    model = context.model
    for joint_id in range(model.njnt):
        qadr = int(model.jnt_qposadr[joint_id])
        if qadr >= context.object_qpos_start:
            continue
        if not bool(model.jnt_limited[joint_id]):
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        lower[qadr], upper[qadr] = model.jnt_range[joint_id]
    return lower, upper


def active_collision_pairs(
    context: RetargetContext,
    data: mujoco.MjData,
    qpos: np.ndarray,
    pair_margin: float,
) -> list[tuple[int, int]]:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(context.model, data)

    pairs: set[tuple[int, int]] = set()
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if float(contact.dist) >= pair_margin:
            continue
        if not is_relevant_collision_pair(context, geom1, geom2):
            continue
        pairs.add(tuple(sorted((geom1, geom2))))
    return sorted(pairs)


def collision_pair_distances(
    context: RetargetContext,
    data: mujoco.MjData,
    qpos: np.ndarray,
    pairs: list[tuple[int, int]],
    default_distance: float,
) -> np.ndarray:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(context.model, data)

    fromto = np.zeros(6, dtype=np.float64)
    distances = np.empty(len(pairs), dtype=np.float64)
    for pair_index, (geom1, geom2) in enumerate(pairs):
        distances[pair_index] = mujoco.mj_geomDistance(
            context.model,
            data,
            geom1,
            geom2,
            default_distance,
            fromto,
        )
    return distances


def project_one_frame_collision_aware(
    context: RetargetContext,
    qpos_target: np.ndarray,
    pair_margin: float,
    safety_margin: float,
    tracking_weight: float,
    wrist_stiffness_scale: float,
    collision_weight: float,
    max_nfev: int,
    outer_iterations: int,
    ik_contact: np.ndarray | None,
    ik_contact_pos: np.ndarray | None,
    use_ik_contact_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project one target frame to the nearest non-penetrating robot qpos.

    This is the IK-style complement to the DexMachina absolute-control rollout:
    object qpos is fixed, robot qpos is optimized to stay near the retargeted
    target, and only hand-object / hand-self distances enter the collision term.
    """
    model = context.model
    data = mujoco.MjData(model)
    lower, upper = robot_qpos_bounds(context)
    robot_width = context.object_qpos_start

    projected = qpos_target.copy()
    x = projected[:robot_width].copy()
    pair_set: set[tuple[int, int]] = set()
    tracking_weights = np.full(robot_width, tracking_weight, dtype=np.float64)
    tracking_weights[: min(6, robot_width)] *= wrist_stiffness_scale

    for _ in range(max(1, outer_iterations)):
        projected[:robot_width] = x
        projected[robot_width:] = qpos_target[robot_width:]
        pairs = active_collision_pairs(
            context,
            data,
            projected,
            pair_margin=pair_margin,
        )
        pair_set.update(pairs)
        if not pair_set:
            break

        pair_list = sorted(pair_set)

        def residual(x_candidate: np.ndarray) -> np.ndarray:
            candidate_qpos = qpos_target.copy()
            candidate_qpos[:robot_width] = x_candidate
            distances = collision_pair_distances(
                context,
                data,
                candidate_qpos,
                pair_list,
                default_distance=pair_margin,
            )
            tracking_residual = np.sqrt(tracking_weights) * (
                x_candidate - qpos_target[:robot_width]
            )
            collision_residual = np.sqrt(collision_weight) * np.minimum(
                0.0,
                distances - safety_margin,
            )
            return np.concatenate([tracking_residual, collision_residual])

        result = least_squares(
            residual,
            x,
            bounds=(lower, upper),
            max_nfev=max(1, max_nfev),
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )
        x = result.x

        projected[:robot_width] = x
        projected[robot_width:] = qpos_target[robot_width:]
        new_pairs = active_collision_pairs(
            context,
            data,
            projected,
            pair_margin=max(pair_margin, safety_margin),
        )
        new_pair_set = set(new_pairs)
        if new_pair_set.issubset(pair_set):
            break
        pair_set.update(new_pair_set)

    projected[:robot_width] = x
    projected[robot_width:] = qpos_target[robot_width:]
    data.qpos[:] = projected
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    contact_pos, contact_mask = extract_mujoco_contacts(context, data)
    if use_ik_contact_fallback:
        contact_pos, contact_mask = fill_contacts_from_ik_fallback(
            context,
            data,
            contact_pos=contact_pos,
            contact_mask=contact_mask,
            ik_contact=ik_contact,
            ik_contact_pos=ik_contact_pos,
        )
    return projected, contact_pos, contact_mask


def settle_one_frame(
    context: RetargetContext,
    qpos_target: np.ndarray,
    settle_steps: int,
    collision_relax_steps: int,
    max_abs_qpos: float,
    wrist_stiffness_scale: float,
    ik_contact: np.ndarray | None,
    ik_contact_pos: np.ndarray | None,
    use_ik_contact_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = context.model
    if qpos_target.shape[0] != model.nq:
        raise ValueError(f"qpos width {qpos_target.shape[0]} does not match model.nq {model.nq}.")

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_target
    data.qvel[:] = 0.0
    pin_frozen_joints(context, data, qpos_target)
    set_position_targets(context, data, qpos_target)
    mujoco.mj_forward(model, data)

    for _ in range(settle_steps):
        pin_frozen_joints(context, data, qpos_target)
        set_position_targets(context, data, qpos_target)
        prev_qpos = data.qpos.copy()
        mujoco.mj_step(model, data)
        pin_frozen_joints(context, data, qpos_target)
        apply_wrist_tracking_correction(
            context,
            data,
            qpos_target,
            wrist_stiffness_scale=wrist_stiffness_scale,
        )
        clamp_limited_robot_joints(context, data)
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or np.max(np.abs(data.qpos)) > max_abs_qpos
        ):
            data.qpos[:] = prev_qpos
            data.qvel[:] = 0.0
            break
        mujoco.mj_forward(model, data)

    for _ in range(collision_relax_steps):
        pin_frozen_joints(context, data, qpos_target)
        relaxed_target = data.qpos.copy()
        relaxed_target[context.object_qpos_start :] = qpos_target[
            context.object_qpos_start :
        ]
        set_position_targets(context, data, relaxed_target)
        prev_qpos = data.qpos.copy()
        mujoco.mj_step(model, data)
        pin_frozen_joints(context, data, qpos_target)
        apply_wrist_tracking_correction(
            context,
            data,
            qpos_target,
            wrist_stiffness_scale=wrist_stiffness_scale,
        )
        clamp_limited_robot_joints(context, data)
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or np.max(np.abs(data.qpos)) > max_abs_qpos
        ):
            data.qpos[:] = prev_qpos
            data.qvel[:] = 0.0
            break
        mujoco.mj_forward(model, data)

    contact_pos, contact_mask = extract_mujoco_contacts(context, data)
    if use_ik_contact_fallback:
        contact_pos, contact_mask = fill_contacts_from_ik_fallback(
            context,
            data,
            contact_pos=contact_pos,
            contact_mask=contact_mask,
            ik_contact=ik_contact,
            ik_contact_pos=ik_contact_pos,
        )

    settled_qpos = data.qpos.copy()
    pin_qpos_target = qpos_target[context.object_qpos_start :]
    settled_qpos[context.object_qpos_start :] = pin_qpos_target
    return settled_qpos, contact_pos, contact_mask


def _init_worker(
    model_path: str,
    sim_dt: float,
    solver_iterations: int,
    solver_tolerance: float,
    integrator: str,
    object_qpos_start: int | None,
    max_object_parts: int | None,
    enable_collision: bool,
    enable_self_collision: bool,
    force_collision_masks: bool,
    clear_contact_exclusions: bool,
    collide_parent_child: bool,
    collision_margin: float,
    wrist_stiffness_scale: float,
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = build_context(
        model_path=model_path,
        sim_dt=sim_dt,
        solver_iterations=solver_iterations,
        solver_tolerance=solver_tolerance,
        integrator=integrator,
        object_qpos_start=object_qpos_start,
        max_object_parts=max_object_parts,
        enable_collision=enable_collision,
        enable_self_collision=enable_self_collision,
        force_collision_masks=force_collision_masks,
        clear_contact_exclusions=clear_contact_exclusions,
        collide_parent_child=collide_parent_child,
        collision_margin=collision_margin,
        wrist_stiffness_scale=wrist_stiffness_scale,
    )


def _settle_frame_worker(
    payload: tuple[int, np.ndarray, int, int, float, float, Any, Any, bool],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context was not initialized.")
    (
        frame_idx,
        qpos,
        settle_steps,
        collision_relax_steps,
        max_abs_qpos,
        wrist_stiffness_scale,
        ik_contact,
        ik_contact_pos,
        use_ik_contact_fallback,
    ) = payload
    settled_qpos, contact_pos, contact_mask = settle_one_frame(
        _WORKER_CONTEXT,
        qpos_target=qpos,
        settle_steps=settle_steps,
        collision_relax_steps=collision_relax_steps,
        max_abs_qpos=max_abs_qpos,
        wrist_stiffness_scale=wrist_stiffness_scale,
        ik_contact=ik_contact,
        ik_contact_pos=ik_contact_pos,
        use_ik_contact_fallback=use_ik_contact_fallback,
    )
    return frame_idx, settled_qpos, contact_pos, contact_mask


def _project_frame_worker(
    payload: tuple[
        int,
        np.ndarray,
        float,
        float,
        float,
        float,
        float,
        int,
        int,
        Any,
        Any,
        bool,
    ],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context was not initialized.")
    (
        frame_idx,
        qpos_target,
        pair_margin,
        safety_margin,
        tracking_weight,
        wrist_stiffness_scale,
        collision_weight,
        max_nfev,
        outer_iterations,
        ik_contact,
        ik_contact_pos,
        use_ik_contact_fallback,
    ) = payload
    projected_qpos, contact_pos, contact_mask = project_one_frame_collision_aware(
        _WORKER_CONTEXT,
        qpos_target=qpos_target,
        pair_margin=pair_margin,
        safety_margin=safety_margin,
        tracking_weight=tracking_weight,
        wrist_stiffness_scale=wrist_stiffness_scale,
        collision_weight=collision_weight,
        max_nfev=max_nfev,
        outer_iterations=outer_iterations,
        ik_contact=ik_contact,
        ik_contact_pos=ik_contact_pos,
        use_ik_contact_fallback=use_ik_contact_fallback,
    )
    return frame_idx, projected_qpos, contact_pos, contact_mask


def settle_trajectory(
    context: RetargetContext,
    model_path: str,
    qpos: np.ndarray,
    settle_steps: int,
    collision_relax_steps: int,
    num_workers: int,
    sim_dt: float,
    solver_iterations: int,
    solver_tolerance: float,
    integrator: str,
    max_abs_qpos: float,
    object_qpos_start: int | None,
    max_object_parts: int | None,
    enable_collision: bool,
    enable_self_collision: bool,
    force_collision_masks: bool,
    clear_contact_exclusions: bool,
    collide_parent_child: bool,
    collision_margin: float,
    wrist_stiffness_scale: float,
    ik_contact: np.ndarray | None,
    ik_contact_pos: np.ndarray | None,
    use_ik_contact_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=np.float64)
    total_frames = qpos.shape[0]
    num_parts = len(context.contacts.part_names)
    num_links = len(context.contacts.hand_link_names)
    settled_qpos = np.zeros_like(qpos)
    contact_pos = np.zeros((total_frames, num_parts, num_links, 3), dtype=np.float64)
    contact_mask = np.zeros((total_frames, num_parts, num_links), dtype=bool)

    worker_count = effective_worker_count(num_workers, total_frames)
    if worker_count <= 1:
        for frame_idx in range(total_frames):
            if frame_idx % 25 == 0:
                loguru.logger.info(f"Settling frame {frame_idx + 1}/{total_frames}")
            frame_contact = None if ik_contact is None else ik_contact[frame_idx]
            frame_contact_pos = None if ik_contact_pos is None else ik_contact_pos[frame_idx]
            settled, cpos, cmask = settle_one_frame(
                context,
                qpos_target=qpos[frame_idx],
                settle_steps=settle_steps,
                collision_relax_steps=collision_relax_steps,
                max_abs_qpos=max_abs_qpos,
                wrist_stiffness_scale=wrist_stiffness_scale,
                ik_contact=frame_contact,
                ik_contact_pos=frame_contact_pos,
                use_ik_contact_fallback=use_ik_contact_fallback,
            )
            settled_qpos[frame_idx] = settled
            contact_pos[frame_idx] = cpos
            contact_mask[frame_idx] = cmask
        return settled_qpos, contact_pos, contact_mask

    payloads = []
    for frame_idx in range(total_frames):
        frame_contact = None if ik_contact is None else ik_contact[frame_idx]
        frame_contact_pos = None if ik_contact_pos is None else ik_contact_pos[frame_idx]
        payloads.append(
            (
                frame_idx,
                qpos[frame_idx],
                settle_steps,
                collision_relax_steps,
                max_abs_qpos,
                wrist_stiffness_scale,
                frame_contact,
                frame_contact_pos,
                use_ik_contact_fallback,
            )
        )

    loguru.logger.info(
        f"Settling {total_frames} frames with {worker_count} worker processes"
    )
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_worker,
        initargs=(
            model_path,
            sim_dt,
            solver_iterations,
            solver_tolerance,
            integrator,
            object_qpos_start,
            max_object_parts,
            enable_collision,
            enable_self_collision,
            force_collision_masks,
            clear_contact_exclusions,
            collide_parent_child,
            collision_margin,
            wrist_stiffness_scale,
        ),
    ) as executor:
        for done_count, (frame_idx, settled, cpos, cmask) in enumerate(
            executor.map(_settle_frame_worker, payloads),
            start=1,
        ):
            if done_count % 25 == 0 or done_count == total_frames:
                loguru.logger.info(f"Settled {done_count}/{total_frames} frames")
            settled_qpos[frame_idx] = settled
            contact_pos[frame_idx] = cpos
            contact_mask[frame_idx] = cmask

    return settled_qpos, contact_pos, contact_mask


def effective_worker_count(num_workers: int, total_frames: int) -> int:
    if num_workers == 0:
        return max(1, min(total_frames, os.cpu_count() or 1, 8))
    return max(1, min(total_frames, num_workers))


def project_collision_aware_trajectory(
    context: RetargetContext,
    model_path: str,
    qpos: np.ndarray,
    num_workers: int,
    sim_dt: float,
    solver_iterations: int,
    solver_tolerance: float,
    integrator: str,
    object_qpos_start: int | None,
    max_object_parts: int | None,
    enable_collision: bool,
    enable_self_collision: bool,
    force_collision_masks: bool,
    clear_contact_exclusions: bool,
    collide_parent_child: bool,
    collision_margin: float,
    wrist_stiffness_scale: float,
    pair_margin: float,
    safety_margin: float,
    tracking_weight: float,
    collision_weight: float,
    max_nfev: int,
    outer_iterations: int,
    ik_contact: np.ndarray | None,
    ik_contact_pos: np.ndarray | None,
    use_ik_contact_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=np.float64)
    total_frames = qpos.shape[0]
    num_parts = len(context.contacts.part_names)
    num_links = len(context.contacts.hand_link_names)
    projected_qpos = np.zeros_like(qpos)
    contact_pos = np.zeros((total_frames, num_parts, num_links, 3), dtype=np.float64)
    contact_mask = np.zeros((total_frames, num_parts, num_links), dtype=bool)

    worker_count = effective_worker_count(num_workers, total_frames)
    payloads = []
    for frame_idx in range(total_frames):
        frame_contact = None if ik_contact is None else ik_contact[frame_idx]
        frame_contact_pos = None if ik_contact_pos is None else ik_contact_pos[frame_idx]
        payloads.append(
            (
                frame_idx,
                qpos[frame_idx],
                pair_margin,
                safety_margin,
                tracking_weight,
                wrist_stiffness_scale,
                collision_weight,
                max_nfev,
                outer_iterations,
                frame_contact,
                frame_contact_pos,
                use_ik_contact_fallback,
            )
        )

    if worker_count <= 1:
        for frame_idx, payload in enumerate(payloads):
            if frame_idx % 25 == 0:
                loguru.logger.info(
                    f"Projecting collision-aware frame {frame_idx + 1}/{total_frames}"
                )
            (
                _,
                qpos_target,
                payload_pair_margin,
                payload_safety_margin,
                payload_tracking_weight,
                payload_wrist_stiffness_scale,
                payload_collision_weight,
                payload_max_nfev,
                payload_outer_iterations,
                frame_contact,
                frame_contact_pos,
                payload_use_ik_contact_fallback,
            ) = payload
            projected, cpos, cmask = project_one_frame_collision_aware(
                context,
                qpos_target=qpos_target,
                pair_margin=payload_pair_margin,
                safety_margin=payload_safety_margin,
                tracking_weight=payload_tracking_weight,
                wrist_stiffness_scale=payload_wrist_stiffness_scale,
                collision_weight=payload_collision_weight,
                max_nfev=payload_max_nfev,
                outer_iterations=payload_outer_iterations,
                ik_contact=frame_contact,
                ik_contact_pos=frame_contact_pos,
                use_ik_contact_fallback=payload_use_ik_contact_fallback,
            )
            projected_qpos[frame_idx] = projected
            contact_pos[frame_idx] = cpos
            contact_mask[frame_idx] = cmask
        return projected_qpos, contact_pos, contact_mask

    loguru.logger.info(
        f"Projecting {total_frames} collision-aware frames with {worker_count} workers"
    )
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_worker,
        initargs=(
            model_path,
            sim_dt,
            solver_iterations,
            solver_tolerance,
            integrator,
            object_qpos_start,
            max_object_parts,
            enable_collision,
            enable_self_collision,
            force_collision_masks,
            clear_contact_exclusions,
            collide_parent_child,
            collision_margin,
            wrist_stiffness_scale,
        ),
    ) as executor:
        for done_count, (frame_idx, projected, cpos, cmask) in enumerate(
            executor.map(_project_frame_worker, payloads),
            start=1,
        ):
            if done_count % 25 == 0 or done_count == total_frames:
                loguru.logger.info(
                    f"Projected {done_count}/{total_frames} collision-aware frames"
                )
            projected_qpos[frame_idx] = projected
            contact_pos[frame_idx] = cpos
            contact_mask[frame_idx] = cmask

    return projected_qpos, contact_pos, contact_mask


def rollout_controlled_trajectory(
    context: RetargetContext,
    target_qpos: np.ndarray,
    object_qpos_ref: np.ndarray,
    steps_per_frame: int,
    collision_relax_steps: int,
    max_abs_qpos: float,
    wrist_stiffness_scale: float,
    ik_contact: np.ndarray | None,
    ik_contact_pos: np.ndarray | None,
    use_ik_contact_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll controller targets through one sequential MuJoCo simulation."""
    target_qpos = np.asarray(target_qpos, dtype=np.float64)
    object_qpos_ref = np.asarray(object_qpos_ref, dtype=np.float64)
    total_frames = target_qpos.shape[0]
    num_parts = len(context.contacts.part_names)
    num_links = len(context.contacts.hand_link_names)
    rollout_qpos = np.zeros_like(target_qpos)
    contact_pos = np.zeros((total_frames, num_parts, num_links, 3), dtype=np.float64)
    contact_mask = np.zeros((total_frames, num_parts, num_links), dtype=bool)

    model = context.model
    data = mujoco.MjData(model)
    data.qpos[:] = target_qpos[0]
    data.qpos[context.object_qpos_start :] = object_qpos_ref[
        0, context.object_qpos_start :
    ]
    data.qvel[:] = 0.0
    ctrl_qpos = data.qpos.copy()
    pin_frozen_joints(context, data, object_qpos_ref[0])
    clamp_limited_robot_joints(context, data)
    mujoco.mj_forward(model, data)

    for frame_idx in range(total_frames):
        if frame_idx % 25 == 0:
            loguru.logger.info(f"Rolling out frame {frame_idx + 1}/{total_frames}")

        frame_target = target_qpos[frame_idx]
        object_target = object_qpos_ref[frame_idx]
        for _ in range(max(1, steps_per_frame)):
            ctrl_qpos[: context.object_qpos_start] = frame_target[
                : context.object_qpos_start
            ]
            ctrl_qpos[context.object_qpos_start :] = object_target[
                context.object_qpos_start :
            ]

            pin_frozen_joints(context, data, object_target)
            set_position_targets(context, data, ctrl_qpos)
            prev_qpos = data.qpos.copy()
            mujoco.mj_step(model, data)
            pin_frozen_joints(context, data, object_target)
            apply_wrist_tracking_correction(
                context,
                data,
                ctrl_qpos,
                wrist_stiffness_scale=wrist_stiffness_scale,
            )
            clamp_limited_robot_joints(context, data)
            if (
                not np.isfinite(data.qpos).all()
                or not np.isfinite(data.qvel).all()
                or np.max(np.abs(data.qpos)) > max_abs_qpos
            ):
                data.qpos[:] = prev_qpos
                data.qvel[:] = 0.0
                break
            mujoco.mj_forward(model, data)

        pin_frozen_joints(context, data, object_target)
        clamp_limited_robot_joints(context, data)
        mujoco.mj_forward(model, data)

        for _ in range(collision_relax_steps):
            pin_frozen_joints(context, data, object_target)
            relaxed_target = data.qpos.copy()
            relaxed_target[context.object_qpos_start :] = object_target[
                context.object_qpos_start :
            ]
            set_position_targets(context, data, relaxed_target)
            prev_qpos = data.qpos.copy()
            mujoco.mj_step(model, data)
            pin_frozen_joints(context, data, object_target)
            apply_wrist_tracking_correction(
                context,
                data,
                relaxed_target,
                wrist_stiffness_scale=wrist_stiffness_scale,
            )
            clamp_limited_robot_joints(context, data)
            if (
                not np.isfinite(data.qpos).all()
                or not np.isfinite(data.qvel).all()
                or np.max(np.abs(data.qpos)) > max_abs_qpos
            ):
                data.qpos[:] = prev_qpos
                data.qvel[:] = 0.0
                break
            mujoco.mj_forward(model, data)

        cpos, cmask = extract_mujoco_contacts(context, data)
        if use_ik_contact_fallback:
            frame_contact = None if ik_contact is None else ik_contact[frame_idx]
            frame_contact_pos = (
                None if ik_contact_pos is None else ik_contact_pos[frame_idx]
            )
            cpos, cmask = fill_contacts_from_ik_fallback(
                context,
                data,
                contact_pos=cpos,
                contact_mask=cmask,
                ik_contact=frame_contact,
                ik_contact_pos=frame_contact_pos,
            )
        rollout_qpos[frame_idx] = data.qpos
        contact_pos[frame_idx] = cpos
        contact_mask[frame_idx] = cmask

    rollout_qpos[:, context.object_qpos_start :] = object_qpos_ref[
        :, context.object_qpos_start :
    ]
    return rollout_qpos, contact_pos, contact_mask


def prepare_target_trajectory(
    context: RetargetContext,
    qpos: np.ndarray,
) -> np.ndarray:
    """Prepare DexMachina-style absolute controller targets for saving/replay."""
    target_qpos = clamp_robot_qpos_targets(context, qpos)
    if target_qpos.shape[0] == 0:
        return target_qpos

    target_qpos[:, context.object_qpos_start :] = qpos[:, context.object_qpos_start :]
    return target_qpos


def make_contact_links(
    contact_pos: np.ndarray,
    contact_mask: np.ndarray,
    part_ids: np.ndarray,
) -> np.ndarray:
    links = np.zeros(contact_pos.shape[:-1] + (4,), dtype=np.float64)
    links[..., :3] = contact_pos
    for part_index, part_id in enumerate(part_ids):
        links[:, part_index, :, 3] = np.where(contact_mask[:, part_index], part_id, 0.0)
    return links


def compute_qvel(model: mujoco.MjModel, qpos: np.ndarray, dt: float) -> np.ndarray:
    qvel = np.zeros((qpos.shape[0], model.nv), dtype=np.float64)
    for frame_idx in range(1, qpos.shape[0]):
        mujoco.mj_differentiatePos(
            model,
            qvel[frame_idx],
            dt,
            qpos[frame_idx - 1],
            qpos[frame_idx],
        )
    return qvel


def compute_penetration_stats(
    context: RetargetContext,
    qpos: np.ndarray,
) -> dict[str, np.ndarray]:
    max_penetration = np.zeros(qpos.shape[0], dtype=np.float64)
    max_hand_object_penetration = np.zeros(qpos.shape[0], dtype=np.float64)
    max_hand_self_penetration = np.zeros(qpos.shape[0], dtype=np.float64)
    num_contacts = np.zeros(qpos.shape[0], dtype=np.float64)

    model = context.model
    data = mujoco.MjData(model)
    hand_geoms = set(context.contacts.hand_geom_to_link)
    object_geoms = set(context.contacts.object_geom_to_part)

    for frame_idx, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        num_contacts[frame_idx] = data.ncon
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            penetration = max(0.0, -float(contact.dist))
            if penetration <= 0.0:
                continue
            max_penetration[frame_idx] = max(
                max_penetration[frame_idx],
                penetration,
            )

            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            geom1_is_hand = geom1 in hand_geoms
            geom2_is_hand = geom2 in hand_geoms
            geom1_is_object = geom1 in object_geoms
            geom2_is_object = geom2 in object_geoms
            if (geom1_is_hand and geom2_is_object) or (
                geom2_is_hand and geom1_is_object
            ):
                max_hand_object_penetration[frame_idx] = max(
                    max_hand_object_penetration[frame_idx],
                    penetration,
                )
            elif geom1_is_hand and geom2_is_hand:
                max_hand_self_penetration[frame_idx] = max(
                    max_hand_self_penetration[frame_idx],
                    penetration,
                )

    return {
        "num_contacts": num_contacts,
        "max_penetration": max_penetration,
        "max_hand_object_penetration": max_hand_object_penetration,
        "max_hand_self_penetration": max_hand_self_penetration,
    }


def hand_penetration_for_qpos(
    context: RetargetContext,
    data: mujoco.MjData,
    qpos: np.ndarray,
) -> tuple[float, float]:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(context.model, data)

    hand_geoms = set(context.contacts.hand_geom_to_link)
    object_geoms = set(context.contacts.object_geom_to_part)
    max_hand_object_penetration = 0.0
    max_hand_self_penetration = 0.0

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        penetration = max(0.0, -float(contact.dist))
        if penetration <= 0.0:
            continue
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        geom1_is_hand = geom1 in hand_geoms
        geom2_is_hand = geom2 in hand_geoms
        geom1_is_object = geom1 in object_geoms
        geom2_is_object = geom2 in object_geoms
        if (geom1_is_hand and geom2_is_object) or (
            geom2_is_hand and geom1_is_object
        ):
            max_hand_object_penetration = max(
                max_hand_object_penetration,
                penetration,
            )
        elif geom1_is_hand and geom2_is_hand:
            max_hand_self_penetration = max(
                max_hand_self_penetration,
                penetration,
            )

    return max_hand_object_penetration, max_hand_self_penetration


def repair_hand_penetration_by_line_search(
    context: RetargetContext,
    candidate_qpos: np.ndarray,
    safe_qpos: np.ndarray,
    object_qpos_ref: np.ndarray,
    iterations: int,
    tolerance: float,
    ik_contact: np.ndarray | None,
    ik_contact_pos: np.ndarray | None,
    use_ik_contact_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_frames = candidate_qpos.shape[0]
    num_parts = len(context.contacts.part_names)
    num_links = len(context.contacts.hand_link_names)
    repaired_qpos = np.zeros_like(candidate_qpos)
    contact_pos = np.zeros((total_frames, num_parts, num_links, 3), dtype=np.float64)
    contact_mask = np.zeros((total_frames, num_parts, num_links), dtype=bool)
    data = mujoco.MjData(context.model)
    robot_width = context.object_qpos_start

    for frame_idx in range(total_frames):
        if frame_idx % 25 == 0:
            loguru.logger.info(
                f"Strict collision repair frame {frame_idx + 1}/{total_frames}"
            )

        candidate = candidate_qpos[frame_idx].copy()
        safe = safe_qpos[frame_idx].copy()
        object_ref = object_qpos_ref[frame_idx]
        candidate[robot_width:] = object_ref[robot_width:]
        safe[robot_width:] = object_ref[robot_width:]

        candidate_hand_object, candidate_hand_self = hand_penetration_for_qpos(
            context,
            data,
            candidate,
        )
        if (
            candidate_hand_object <= tolerance
            and candidate_hand_self <= tolerance
        ):
            repaired = candidate
        else:
            safe_hand_object, safe_hand_self = hand_penetration_for_qpos(
                context,
                data,
                safe,
            )
            if safe_hand_object > tolerance or safe_hand_self > tolerance:
                repaired = safe
            else:
                lower = 0.0
                upper = 1.0
                repaired = safe.copy()
                for _ in range(max(1, iterations)):
                    weight = 0.5 * (lower + upper)
                    trial = safe.copy()
                    trial[:robot_width] = (
                        safe[:robot_width]
                        + weight * (candidate[:robot_width] - safe[:robot_width])
                    )
                    trial[robot_width:] = object_ref[robot_width:]
                    hand_object, hand_self = hand_penetration_for_qpos(
                        context,
                        data,
                        trial,
                    )
                    if hand_object <= tolerance and hand_self <= tolerance:
                        lower = weight
                        repaired = trial
                    else:
                        upper = weight

        data.qpos[:] = repaired
        data.qvel[:] = 0.0
        mujoco.mj_forward(context.model, data)
        cpos, cmask = extract_mujoco_contacts(context, data)
        if use_ik_contact_fallback:
            frame_contact = None if ik_contact is None else ik_contact[frame_idx]
            frame_contact_pos = (
                None if ik_contact_pos is None else ik_contact_pos[frame_idx]
            )
            cpos, cmask = fill_contacts_from_ik_fallback(
                context,
                data,
                contact_pos=cpos,
                contact_mask=cmask,
                ik_contact=frame_contact,
                ik_contact_pos=frame_contact_pos,
            )
        repaired_qpos[frame_idx] = repaired
        contact_pos[frame_idx] = cpos
        contact_mask[frame_idx] = cmask

    return repaired_qpos, contact_pos, contact_mask


def extract_contacts_for_trajectory(
    context: RetargetContext,
    qpos: np.ndarray,
    ik_contact: np.ndarray | None,
    ik_contact_pos: np.ndarray | None,
    use_ik_contact_fallback: bool,
) -> tuple[np.ndarray, np.ndarray]:
    total_frames = qpos.shape[0]
    num_parts = len(context.contacts.part_names)
    num_links = len(context.contacts.hand_link_names)
    contact_pos = np.zeros((total_frames, num_parts, num_links, 3), dtype=np.float64)
    contact_mask = np.zeros((total_frames, num_parts, num_links), dtype=bool)
    data = mujoco.MjData(context.model)

    for frame_idx, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(context.model, data)
        cpos, cmask = extract_mujoco_contacts(context, data)
        if use_ik_contact_fallback:
            frame_contact = None if ik_contact is None else ik_contact[frame_idx]
            frame_contact_pos = (
                None if ik_contact_pos is None else ik_contact_pos[frame_idx]
            )
            cpos, cmask = fill_contacts_from_ik_fallback(
                context,
                data,
                contact_pos=cpos,
                contact_mask=cmask,
                ik_contact=frame_contact,
                ik_contact_pos=frame_contact_pos,
            )
        contact_pos[frame_idx] = cpos
        contact_mask[frame_idx] = cmask

    return contact_pos, contact_mask


def contact_part_color(part_index: int, alpha: float = 1.0) -> np.ndarray:
    color = CONTACT_PART_COLORS[part_index % len(CONTACT_PART_COLORS)].copy()
    color[3] = alpha
    return color


def next_scene_geom(scene: Any) -> Any | None:
    if scene is None:
        return None
    maxgeom = getattr(scene, "maxgeom", len(getattr(scene, "geoms", [])))
    if int(scene.ngeom) >= int(maxgeom):
        return None
    geom = scene.geoms[int(scene.ngeom)]
    scene.ngeom += 1
    return geom


def add_scene_sphere(
    scene: Any,
    pos: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> bool:
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape != (3,) or not np.isfinite(pos).all():
        return False
    geom = next_scene_geom(scene)
    if geom is None:
        return False
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        pos,
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    return True


def add_scene_capsule(
    scene: Any,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> bool:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    if start.shape != (3,) or end.shape != (3,):
        return False
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        return False
    if np.linalg.norm(end - start) < 1e-9:
        return False
    geom = next_scene_geom(scene)
    if geom is None:
        return False
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        float(radius),
        start,
        end,
    )
    return True


def object_hand_contact_part_index(
    context: RetargetContext,
    geom1: int,
    geom2: int,
) -> int | None:
    part_index = context.contacts.object_geom_to_part.get(geom1)
    link_index = context.contacts.hand_geom_to_link.get(geom2)
    if part_index is not None and link_index is not None:
        return part_index
    part_index = context.contacts.object_geom_to_part.get(geom2)
    link_index = context.contacts.hand_geom_to_link.get(geom1)
    if part_index is not None and link_index is not None:
        return part_index
    return None


def is_hand_self_contact(
    context: RetargetContext,
    geom1: int,
    geom2: int,
) -> bool:
    return (
        geom1 in context.contacts.hand_geom_to_link
        and geom2 in context.contacts.hand_geom_to_link
    )


def draw_grouped_contact_points(
    scene: Any,
    contact_pos: np.ndarray | None,
    contact_mask: np.ndarray | None,
    radius: float,
) -> int:
    if contact_pos is None or contact_mask is None:
        return 0
    if contact_pos.shape[:2] != contact_mask.shape or contact_pos.shape[-1] != 3:
        return 0

    drawn = 0
    for part_index, link_index in zip(*np.nonzero(contact_mask), strict=False):
        rgba = contact_part_color(int(part_index), alpha=1.0)
        if add_scene_sphere(
            scene,
            contact_pos[int(part_index), int(link_index)],
            radius=radius,
            rgba=rgba,
        ):
            drawn += 1
    return drawn


def draw_raw_contact_points(
    context: RetargetContext,
    data: mujoco.MjData,
    scene: Any,
    radius: float,
    include_hand_self: bool,
) -> int:
    drawn = 0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        part_index = object_hand_contact_part_index(context, geom1, geom2)
        if part_index is not None:
            rgba = contact_part_color(part_index, alpha=1.0)
        elif include_hand_self and is_hand_self_contact(context, geom1, geom2):
            rgba = np.array([1.0, 0.95, 0.1, 1.0], dtype=np.float32)
        else:
            continue
        if add_scene_sphere(
            scene,
            np.asarray(contact.pos, dtype=np.float64),
            radius=radius,
            rgba=rgba,
        ):
            drawn += 1
    return drawn


def draw_raw_contact_pair_lines(
    context: RetargetContext,
    data: mujoco.MjData,
    scene: Any,
    radius: float,
    include_hand_self: bool,
) -> int:
    drawn = 0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        part_index = object_hand_contact_part_index(context, geom1, geom2)
        if part_index is not None:
            rgba = contact_part_color(part_index, alpha=0.55)
        elif include_hand_self and is_hand_self_contact(context, geom1, geom2):
            rgba = np.array([1.0, 0.95, 0.1, 0.45], dtype=np.float32)
        else:
            continue
        if add_scene_capsule(
            scene,
            data.geom_xpos[geom1],
            data.geom_xpos[geom2],
            radius=radius,
            rgba=rgba,
        ):
            drawn += 1
    return drawn


def draw_grouped_contact_pair_lines(
    context: RetargetContext,
    data: mujoco.MjData,
    scene: Any,
    contact_mask: np.ndarray | None,
    radius: float,
) -> int:
    if contact_mask is None:
        return 0

    centers_part = part_centers(context, data)
    centers_link = link_centers(context, data)
    drawn = 0
    for part_index, link_index in zip(*np.nonzero(contact_mask), strict=False):
        part_index = int(part_index)
        link_index = int(link_index)
        if part_index >= centers_part.shape[0] or link_index >= centers_link.shape[0]:
            continue
        if context.contacts.part_geom_ids[part_index].size == 0:
            continue
        rgba = contact_part_color(part_index, alpha=0.4)
        if add_scene_capsule(
            scene,
            centers_part[part_index],
            centers_link[link_index],
            radius=radius,
            rgba=rgba,
        ):
            drawn += 1
    return drawn


def draw_contact_visualization(
    context: RetargetContext,
    data: mujoco.MjData,
    scene: Any,
    contact_pos: np.ndarray | None,
    contact_mask: np.ndarray | None,
    contact_visualization: Literal["points", "lines", "both"],
    point_radius: float,
    line_width: float,
    include_hand_self: bool,
) -> None:
    if scene is None:
        return

    scene.ngeom = 0
    if contact_visualization not in {"points", "lines", "both"}:
        raise ValueError(
            "contact_visualization must be one of `points`, `lines`, or `both`."
        )

    if contact_visualization in {"points", "both"}:
        drawn_points = draw_grouped_contact_points(
            scene,
            contact_pos=contact_pos,
            contact_mask=contact_mask,
            radius=point_radius,
        )
        if drawn_points == 0:
            draw_raw_contact_points(
                context,
                data,
                scene,
                radius=point_radius,
                include_hand_self=include_hand_self,
            )

    if contact_visualization in {"lines", "both"}:
        drawn_lines = draw_raw_contact_pair_lines(
            context,
            data,
            scene,
            radius=line_width,
            include_hand_self=include_hand_self,
        )
        if drawn_lines == 0:
            draw_grouped_contact_pair_lines(
                context,
                data,
                scene,
                contact_mask=contact_mask,
                radius=line_width,
            )


def body_with_ancestors_and_descendants(
    model: mujoco.MjModel,
    body_id: int,
) -> set[int]:
    body_ids = descendants_of(model, body_id)
    current = body_id
    while current != 0:
        body_ids.add(current)
        current = int(model.body_parentid[current])
    return body_ids


def contact_visualization_geom_ids(context: RetargetContext) -> np.ndarray:
    """Return all model geoms that should fade in the contact-focused view."""
    model = context.model
    body_ids: set[int] = set()
    seed_geom_ids = set(context.contacts.object_geom_to_part) | set(
        context.contacts.hand_geom_to_link
    )
    for geom_id in seed_geom_ids:
        body_ids.update(
            body_with_ancestors_and_descendants(
                model,
                int(model.geom_bodyid[geom_id]),
            )
        )

    geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
    ]
    return np.asarray(geom_ids, dtype=np.int64)


def make_contact_view_context(
    context: RetargetContext,
    contact_view_alpha: float,
) -> RetargetContext:
    model = copy.copy(context.model)
    alpha = float(np.clip(contact_view_alpha, 0.0, 1.0))
    geom_ids = contact_visualization_geom_ids(context)
    if geom_ids.size > 0:
        model.geom_rgba[geom_ids, 3] = np.minimum(model.geom_rgba[geom_ids, 3], alpha)
        mat_ids = np.unique(model.geom_matid[geom_ids])
        mat_ids = mat_ids[mat_ids >= 0]
        if mat_ids.size > 0:
            model.mat_rgba[mat_ids, 3] = np.minimum(
                model.mat_rgba[mat_ids, 3],
                alpha,
            )
    return replace(context, model=model)


def set_replay_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
) -> None:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def frame_contact_arrays(
    frame_idx: int,
    contact_pos: np.ndarray | None,
    contact_mask: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    frame_contact_pos = None
    frame_contact_mask = None
    if contact_pos is not None and frame_idx < contact_pos.shape[0]:
        frame_contact_pos = contact_pos[frame_idx]
    if contact_mask is not None and frame_idx < contact_mask.shape[0]:
        frame_contact_mask = contact_mask[frame_idx]
    return frame_contact_pos, frame_contact_mask


def sync_contact_viewer(
    context: RetargetContext,
    data: mujoco.MjData,
    viewer: Any,
    frame_idx: int,
    qpos: np.ndarray,
    contact_pos: np.ndarray | None,
    contact_mask: np.ndarray | None,
    contact_visualization: Literal["points", "lines", "both"],
    contact_point_radius: float,
    contact_line_width: float,
    visualize_hand_self_contacts: bool,
) -> None:
    set_replay_frame(context.model, data, qpos[frame_idx])
    frame_contact_pos, frame_contact_mask = frame_contact_arrays(
        frame_idx,
        contact_pos,
        contact_mask,
    )
    draw_contact_visualization(
        context,
        data,
        getattr(viewer, "user_scn", None),
        contact_pos=frame_contact_pos,
        contact_mask=frame_contact_mask,
        contact_visualization=contact_visualization,
        point_radius=contact_point_radius,
        line_width=contact_line_width,
        include_hand_self=visualize_hand_self_contacts,
    )
    viewer.sync()


def write_h5(
    path: str,
    datasets: dict[str, np.ndarray],
    string_datasets: dict[str, list[str]],
    attrs: dict[str, str | float | int],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as h5_file:
        for key, value in datasets.items():
            if value.dtype == np.bool_:
                h5_file.create_dataset(key, data=value)
            else:
                h5_file.create_dataset(key, data=value.astype(np.float64))
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for key, values in string_datasets.items():
            h5_file.create_dataset(key, data=np.asarray(values, dtype=string_dtype))
        for key, value in attrs.items():
            h5_file.attrs[key] = value


def resolve_trajectory_path(
    processed_dir: str,
    robot_type: str,
    task: str,
    trajectory_path: str | None,
) -> str:
    if trajectory_path is not None:
        return os.path.abspath(trajectory_path)

    candidates = [
        f"trajectory_kinematic_{robot_type}.npz",
        f"trajectory_kinematic_mink_{robot_type}_{task}.npz",
        "trajectory_kinematic_mink.npz",
        "trajectory_kinematic.npz",
    ]
    for name in candidates:
        candidate = os.path.join(processed_dir, name)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    candidate_list = "\n  ".join(os.path.join(processed_dir, name) for name in candidates)
    raise FileNotFoundError(f"Could not find a trajectory file. Checked:\n  {candidate_list}")


def load_input_npz(path: str) -> tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None]:
    with np.load(path) as trajectory:
        qpos = np.asarray(trajectory["qpos"], dtype=np.float64)
        frequency = float(trajectory["frequency"]) if "frequency" in trajectory else np.nan
        ik_contact = np.asarray(trajectory["contact"]) if "contact" in trajectory else None
        ik_contact_pos = (
            np.asarray(trajectory["contact_pos"], dtype=np.float64)
            if "contact_pos" in trajectory
            else None
        )
    qpos = qpos.reshape(-1, qpos.shape[-1])
    return qpos, frequency, ik_contact, ik_contact_pos


def replay_viewer(
    context: RetargetContext,
    qpos: np.ndarray,
    fps: int,
    visualize_contacts: bool,
    contact_view_mode: Literal["overlay", "separate", "only"],
    contact_view_alpha: float,
    contact_pos: np.ndarray | None,
    contact_mask: np.ndarray | None,
    contact_visualization: Literal["points", "lines", "both"],
    contact_point_radius: float,
    contact_line_width: float,
    visualize_hand_self_contacts: bool,
) -> None:
    try:
        from loop_rate_limiters import RateLimiter
    except ImportError:
        RateLimiter = None

    model = context.model
    show_contact_view = visualize_contacts and contact_view_mode in {"separate", "only"}
    show_regular_view = not show_contact_view or contact_view_mode != "only"
    draw_contacts_in_regular_view = visualize_contacts and contact_view_mode == "overlay"

    data = mujoco.MjData(model) if show_regular_view else None
    contact_context = (
        make_contact_view_context(context, contact_view_alpha)
        if show_contact_view
        else None
    )
    contact_data = (
        mujoco.MjData(contact_context.model)
        if contact_context is not None
        else None
    )
    frame_idx = 0
    rate_limiter = RateLimiter(fps) if RateLimiter is not None else None

    with ExitStack() as stack:
        viewer = (
            stack.enter_context(mujoco.viewer.launch_passive(model, data))
            if data is not None
            else None
        )
        contact_viewer = (
            stack.enter_context(
                mujoco.viewer.launch_passive(contact_context.model, contact_data)
            )
            if contact_context is not None and contact_data is not None
            else None
        )

        def any_viewer_running() -> bool:
            return any(
                view is not None and view.is_running()
                for view in (viewer, contact_viewer)
            )

        while any_viewer_running():
            if viewer is not None and viewer.is_running() and data is not None:
                set_replay_frame(model, data, qpos[frame_idx])
                scene = getattr(viewer, "user_scn", None)
                if draw_contacts_in_regular_view:
                    frame_contact_pos, frame_contact_mask = frame_contact_arrays(
                        frame_idx,
                        contact_pos,
                        contact_mask,
                    )
                    draw_contact_visualization(
                        context,
                        data,
                        scene,
                        contact_pos=frame_contact_pos,
                        contact_mask=frame_contact_mask,
                        contact_visualization=contact_visualization,
                        point_radius=contact_point_radius,
                        line_width=contact_line_width,
                        include_hand_self=visualize_hand_self_contacts,
                    )
                elif scene is not None:
                    scene.ngeom = 0
                viewer.sync()

            if (
                contact_viewer is not None
                and contact_viewer.is_running()
                and contact_context is not None
                and contact_data is not None
            ):
                sync_contact_viewer(
                    contact_context,
                    contact_data,
                    contact_viewer,
                    frame_idx,
                    qpos,
                    contact_pos,
                    contact_mask,
                    contact_visualization=contact_visualization,
                    contact_point_radius=contact_point_radius,
                    contact_line_width=contact_line_width,
                    visualize_hand_self_contacts=visualize_hand_self_contacts,
                )

            frame_idx = (frame_idx + 1) % qpos.shape[0]
            if rate_limiter is not None:
                rate_limiter.sleep()


def main(
    dataset_dir: str = "example_datasets",
    dataset_name: str = "arctic",
    robot_type: str = "leap",
    embodiment_type: str = "right",
    task: str = "scissors",
    data_id: int = 0,
    trajectory_path: str | None = None,
    output_path: str | None = None,
    output_dir: str | None = None,
    model_path: str | None = None,
    robot_joint_count: int | None = None,
    object_qpos_start: int | None = None,
    max_object_parts: int | None = 2,
    start_idx: int = 0,
    end_idx: int = -1,
    settle_steps: int = 500,
    collision_relax_steps: int = 0,
    num_workers: int = 0,
    sim_dt: float = 0.002, # 0.002
    solver_iterations: int = 80,
    solver_tolerance: float = 1e-10,
    integrator: str = "implicitfast",
    max_abs_qpos: float = 10.0,
    enable_collision: bool = True,
    enable_self_collision: bool = True,
    force_collision_masks: bool = True,
    clear_contact_exclusions: bool = True,
    collide_parent_child: bool = False,
    collision_margin: float = 0.002,
    collision_projection: bool = True,
    projection_backend: Literal["physics", "least_squares"] = "physics",
    projection_pair_margin: float = 0.02,
    projection_safety_margin: float = 0.0005,
    projection_tracking_weight: float = 1.0,
    projection_collision_weight: float = 20000.0,
    projection_max_nfev: int = 100,
    projection_outer_iterations: int = 2,
    strict_collision_repair: bool = True,
    strict_repair_settle_steps: int = 100,
    strict_repair_relax_steps: int = 1000,
    strict_repair_fallback_relax_steps: int = 200,
    repair_bisection_steps: int = 20,
    repair_penetration_tolerance: float = 1e-6,
    output_qpos_source: Literal["target", "achieved", "rollout"] = "rollout",
    rollout_target_source: Literal["target", "achieved"] = "target",
    control_steps_per_frame: int | None = 16,
    wrist_stiffness_scale: float = 1.0,
    use_ik_contact_fallback: bool = True,
    show_viewer: bool = True,
    viewer_fps: int = 60,
    visualize_contacts: bool = True,
    contact_view_mode: Literal["overlay", "separate", "only"] = "separate",
    contact_view_alpha: float = 0.28,
    contact_visualization: Literal["points", "lines", "both"] = "both",
    contact_point_radius: float = 0.006,
    contact_line_width: float = 0.0015,
    visualize_hand_self_contacts: bool = False,
) -> str:
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    trajectory_path = resolve_trajectory_path(
        processed_dir=processed_dir,
        robot_type=robot_type,
        task=task,
        trajectory_path=trajectory_path,
    )
    if model_path is None:
        model_path = os.path.join(processed_dir, "..", "scene.xml")
    model_path = os.path.abspath(model_path)

    if output_dir is None:
        output_dir = os.path.join(Path(__file__).resolve().parent, "mink")
    if output_path is None:
        output_path = os.path.join(output_dir, f"dexmachina_retargeted_{robot_type}_{task}.h5")
    output_path = os.path.abspath(output_path)

    loguru.logger.info(f"Loading IK trajectory from {trajectory_path}")
    qpos, frequency, ik_contact, ik_contact_pos = load_input_npz(trajectory_path)
    end = None if end_idx == -1 else end_idx
    qpos = qpos[start_idx:end]
    if ik_contact is not None:
        ik_contact = ik_contact.reshape(-1, ik_contact.shape[-1])[start_idx:end]
    if ik_contact_pos is not None:
        ik_contact_pos = ik_contact_pos.reshape(-1, ik_contact_pos.shape[-2], 3)[start_idx:end]

    if qpos.shape[0] == 0:
        raise ValueError("Selected trajectory slice is empty.")

    if collision_projection and projection_backend == "least_squares":
        projection_collision_margin = max(collision_margin, projection_pair_margin)
    else:
        projection_collision_margin = collision_margin

    context = build_context(
        model_path=model_path,
        sim_dt=sim_dt,
        solver_iterations=solver_iterations,
        solver_tolerance=solver_tolerance,
        integrator=integrator,
        object_qpos_start=object_qpos_start,
        max_object_parts=max_object_parts,
        enable_collision=enable_collision,
        enable_self_collision=enable_self_collision,
        force_collision_masks=force_collision_masks,
        clear_contact_exclusions=clear_contact_exclusions,
        collide_parent_child=collide_parent_child,
        collision_margin=collision_margin,
        wrist_stiffness_scale=wrist_stiffness_scale,
    )
    if qpos.shape[1] != context.model.nq:
        raise ValueError(
            f"Trajectory qpos width {qpos.shape[1]} does not match MuJoCo model.nq "
            f"{context.model.nq} from {model_path}."
        )

    loguru.logger.info(
        "Discovered "
        f"{len(context.contacts.part_names)} object parts "
        f"({context.contacts.part_names}) and "
        f"{len(context.contacts.hand_link_names)} hand collision links."
    )

    dt = (1.0 / frequency) if not np.isnan(frequency) and frequency > 0 else sim_dt
    target_qpos = prepare_target_trajectory(
        context=context,
        qpos=qpos,
    )
    target_delta = np.max(
        np.abs(
            target_qpos[:, : context.object_qpos_start]
            - qpos[:, : context.object_qpos_start]
        )
    )
    if target_delta > 1e-9:
        loguru.logger.info(
            "Clamped absolute controller targets; "
            f"max robot-space change from IK input is {target_delta:.6f}."
        )

    if collision_projection:
        projection_context = context
        if projection_backend == "physics":
            loguru.logger.info(
                "Running DexMachina-style per-frame physics projection with "
                f"{settle_steps} target-held MuJoCo steps."
            )
            achieved_qpos, contact_pos, contact_mask = settle_trajectory(
                context=context,
                model_path=model_path,
                qpos=target_qpos,
                settle_steps=settle_steps,
                collision_relax_steps=collision_relax_steps,
                num_workers=num_workers,
                sim_dt=sim_dt,
                solver_iterations=solver_iterations,
                solver_tolerance=solver_tolerance,
                integrator=integrator,
                max_abs_qpos=max_abs_qpos,
                object_qpos_start=context.object_qpos_start,
                max_object_parts=max_object_parts,
                enable_collision=enable_collision,
                enable_self_collision=enable_self_collision,
                force_collision_masks=force_collision_masks,
                clear_contact_exclusions=clear_contact_exclusions,
                collide_parent_child=collide_parent_child,
                collision_margin=collision_margin,
                wrist_stiffness_scale=wrist_stiffness_scale,
                ik_contact=ik_contact,
                ik_contact_pos=ik_contact_pos,
                use_ik_contact_fallback=use_ik_contact_fallback,
            )
        elif projection_backend == "least_squares":
            if projection_collision_margin > collision_margin:
                loguru.logger.info(
                    "Using a wider contact margin for projection pair discovery "
                    f"({projection_collision_margin}) while keeping the final "
                    f"trajectory/viewer margin at {collision_margin}."
                )
                projection_context = build_context(
                    model_path=model_path,
                    sim_dt=sim_dt,
                    solver_iterations=solver_iterations,
                    solver_tolerance=solver_tolerance,
                    integrator=integrator,
                    object_qpos_start=context.object_qpos_start,
                    max_object_parts=max_object_parts,
                    enable_collision=enable_collision,
                    enable_self_collision=enable_self_collision,
                    force_collision_masks=force_collision_masks,
                    clear_contact_exclusions=clear_contact_exclusions,
                    collide_parent_child=collide_parent_child,
                    collision_margin=projection_collision_margin,
                    wrist_stiffness_scale=wrist_stiffness_scale,
                )
            achieved_qpos, contact_pos, contact_mask = project_collision_aware_trajectory(
                context=projection_context,
                model_path=model_path,
                qpos=target_qpos,
                num_workers=num_workers,
                sim_dt=sim_dt,
                solver_iterations=solver_iterations,
                solver_tolerance=solver_tolerance,
                integrator=integrator,
                object_qpos_start=context.object_qpos_start,
                max_object_parts=max_object_parts,
                enable_collision=enable_collision,
                enable_self_collision=enable_self_collision,
                force_collision_masks=force_collision_masks,
                clear_contact_exclusions=clear_contact_exclusions,
                collide_parent_child=collide_parent_child,
                collision_margin=projection_collision_margin,
                wrist_stiffness_scale=wrist_stiffness_scale,
                pair_margin=projection_pair_margin,
                safety_margin=projection_safety_margin,
                tracking_weight=projection_tracking_weight,
                collision_weight=projection_collision_weight,
                max_nfev=projection_max_nfev,
                outer_iterations=projection_outer_iterations,
                ik_contact=ik_contact,
                ik_contact_pos=ik_contact_pos,
                use_ik_contact_fallback=use_ik_contact_fallback,
            )
        else:
            raise ValueError(
                f"Unknown projection_backend `{projection_backend}`. "
                "Choose `physics` or `least_squares`."
            )
        if strict_collision_repair:
            loguru.logger.info(
                "Building independent near-collision baseline for strict repair."
            )
            near_safe_qpos, _, _ = settle_trajectory(
                context=context,
                model_path=model_path,
                qpos=target_qpos,
                settle_steps=strict_repair_settle_steps,
                collision_relax_steps=strict_repair_relax_steps,
                num_workers=num_workers,
                sim_dt=sim_dt,
                solver_iterations=solver_iterations,
                solver_tolerance=solver_tolerance,
                integrator=integrator,
                max_abs_qpos=max_abs_qpos,
                object_qpos_start=context.object_qpos_start,
                max_object_parts=max_object_parts,
                enable_collision=enable_collision,
                enable_self_collision=enable_self_collision,
                force_collision_masks=force_collision_masks,
                clear_contact_exclusions=clear_contact_exclusions,
                collide_parent_child=collide_parent_child,
                collision_margin=collision_margin,
                wrist_stiffness_scale=wrist_stiffness_scale,
                ik_contact=None,
                ik_contact_pos=None,
                use_ik_contact_fallback=False,
            )
            loguru.logger.info(
                "Building conservative collision-free fallback for strict repair."
            )
            fallback_safe_qpos, _, _ = rollout_controlled_trajectory(
                context=context,
                target_qpos=target_qpos,
            object_qpos_ref=target_qpos,
            steps_per_frame=1,
            collision_relax_steps=strict_repair_fallback_relax_steps,
            max_abs_qpos=max_abs_qpos,
            wrist_stiffness_scale=wrist_stiffness_scale,
            ik_contact=None,
            ik_contact_pos=None,
            use_ik_contact_fallback=False,
            )
            safe_qpos, _, _ = repair_hand_penetration_by_line_search(
                context=context,
                candidate_qpos=near_safe_qpos,
                safe_qpos=fallback_safe_qpos,
                object_qpos_ref=target_qpos,
                iterations=repair_bisection_steps,
                tolerance=repair_penetration_tolerance,
                ik_contact=None,
                ik_contact_pos=None,
                use_ik_contact_fallback=False,
            )
            achieved_qpos, contact_pos, contact_mask = (
                repair_hand_penetration_by_line_search(
                    context=context,
                    candidate_qpos=achieved_qpos,
                    safe_qpos=safe_qpos,
                    object_qpos_ref=target_qpos,
                    iterations=repair_bisection_steps,
                    tolerance=repair_penetration_tolerance,
                    ik_contact=ik_contact,
                    ik_contact_pos=ik_contact_pos,
                    use_ik_contact_fallback=use_ik_contact_fallback,
                )
            )
        elif projection_context is not context:
            contact_pos, contact_mask = extract_contacts_for_trajectory(
                context=context,
                qpos=achieved_qpos,
                ik_contact=ik_contact,
                ik_contact_pos=ik_contact_pos,
                use_ik_contact_fallback=use_ik_contact_fallback,
            )
    else:
        stage1_achieved_qpos, stage1_contact_pos, stage1_contact_mask = settle_trajectory(
            context=context,
            model_path=model_path,
            qpos=target_qpos,
            settle_steps=settle_steps,
            collision_relax_steps=collision_relax_steps,
            num_workers=num_workers,
            sim_dt=sim_dt,
            solver_iterations=solver_iterations,
            solver_tolerance=solver_tolerance,
            integrator=integrator,
            max_abs_qpos=max_abs_qpos,
            object_qpos_start=object_qpos_start,
            max_object_parts=max_object_parts,
            enable_collision=enable_collision,
            enable_self_collision=enable_self_collision,
            force_collision_masks=force_collision_masks,
            clear_contact_exclusions=clear_contact_exclusions,
            collide_parent_child=collide_parent_child,
            collision_margin=collision_margin,
            wrist_stiffness_scale=wrist_stiffness_scale,
            ik_contact=ik_contact,
            ik_contact_pos=ik_contact_pos,
            use_ik_contact_fallback=use_ik_contact_fallback,
        )
        achieved_qpos = stage1_achieved_qpos
        contact_pos = stage1_contact_pos
        contact_mask = stage1_contact_mask

    achieved_contact_pos = contact_pos
    achieved_contact_mask = contact_mask
    rollout_qpos: np.ndarray | None = None
    rollout_contact_pos: np.ndarray | None = None
    rollout_contact_mask: np.ndarray | None = None
    if control_steps_per_frame is None:
        stage2_control_steps = max(1, int(round(dt / sim_dt)))
    else:
        stage2_control_steps = max(1, control_steps_per_frame)

    if output_qpos_source == "rollout":
        loguru.logger.info(
            "Building DexMachina-style stage-2 replay by rolling out the "
            f"{rollout_target_source} controller target in one MuJoCo environment "
            f"({stage2_control_steps} control steps per frame)."
        )
        if rollout_target_source == "target":
            rollout_target_qpos = target_qpos.copy()
        elif rollout_target_source == "achieved":
            rollout_target_qpos = achieved_qpos.copy()
            rollout_target_qpos[:, context.object_qpos_start :] = target_qpos[
                :, context.object_qpos_start :
            ]
        else:
            raise ValueError(
                f"Unknown rollout_target_source `{rollout_target_source}`. "
                "Choose `target` or `achieved`."
            )
        rollout_qpos, rollout_contact_pos, rollout_contact_mask = rollout_controlled_trajectory(
            context=context,
            target_qpos=rollout_target_qpos,
            object_qpos_ref=target_qpos,
            steps_per_frame=stage2_control_steps,
            collision_relax_steps=0,
            max_abs_qpos=max_abs_qpos,
            wrist_stiffness_scale=wrist_stiffness_scale,
            ik_contact=ik_contact,
            ik_contact_pos=ik_contact_pos,
            use_ik_contact_fallback=use_ik_contact_fallback,
        )
        output_qpos = rollout_qpos
        output_contact_pos = rollout_contact_pos
        output_contact_mask = rollout_contact_mask
    elif output_qpos_source == "achieved":
        output_qpos = achieved_qpos
        output_contact_pos = achieved_contact_pos
        output_contact_mask = achieved_contact_mask
    else:
        output_qpos = target_qpos
        output_contact_pos, output_contact_mask = extract_contacts_for_trajectory(
            context=context,
            qpos=output_qpos,
            ik_contact=ik_contact,
            ik_contact_pos=ik_contact_pos,
            use_ik_contact_fallback=use_ik_contact_fallback,
        )

    if robot_joint_count is None:
        robot_joint_count = max(0, context.object_qpos_start - 6)

    h5_data = split_mink_qpos(
        qpos=output_qpos,
        robot_type=robot_type,
        robot_joint_count=robot_joint_count,
    )
    target_components = split_mink_qpos(
        qpos=target_qpos,
        robot_type=robot_type,
        robot_joint_count=robot_joint_count,
    )
    achieved_components = split_mink_qpos(
        qpos=achieved_qpos,
        robot_type=robot_type,
        robot_joint_count=robot_joint_count,
    )
    rollout_components = (
        None
        if rollout_qpos is None
        else split_mink_qpos(
            qpos=rollout_qpos,
            robot_type=robot_type,
            robot_joint_count=robot_joint_count,
        )
    )
    for key, value in target_components.items():
        h5_data[f"target_{key}"] = value
    for key, value in achieved_components.items():
        h5_data[f"achieved_{key}"] = value
    if rollout_components is not None:
        for key, value in rollout_components.items():
            h5_data[f"rollout_{key}"] = value

    contact_links = make_contact_links(
        contact_pos=output_contact_pos,
        contact_mask=output_contact_mask,
        part_ids=context.contacts.part_ids,
    )
    qvel = compute_qvel(context.model, output_qpos, dt=dt)
    achieved_qvel = compute_qvel(context.model, achieved_qpos, dt=dt)
    output_penetration = compute_penetration_stats(context, output_qpos)
    target_penetration = compute_penetration_stats(context, target_qpos)
    achieved_penetration = compute_penetration_stats(context, achieved_qpos)
    rollout_penetration = (
        None if rollout_qpos is None else compute_penetration_stats(context, rollout_qpos)
    )
    h5_data.update(
        {
            "qpos": output_qpos,
            "qvel": qvel,
            "target_qpos": target_qpos,
            "target_qvel": compute_qvel(context.model, target_qpos, dt=dt),
            "achieved_qpos": achieved_qpos,
            "achieved_qvel": achieved_qvel,
            "joint_targets": target_components["robot_joints"],
            "joint_qpos": h5_data["robot_joints"],
            "contact_pos": output_contact_pos,
            "contact_mask": output_contact_mask,
            "contact_links": contact_links,
            "achieved_contact_pos": achieved_contact_pos,
            "achieved_contact_mask": achieved_contact_mask,
        }
    )
    if rollout_qpos is not None:
        h5_data["rollout_qpos"] = rollout_qpos
        h5_data["rollout_qvel"] = compute_qvel(context.model, rollout_qpos, dt=dt)
    for key, value in output_penetration.items():
        h5_data[key] = value
    for key, value in target_penetration.items():
        h5_data[f"target_{key}"] = value
    for key, value in achieved_penetration.items():
        h5_data[f"achieved_{key}"] = value
    if rollout_penetration is not None:
        for key, value in rollout_penetration.items():
            h5_data[f"rollout_{key}"] = value

    loguru.logger.info(
        "Penetration stats for saved qpos: "
        f"max={output_penetration['max_penetration'].max():.6f}, "
        "hand-object="
        f"{output_penetration['max_hand_object_penetration'].max():.6f}, "
        f"hand-self={output_penetration['max_hand_self_penetration'].max():.6f}."
    )

    attrs: dict[str, str | float | int] = {
        "source_npz": trajectory_path,
        "model_path": model_path,
        "dataset_name": dataset_name,
        "robot_type": robot_type,
        "embodiment_type": embodiment_type,
        "task": task,
        "data_id": int(data_id),
        "settle_steps": int(settle_steps),
        "collision_relax_steps": int(collision_relax_steps),
        "control_steps_per_frame": int(stage2_control_steps),
        "rollout_target_source": rollout_target_source,
        "wrist_stiffness_scale": float(wrist_stiffness_scale),
        "sim_dt": float(sim_dt),
        "integrator": integrator,
        "object_qpos_start": int(context.object_qpos_start),
        "enable_collision": int(enable_collision),
        "enable_self_collision": int(enable_self_collision),
        "force_collision_masks": int(force_collision_masks),
        "clear_contact_exclusions": int(clear_contact_exclusions),
        "collide_parent_child": int(collide_parent_child),
        "collision_margin": float(collision_margin),
        "projection_collision_margin": float(projection_collision_margin),
        "collision_projection": int(collision_projection),
        "projection_backend": projection_backend,
        "projection_pair_margin": float(projection_pair_margin),
        "projection_safety_margin": float(projection_safety_margin),
        "projection_tracking_weight": float(projection_tracking_weight),
        "projection_collision_weight": float(projection_collision_weight),
        "projection_max_nfev": int(projection_max_nfev),
        "projection_outer_iterations": int(projection_outer_iterations),
        "strict_collision_repair": int(strict_collision_repair),
        "strict_repair_settle_steps": int(strict_repair_settle_steps),
        "strict_repair_relax_steps": int(strict_repair_relax_steps),
        "strict_repair_fallback_relax_steps": int(
            strict_repair_fallback_relax_steps
        ),
        "repair_bisection_steps": int(repair_bisection_steps),
        "repair_penetration_tolerance": float(repair_penetration_tolerance),
        "qpos_source": output_qpos_source,
        "achieved_qpos_source": f"collision_aware_mujoco_{projection_backend}",
        "contact_source": f"{output_qpos_source}_qpos_mujoco_contacts_with_optional_ik_fallback",
        "contact_schema": (
            "contact_pos/contact_mask shape is "
            "T,num_object_parts,num_hand_links,3/1; contact_links appends part id"
        ),
        "object_part_order": ",".join(context.contacts.part_names),
        "contact_part_ids": ",".join(
            str(int(part_id)) for part_id in context.contacts.part_ids
        ),
    }
    if not np.isnan(frequency):
        attrs["frequency"] = frequency
        attrs["dt"] = 1.0 / frequency

    write_h5(
        output_path,
        datasets=h5_data,
        string_datasets={
            "contact_part_names": context.contacts.part_names,
            "contact_link_names": context.contacts.hand_link_names,
        },
        attrs=attrs,
    )
    loguru.logger.info(
        f"Saved DexMachina-style H5 to {output_path} with {output_qpos.shape[0]} frames."
    )

    if show_viewer:
        loguru.logger.info("Opening MuJoCo viewer for the final retargeted trajectory.")
        replay_viewer(
            context,
            output_qpos,
            fps=viewer_fps,
            visualize_contacts=visualize_contacts,
            contact_view_mode=contact_view_mode,
            contact_view_alpha=contact_view_alpha,
            contact_pos=output_contact_pos,
            contact_mask=output_contact_mask,
            contact_visualization=contact_visualization,
            contact_point_radius=contact_point_radius,
            contact_line_width=contact_line_width,
            visualize_hand_self_contacts=visualize_hand_self_contacts,
        )

    return output_path


if __name__ == "__main__":
    tyro.cli(main)
