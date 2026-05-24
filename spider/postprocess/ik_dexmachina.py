"""
Collision-aware MuJoCo postprocess pass for DexMachina-style retargeting.

Input format: npz file named like ``trajectory_kinematic_{robot_type}.npz``.
The qpos layout is expected to match ik_mink/isaac.py:

    [robot_pos_xyz(3), robot_euler_XYZ(3), robot_finger_joints,
     object_pos_xyz(3), object_quat_wxyz(4), optional_object_joint]

This script runs a DexMachina-style collision-aware retargeting pass: robot qpos
values are used as absolute position-actuator targets, the object state is
pinned to the IK/demo state, contacts are enabled, and every frame can be solved
independently. Like DexMachina, the H5 keeps the smooth controller targets
separate from the physics-achieved qpos used for contact extraction. The default
saved replay keys use a sequential smooth MuJoCo rollout for stable
visualization/replay, while the strict collision-repaired achieved qpos is also
stored for diagnostics and contact-aware rewards.

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
    target_qpos: shape=(T, nq), float64, smooth absolute controller targets

Example:
    python spider/postprocess/ik_dexmachina.py --task scissors --embodiment-type right --dataset-dir example_datasets --dataset-name arctic --robot-type leap
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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
            tracking_residual = np.sqrt(tracking_weight) * (
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
    )


def _settle_frame_worker(
    payload: tuple[int, np.ndarray, int, int, float, Any, Any, bool],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context was not initialized.")
    (
        frame_idx,
        qpos,
        settle_steps,
        collision_relax_steps,
        max_abs_qpos,
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


def rollout_smoothed_trajectory(
    context: RetargetContext,
    target_qpos: np.ndarray,
    object_qpos_ref: np.ndarray,
    steps_per_frame: int,
    collision_relax_steps: int,
    frame_dt: float,
    max_abs_qpos: float,
    ctrl_lowpass_alpha: float,
    max_root_linear_velocity: float,
    max_root_angular_velocity: float,
    max_joint_velocity: float,
    ik_contact: np.ndarray | None,
    ik_contact_pos: np.ndarray | None,
    use_ik_contact_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optionally roll controller targets through one sequential MuJoCo simulation.

    DexMachina's retargeting output keeps controller targets separate from the
    achieved collision-aware qpos. This optional rollout follows the same
    separation: its qpos is saved as achieved state/contact evidence, not as the
    replay target trajectory.
    """
    target_qpos = np.asarray(target_qpos, dtype=np.float64)
    object_qpos_ref = np.asarray(object_qpos_ref, dtype=np.float64)
    total_frames = target_qpos.shape[0]
    num_parts = len(context.contacts.part_names)
    num_links = len(context.contacts.hand_link_names)
    smoothed_qpos = np.zeros_like(target_qpos)
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

    alpha = float(np.clip(ctrl_lowpass_alpha, 0.0, 1.0))
    if alpha <= 0.0:
        alpha = 1.0

    for frame_idx in range(total_frames):
        if frame_idx % 25 == 0:
            loguru.logger.info(f"Smoothing frame {frame_idx + 1}/{total_frames}")

        frame_target = target_qpos[frame_idx]
        object_target = object_qpos_ref[frame_idx]
        for _ in range(max(1, steps_per_frame)):
            ctrl_qpos[: context.object_qpos_start] = (
                (1.0 - alpha) * ctrl_qpos[: context.object_qpos_start]
                + alpha * frame_target[: context.object_qpos_start]
            )
            ctrl_qpos[context.object_qpos_start :] = object_target[
                context.object_qpos_start :
            ]

            pin_frozen_joints(context, data, object_target)
            set_position_targets(context, data, ctrl_qpos)
            prev_qpos = data.qpos.copy()
            mujoco.mj_step(model, data)
            pin_frozen_joints(context, data, object_target)
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
        if frame_idx > 0:
            data.qpos[:] = clip_robot_qpos_step(
                context=context,
                qpos=data.qpos,
                previous_qpos=smoothed_qpos[frame_idx - 1],
                frame_dt=frame_dt,
                max_root_linear_velocity=max_root_linear_velocity,
                max_root_angular_velocity=max_root_angular_velocity,
                max_joint_velocity=max_joint_velocity,
            )
            data.qvel[:] = 0.0
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
        smoothed_qpos[frame_idx] = data.qpos
        contact_pos[frame_idx] = cpos
        contact_mask[frame_idx] = cmask

    smoothed_qpos[:, context.object_qpos_start :] = object_qpos_ref[
        :, context.object_qpos_start :
    ]
    return smoothed_qpos, contact_pos, contact_mask


def clip_robot_qpos_step(
    context: RetargetContext,
    qpos: np.ndarray,
    previous_qpos: np.ndarray,
    frame_dt: float,
    max_root_linear_velocity: float,
    max_root_angular_velocity: float,
    max_joint_velocity: float,
) -> np.ndarray:
    clipped = qpos.copy()
    robot_width = context.object_qpos_start
    if robot_width <= 0:
        return clipped

    max_step = np.full(robot_width, max_joint_velocity * frame_dt, dtype=np.float64)
    max_step[: min(3, robot_width)] = max_root_linear_velocity * frame_dt
    if robot_width > 3:
        max_step[3 : min(6, robot_width)] = max_root_angular_velocity * frame_dt

    delta = clipped[:robot_width] - previous_qpos[:robot_width]
    clipped[:robot_width] = previous_qpos[:robot_width] + np.clip(
        delta,
        -max_step,
        max_step,
    )
    return clipped


def prepare_target_trajectory(
    context: RetargetContext,
    qpos: np.ndarray,
    frame_dt: float,
    max_root_linear_velocity: float,
    max_root_angular_velocity: float,
    max_joint_velocity: float,
) -> np.ndarray:
    """Prepare DexMachina-style absolute controller targets for saving/replay."""
    target_qpos = clamp_robot_qpos_targets(context, qpos)
    if target_qpos.shape[0] == 0:
        return target_qpos

    for frame_idx in range(1, target_qpos.shape[0]):
        target_qpos[frame_idx] = clip_robot_qpos_step(
            context=context,
            qpos=target_qpos[frame_idx],
            previous_qpos=target_qpos[frame_idx - 1],
            frame_dt=frame_dt,
            max_root_linear_velocity=max_root_linear_velocity,
            max_root_angular_velocity=max_root_angular_velocity,
            max_joint_velocity=max_joint_velocity,
        )

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
    model: mujoco.MjModel,
    qpos: np.ndarray,
    fps: int,
) -> None:
    try:
        from loop_rate_limiters import RateLimiter
    except ImportError:
        RateLimiter = None

    data = mujoco.MjData(model)
    frame_idx = 0
    rate_limiter = RateLimiter(fps) if RateLimiter is not None else None
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.qpos[:] = qpos[frame_idx]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()
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
    sim_dt: float = 0.002,
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
    smooth_rollout: bool = False,
    output_qpos_source: Literal["target", "achieved", "smooth"] = "smooth",
    rollout_steps_per_frame: int | None = 1,
    ctrl_lowpass_alpha: float = 1.0,
    smooth_output_steps_per_frame: int | None = 16,
    smooth_output_collision_relax_steps: int = 0,
    smooth_output_ctrl_lowpass_alpha: float = 1.0,
    max_root_linear_velocity: float = 0.75,
    max_root_angular_velocity: float = 3.0,
    max_joint_velocity: float = 3.0,
    use_ik_contact_fallback: bool = True,
    show_viewer: bool = True,
    viewer_fps: int = 60,
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
        frame_dt=dt,
        max_root_linear_velocity=max_root_linear_velocity,
        max_root_angular_velocity=max_root_angular_velocity,
        max_joint_velocity=max_joint_velocity,
    )
    target_delta = np.max(
        np.abs(
            target_qpos[:, : context.object_qpos_start]
            - qpos[:, : context.object_qpos_start]
        )
    )
    if target_delta > 1e-9:
        loguru.logger.info(
            "Clamped/smoothed absolute controller targets; "
            f"max robot-space change from IK input is {target_delta:.6f}."
        )

    if collision_projection:
        projection_context = context
        rollout_steps = 0
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
                ik_contact=None,
                ik_contact_pos=None,
                use_ik_contact_fallback=False,
            )
            loguru.logger.info(
                "Building conservative collision-free fallback for strict repair."
            )
            fallback_safe_qpos, _, _ = rollout_smoothed_trajectory(
                context=context,
                target_qpos=target_qpos,
                object_qpos_ref=target_qpos,
                steps_per_frame=1,
                collision_relax_steps=strict_repair_fallback_relax_steps,
                frame_dt=dt,
                max_abs_qpos=max_abs_qpos,
                ctrl_lowpass_alpha=1.0,
                max_root_linear_velocity=max_root_linear_velocity,
                max_root_angular_velocity=max_root_angular_velocity,
                max_joint_velocity=max_joint_velocity,
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
    elif smooth_rollout:
        if rollout_steps_per_frame is None:
            rollout_steps = max(1, int(round(dt / sim_dt)))
        else:
            rollout_steps = max(1, rollout_steps_per_frame)
        loguru.logger.info(
            "Running sequential smoothing rollout with "
            f"{rollout_steps} MuJoCo steps per frame."
        )
        achieved_qpos, contact_pos, contact_mask = rollout_smoothed_trajectory(
            context=context,
            target_qpos=target_qpos,
            object_qpos_ref=target_qpos,
            steps_per_frame=rollout_steps,
            collision_relax_steps=collision_relax_steps,
            frame_dt=dt,
            max_abs_qpos=max_abs_qpos,
            ctrl_lowpass_alpha=ctrl_lowpass_alpha,
            max_root_linear_velocity=max_root_linear_velocity,
            max_root_angular_velocity=max_root_angular_velocity,
            max_joint_velocity=max_joint_velocity,
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
            ik_contact=ik_contact,
            ik_contact_pos=ik_contact_pos,
            use_ik_contact_fallback=use_ik_contact_fallback,
        )
        rollout_steps = 0
        achieved_qpos = stage1_achieved_qpos
        contact_pos = stage1_contact_pos
        contact_mask = stage1_contact_mask

    achieved_contact_pos = contact_pos
    achieved_contact_mask = contact_mask
    smooth_qpos: np.ndarray | None = None
    smooth_contact_pos: np.ndarray | None = None
    smooth_contact_mask: np.ndarray | None = None
    smooth_output_steps = 0

    if output_qpos_source == "smooth":
        if smooth_output_steps_per_frame is None:
            smooth_output_steps = max(1, int(round(dt / sim_dt)))
        else:
            smooth_output_steps = max(1, smooth_output_steps_per_frame)
        loguru.logger.info(
            "Building DexMachina-style smooth replay by rolling out the "
            "collision-aware target in one MuJoCo environment "
            f"({smooth_output_steps} steps per frame)."
        )
        smooth_target_qpos = achieved_qpos.copy()
        smooth_target_qpos[:, context.object_qpos_start :] = target_qpos[
            :, context.object_qpos_start :
        ]
        smooth_qpos, smooth_contact_pos, smooth_contact_mask = rollout_smoothed_trajectory(
            context=context,
            target_qpos=smooth_target_qpos,
            object_qpos_ref=target_qpos,
            steps_per_frame=smooth_output_steps,
            collision_relax_steps=smooth_output_collision_relax_steps,
            frame_dt=dt,
            max_abs_qpos=max_abs_qpos,
            ctrl_lowpass_alpha=smooth_output_ctrl_lowpass_alpha,
            max_root_linear_velocity=max_root_linear_velocity,
            max_root_angular_velocity=max_root_angular_velocity,
            max_joint_velocity=max_joint_velocity,
            ik_contact=ik_contact,
            ik_contact_pos=ik_contact_pos,
            use_ik_contact_fallback=use_ik_contact_fallback,
        )
        output_qpos = smooth_qpos
        output_contact_pos = smooth_contact_pos
        output_contact_mask = smooth_contact_mask
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
    smooth_components = (
        None
        if smooth_qpos is None
        else split_mink_qpos(
            qpos=smooth_qpos,
            robot_type=robot_type,
            robot_joint_count=robot_joint_count,
        )
    )
    for key, value in target_components.items():
        h5_data[f"target_{key}"] = value
    for key, value in achieved_components.items():
        h5_data[f"achieved_{key}"] = value
    if smooth_components is not None:
        for key, value in smooth_components.items():
            h5_data[f"smooth_{key}"] = value

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
    smooth_penetration = (
        None if smooth_qpos is None else compute_penetration_stats(context, smooth_qpos)
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
    if smooth_qpos is not None:
        h5_data["smooth_qpos"] = smooth_qpos
        h5_data["smooth_qvel"] = compute_qvel(context.model, smooth_qpos, dt=dt)
    for key, value in output_penetration.items():
        h5_data[key] = value
    for key, value in target_penetration.items():
        h5_data[f"target_{key}"] = value
    for key, value in achieved_penetration.items():
        h5_data[f"achieved_{key}"] = value
    if smooth_penetration is not None:
        for key, value in smooth_penetration.items():
            h5_data[f"smooth_{key}"] = value

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
        "smooth_rollout": int(smooth_rollout),
        "rollout_steps_per_frame": int(rollout_steps),
        "ctrl_lowpass_alpha": float(ctrl_lowpass_alpha),
        "smooth_output_steps_per_frame": int(smooth_output_steps),
        "smooth_output_collision_relax_steps": int(
            smooth_output_collision_relax_steps
        ),
        "smooth_output_ctrl_lowpass_alpha": float(
            smooth_output_ctrl_lowpass_alpha
        ),
        "max_root_linear_velocity": float(max_root_linear_velocity),
        "max_root_angular_velocity": float(max_root_angular_velocity),
        "max_joint_velocity": float(max_joint_velocity),
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
        replay_viewer(context.model, output_qpos, fps=viewer_fps)

    return output_path


if __name__ == "__main__":
    tyro.cli(main)
