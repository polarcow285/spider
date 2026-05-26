"""
Minimal faithful DexMachina-style collision-aware retargeting post-processing.

This file mirrors the core procedure described in DexMachina (Mandi et al., 2025;
Appendix A.2) and implemented in the reference repository's
``dexmachina/retargeting/parallel_retarget.py``:

    For each demonstrated timestep, fixate the object to its target state (both
    root pose and object joint angle), set retargeted joint values as absolute
    position-actuator control targets, and step the physics simulator so that
    contacts resolve any hand-object penetration. Record the achieved joint
    values.

That is the entirety of the algorithm. There is no analytical IK projection,
no strict-collision-repair bisection, no wrist-tracking blending, and no IK
contact fallback -- those are SPIDER additions that live in
``ik_dexmachina.py``. This file is intentionally kept small so it is easy to
audit against the paper / reference repo.

Input format: ``trajectory_kinematic*.npz`` produced by the SPIDER IK stages,
with ``qpos`` arranged as

    [robot_pos(3), robot_euler_XYZ(3), robot_finger_joints,
     object_pos(3), object_quat_wxyz(4), optional_object_joint(s)]

Output format: a single ``.npz`` (same schema as the input) whose ``qpos`` is
the collision-free, MuJoCo-achieved hand trajectory. The object qpos is left
equal to the input (since the object is pinned during settling).

Example:
    python spider/postprocess/ik_dexmachina_minimal.py \
        --task scissors --embodiment-type right \
        --dataset-dir example_datasets --dataset-name arctic --robot-type leap
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import loguru
import mujoco
import mujoco.viewer
import numpy as np
import tyro

from spider.io import get_processed_data_dir


@dataclass(frozen=True)
class Context:
    """Everything we need to drive a single MuJoCo instance for retargeting."""

    model: mujoco.MjModel
    actuator_qpos_addr: np.ndarray   # qpos index that each actuator drives (-1 if N/A)
    actuator_ctrlrange: np.ndarray   # (nu, 2)
    actuator_ctrllimited: np.ndarray # (nu,) bool
    object_qpos_start: int           # first qpos index that belongs to the object
    object_qvel_start: int           # corresponding qvel index


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


def build_actuator_qpos_map(model: mujoco.MjModel) -> np.ndarray:
    """Return the qpos index each actuator drives (-1 for non-joint actuators)."""
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
    override: int | None,
) -> int:
    """Robot qpos = everything actuated; object qpos = the rest."""
    if override is not None:
        return int(override)
    valid_addr = actuator_qpos_addr[actuator_qpos_addr >= 0]
    if valid_addr.size == 0:
        raise ValueError("Could not infer object_qpos_start: no joint actuators in model.")
    return int(valid_addr.max()) + 1


def infer_object_qvel_start(model: mujoco.MjModel, object_qpos_start: int) -> int:
    """Find the qvel index of the first joint whose qpos starts at or after object_qpos_start."""
    candidates: list[int] = []
    for joint_id in range(model.njnt):
        if int(model.jnt_qposadr[joint_id]) >= object_qpos_start:
            candidates.append(int(model.jnt_dofadr[joint_id]))
    return min(candidates) if candidates else model.nv


def enable_contacts(model: mujoco.MjModel, enable_self_collision: bool) -> None:
    """Make sure collisions/constraints are enabled. Trust the asset otherwise.

    DexMachina's reference ``parallel_retarget.py`` defaults to
    ``enable_self_collision=False`` and relies on the URDF's collision setup;
    we mirror that here. Self-collision can be opted in by clearing MuJoCo's
    compiled ``exclude_signature`` table.
    """
    model.opt.disableflags = int(model.opt.disableflags) & ~int(
        mujoco.mjtDisableBit.mjDSBL_CONTACT
    )
    model.opt.disableflags = int(model.opt.disableflags) & ~int(
        mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    )
    if enable_self_collision and model.nexclude > 0:
        model.exclude_signature[:] = -1


def build_context(
    model_path: str,
    sim_dt: float,
    solver_iterations: int,
    integrator: str,
    object_qpos_start: int | None,
    enable_self_collision: bool,
) -> Context:
    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = float(sim_dt)
    model.opt.iterations = int(solver_iterations)
    integrator_map = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }
    if integrator not in integrator_map:
        raise ValueError(f"Unknown integrator `{integrator}`.")
    model.opt.integrator = integrator_map[integrator]
    enable_contacts(model, enable_self_collision=enable_self_collision)

    actuator_qpos_addr = build_actuator_qpos_map(model)
    obj_start = infer_object_qpos_start(model, actuator_qpos_addr, object_qpos_start)
    obj_qvel_start = infer_object_qvel_start(model, obj_start)
    return Context(
        model=model,
        actuator_qpos_addr=actuator_qpos_addr,
        actuator_ctrlrange=np.asarray(model.actuator_ctrlrange, dtype=np.float64).copy(),
        actuator_ctrllimited=np.asarray(model.actuator_ctrllimited, dtype=np.int32).copy(),
        object_qpos_start=obj_start,
        object_qvel_start=obj_qvel_start,
    )


def set_position_targets(
    context: Context,
    data: mujoco.MjData,
    qpos_target: np.ndarray,
) -> None:
    """Write PD targets to ``data.ctrl`` from a full-qpos target vector."""
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


def pin_object(
    context: Context,
    data: mujoco.MjData,
    qpos_target: np.ndarray,
) -> None:
    """Hold the object at its demo state (the 'fixate the object' step)."""
    data.qpos[context.object_qpos_start :] = qpos_target[context.object_qpos_start :]
    data.qvel[context.object_qvel_start :] = 0.0


def settle_one_frame(
    context: Context,
    qpos_target: np.ndarray,
    control_steps: int,
) -> np.ndarray:
    """The DexMachina single-frame procedure (one parallel env in their setup).

    1. Initialize the simulator at the kinematic IK target (cold start, matching
       DexMachina where every parallel env starts at its own demo frame).
    2. For ``control_steps`` MuJoCo steps: pin the object to its demo state,
       set absolute PD targets from the IK target, and step the simulator so
       contacts push penetrating fingers out.
    3. Return the achieved qpos. The object slice is reset to the demo state
       so the saved trajectory exactly preserves the input object motion.
    """
    model = context.model
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_target
    data.qvel[:] = 0.0
    pin_object(context, data, qpos_target)
    set_position_targets(context, data, qpos_target)
    mujoco.mj_forward(model, data)

    for _ in range(max(1, control_steps)):
        pin_object(context, data, qpos_target)
        set_position_targets(context, data, qpos_target)
        mujoco.mj_step(model, data)

    pin_object(context, data, qpos_target)
    mujoco.mj_forward(model, data)
    return data.qpos.copy()


def settle_trajectory(
    context: Context,
    qpos: np.ndarray,
    control_steps: int,
    log_every: int,
) -> np.ndarray:
    """Run the per-frame settle for every demo step (the DexMachina stage-1 loop)."""
    settled = np.empty_like(qpos)
    total_frames = qpos.shape[0]
    for frame_idx in range(total_frames):
        if frame_idx % max(1, log_every) == 0:
            loguru.logger.info(
                f"Settling frame {frame_idx + 1}/{total_frames}"
            )
        settled[frame_idx] = settle_one_frame(
            context=context,
            qpos_target=qpos[frame_idx],
            control_steps=control_steps,
        )
    return settled


def rollout_smoothing(
    context: Context,
    qpos_targets: np.ndarray,
    steps_per_frame: int,
) -> np.ndarray:
    """Optional stage-2 single-env rollout (the 'smoothed-out motions' step).

    The DexMachina documentation describes a second stage that re-runs the
    targets through one persistent simulator to smooth out the per-frame
    cold-start jumps. We do the same: state carries over between frames, only
    the object is repinned and the PD target is updated each iteration.
    """
    model = context.model
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_targets[0]
    data.qvel[:] = 0.0
    pin_object(context, data, qpos_targets[0])
    mujoco.mj_forward(model, data)

    smoothed = np.empty_like(qpos_targets)
    for frame_idx in range(qpos_targets.shape[0]):
        target = qpos_targets[frame_idx]
        for _ in range(max(1, steps_per_frame)):
            pin_object(context, data, target)
            set_position_targets(context, data, target)
            mujoco.mj_step(model, data)
        pin_object(context, data, target)
        mujoco.mj_forward(model, data)
        smoothed[frame_idx] = data.qpos.copy()
    return smoothed


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
    listing = "\n  ".join(os.path.join(processed_dir, name) for name in candidates)
    raise FileNotFoundError(f"Could not find a trajectory file. Checked:\n  {listing}")


def load_input_npz(path: str) -> tuple[np.ndarray, float]:
    with np.load(path) as trajectory:
        qpos = np.asarray(trajectory["qpos"], dtype=np.float64)
        frequency = float(trajectory["frequency"]) if "frequency" in trajectory else float("nan")
    return qpos.reshape(-1, qpos.shape[-1]), frequency


def save_output_npz(path: str, qpos: np.ndarray, frequency: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: dict[str, np.ndarray] = {"qpos": qpos.astype(np.float64)}
    if not np.isnan(frequency):
        payload["frequency"] = np.array(frequency, dtype=np.float64)
    np.savez(path, **payload)


def replay_viewer(model: mujoco.MjModel, qpos: np.ndarray, fps: int) -> None:
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
    model_path: str | None = None,
    object_qpos_start: int | None = None,
    start_idx: int = 0,
    end_idx: int = -1,
    control_steps: int = 500,
    sim_dt: float = 0.002,
    solver_iterations: int = 80,
    integrator: str = "implicitfast",
    enable_self_collision: bool = False,
    smooth_rollout: bool = False,
    smooth_steps_per_frame: int = 16,
    show_viewer: bool = True,
    viewer_fps: int = 60,
) -> str:
    """Run the minimal faithful DexMachina post-processing pass.

    Args:
        control_steps: MuJoCo steps per frame during the per-frame settle.
            Mirrors ``--control_steps`` in DexMachina's ``parallel_retarget.py``
            (their default is 2000 across GPU-parallel envs; serial MuJoCo
            converges in a few hundred for typical hands).
        enable_self_collision: Clear the compiled exclude-signature table so
            hand self-contacts can fire. DexMachina defaults this off.
        smooth_rollout: Run the optional stage-2 single-env rollout that the
            DexMachina docs describe ("smoothed-out motions"). Off by default
            so the output is purely the per-frame collision-free settle.
    """
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

    if output_path is None:
        output_path = os.path.join(
            processed_dir, f"trajectory_dexmachina_minimal_{robot_type}.npz"
        )
    output_path = os.path.abspath(output_path)

    loguru.logger.info(f"Loading IK trajectory from {trajectory_path}")
    qpos, frequency = load_input_npz(trajectory_path)
    end = None if end_idx == -1 else end_idx
    qpos = qpos[start_idx:end]
    if qpos.shape[0] == 0:
        raise ValueError("Selected trajectory slice is empty.")

    loguru.logger.info(f"Loading MuJoCo model from {model_path}")
    context = build_context(
        model_path=model_path,
        sim_dt=sim_dt,
        solver_iterations=solver_iterations,
        integrator=integrator,
        object_qpos_start=object_qpos_start,
        enable_self_collision=enable_self_collision,
    )
    if qpos.shape[1] != context.model.nq:
        raise ValueError(
            f"Trajectory qpos width {qpos.shape[1]} does not match MuJoCo model.nq "
            f"{context.model.nq} from {model_path}."
        )
    loguru.logger.info(
        f"Robot qpos: [0:{context.object_qpos_start}] | "
        f"Object qpos: [{context.object_qpos_start}:{context.model.nq}] | "
        f"Object qvel: [{context.object_qvel_start}:{context.model.nv}]"
    )

    loguru.logger.info(
        f"Stage 1: per-frame settle ({qpos.shape[0]} frames, "
        f"{control_steps} control steps each, object pinned)."
    )
    achieved = settle_trajectory(
        context=context,
        qpos=qpos,
        control_steps=control_steps,
        log_every=25,
    )
    achieved[:, context.object_qpos_start :] = qpos[:, context.object_qpos_start :]

    if smooth_rollout:
        loguru.logger.info(
            f"Stage 2: single-env smoothing rollout ({smooth_steps_per_frame} "
            "MuJoCo steps per frame)."
        )
        achieved = rollout_smoothing(
            context=context,
            qpos_targets=achieved,
            steps_per_frame=smooth_steps_per_frame,
        )
        achieved[:, context.object_qpos_start :] = qpos[:, context.object_qpos_start :]

    save_output_npz(output_path, achieved, frequency)
    loguru.logger.info(
        f"Saved collision-free trajectory to {output_path} "
        f"with {achieved.shape[0]} frames."
    )

    if show_viewer:
        loguru.logger.info("Opening MuJoCo viewer for the settled trajectory.")
        replay_viewer(context.model, achieved, fps=viewer_fps)

    return output_path


if __name__ == "__main__":
    tyro.cli(main)
