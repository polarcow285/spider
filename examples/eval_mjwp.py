# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Run one MJWP retargeting job and evaluate the saved trajectory.

This is intentionally a thin wrapper around examples/run_mjwp.py. It keeps the
retargeting implementation in one place, converts the produced MJWP qpos result
into the real-world evaluator trajectory format, and appends one CSV row.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from spider.config import Config, process_config
from spider.interp import interp

EVAL_TASK_CHOICES = (
    "notebook",
    "scissors",
    "real_scissors",
    "screwdriver",
    "screwdriver2",
    "hammer",
)
VERBOSE_TASKS = frozenset(
    {"screwdriver", "screwdriver2", "real_scissors", "scissors"}
)
TASKS_WITH_OBJECT_JOINT = frozenset({"notebook", "scissors", "real_scissors"})
DEFAULT_DEXRL_DATA_DIR = Path("/share/culbertson/nl455/dexrl/data")
DEFAULT_EVAL_MODULE_DIR = Path("/share/culbertson/nl455/dexrl/scripts/real_world")
CSV_FIELDS = [
    "timestamp_utc",
    "seed",
    "task",
    "eval_task",
    "robot_type",
    "embodiment_type",
    "dataset_name",
    "data_id",
    "error_m",
    "error_mm",
    "success",
    "num_steps",
    "result_npz_path",
    "reference_data_path",
    "model_path",
    "output_dir",
    "csv_path",
    "sim_dt",
    "ctrl_dt",
    "ref_dt",
    "horizon",
    "horizon_steps",
    "ctrl_steps",
    "max_sim_steps",
    "num_samples",
    "max_num_iterations",
    "temperature",
    "terminal_rew_scale",
    "pos_rew_scale",
    "rot_rew_scale",
    "base_pos_rew_scale",
    "base_rot_rew_scale",
    "joint_rew_scale",
    "vel_rew_scale",
    "object_pos_threshold",
    "object_rot_threshold",
    "use_mink",
    "skip_run",
    "elapsed_s",
    "max_eval_steps",
    "run_command",
    "hydra_overrides",
    "eval_module_dir",
    "dexrl_data_dir",
    "parameters_json",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse wrapper args and leave remaining tokens for Hydra/run_mjwp.py."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run examples/run_mjwp.py once, evaluate its *_trajectory_mjwp.npz, "
            "and append one CSV row. Pass Hydra overrides after the wrapper args."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for this run.")
    parser.add_argument(
        "--eval-task",
        choices=EVAL_TASK_CHOICES,
        default=None,
        help="Metric task key. Inferred from the SPIDER task when omitted.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("mjwp_eval_results.csv"),
        help="CSV file to append one row to.",
    )
    parser.add_argument(
        "--result-npz",
        type=Path,
        default=None,
        help="Evaluate this MJWP result instead of auto-detecting the newest one.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not run MJWP; evaluate --result-npz or the newest result in output_dir.",
    )
    parser.add_argument(
        "--max-eval-steps",
        type=int,
        default=None,
        help="Optional cap for quick debugging; full trajectory is evaluated by default.",
    )
    parser.add_argument(
        "--run-mjwp-path",
        type=Path,
        default=repo_root / "examples" / "run_mjwp.py",
        help="Path to the existing MJWP entrypoint.",
    )
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=repo_root / "examples" / "config" / "default.yaml",
        help="Default YAML used to reconstruct SPIDER config metadata.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run examples/run_mjwp.py.",
    )
    parser.add_argument(
        "--eval-module-dir",
        type=Path,
        default=DEFAULT_EVAL_MODULE_DIR,
        help="Directory containing eval_recorded_traj.py from the reference evaluator.",
    )
    parser.add_argument(
        "--dexrl-data-dir",
        type=Path,
        default=None,
        help="Directory containing evaluator keypoints/.npy assets.",
    )
    args, run_args = parser.parse_known_args()
    if run_args and run_args[0] == "--":
        run_args = run_args[1:]
    return args, run_args


def has_dotlist_override(overrides: list[str], key: str) -> bool:
    """Return whether a Hydra dotlist override assigns a given key."""
    prefixes = (f"{key}=", f"+{key}=", f"++{key}=", f"~{key}")
    return any(override.startswith(prefixes) for override in overrides)


def prepare_run_args(seed: int, run_args: list[str]) -> tuple[list[str], bool]:
    """Add seed to Hydra overrides and separate the custom --mink flag."""
    use_mink = "--mink" in run_args
    hydra_overrides = [arg for arg in run_args if arg != "--mink"]
    unsupported = [arg for arg in hydra_overrides if arg.startswith("-")]
    if unsupported:
        raise ValueError(
            "Only Hydra dotlist overrides and run_mjwp.py's --mink flag are "
            f"supported here. Unsupported args: {unsupported}"
        )
    if not has_dotlist_override(hydra_overrides, "seed"):
        hydra_overrides.append(f"seed={seed}")
    run_args_prepared = [*hydra_overrides]
    if use_mink:
        run_args_prepared.append("--mink")
    return run_args_prepared, use_mink


def cfg_dict_from_yaml(config_yaml: Path, hydra_overrides: list[str]) -> dict[str, Any]:
    """Load default.yaml plus simple Hydra dotlist overrides into Config kwargs."""
    base = OmegaConf.load(config_yaml)
    override_cfg = OmegaConf.from_dotlist(hydra_overrides)
    merged = OmegaConf.merge(base, override_cfg)
    cfg_dict = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"Expected mapping config from {config_yaml}")

    # Hydra removes these before run_mjwp.py constructs Config; OmegaConf.load does not.
    cfg_dict.pop("defaults", None)
    cfg_dict.pop("hydra", None)

    if cfg_dict.get("noise_scale") is None:
        cfg_dict.pop("noise_scale", None)
    for key in ("pair_margin_range", "xy_offset_range"):
        if key in cfg_dict:
            cfg_dict[key] = tuple(cfg_dict[key])
    return cfg_dict


def build_config(config_yaml: Path, hydra_overrides: list[str], use_mink: bool) -> Config:
    """Reconstruct the same SPIDER Config path/output metadata used by run_mjwp.py."""
    cfg_dict = cfg_dict_from_yaml(config_yaml, hydra_overrides)
    config = Config(**cfg_dict)
    requested_device = config.device

    # Metadata/reference reconstruction should not require a GPU just because the
    # run itself uses one.
    config.device = "cpu"
    config = process_config(config)
    config.device = requested_device

    if use_mink:
        config.data_path = str(
            Path(config.data_path).with_name(
                f"trajectory_kinematic_mink_{config.robot_type}_{config.task}.npz"
            )
        )
    return config


def infer_eval_task(task_name: str) -> str:
    """Infer the evaluator task key from the SPIDER task name."""
    task_lower = task_name.lower()
    if "screwdriver2" in task_lower:
        return "screwdriver2"
    if "screwdriver" in task_lower:
        return "screwdriver"
    if "real_scissors" in task_lower:
        return "real_scissors"
    if "scissors" in task_lower:
        return "scissors"
    if "notebook" in task_lower:
        return "notebook"
    if "hammer" in task_lower:
        return "hammer"
    raise ValueError(
        f"Could not infer --eval-task from SPIDER task {task_name!r}; pass --eval-task."
    )


def import_reference_evaluator(eval_module_dir: Path):
    """Import eval_recorded_traj.py from the provided reference directory."""
    if eval_module_dir and eval_module_dir.exists():
        sys.path.insert(0, str(eval_module_dir))
    try:
        from eval_recorded_traj import EVAL_FNS  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import eval_recorded_traj.py. Pass --eval-module-dir or "
            "put the reference real_world directory on PYTHONPATH."
        ) from exc
    return EVAL_FNS


def configure_dexrl_data_dir(dexrl_data_dir: Path | None) -> Path:
    """Set DEXRL_DATA_DIR so the reference evaluator can load keypoint assets."""
    if dexrl_data_dir is not None:
        resolved = dexrl_data_dir
    elif "DEXRL_DATA_DIR" in os.environ:
        resolved = Path(os.environ["DEXRL_DATA_DIR"])
    elif DEFAULT_DEXRL_DATA_DIR.exists():
        resolved = DEFAULT_DEXRL_DATA_DIR
    else:
        resolved = Path("data")
    os.environ["DEXRL_DATA_DIR"] = str(resolved)
    return resolved


def run_mjwp(args: argparse.Namespace, run_args: list[str]) -> tuple[str, float]:
    """Run the existing MJWP entrypoint and return the command plus elapsed seconds."""
    cmd = [args.python, str(args.run_mjwp_path), *run_args]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    started = time.perf_counter()
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], env=env, check=True)
    elapsed_s = time.perf_counter() - started
    return shlex.join(cmd), elapsed_s


def find_result_npz(output_dir: Path, started_wall_time: float | None = None) -> Path:
    """Find the newest MJWP trajectory result in output_dir."""
    candidates = sorted(output_dir.glob("*_trajectory_mjwp.npz"))
    if not candidates:
        raise FileNotFoundError(f"No *_trajectory_mjwp.npz files found in {output_dir}")
    if started_wall_time is not None:
        recent = [p for p in candidates if p.stat().st_mtime >= started_wall_time - 1.0]
        if recent:
            candidates = recent
    return max(candidates, key=lambda path: path.stat().st_mtime)


def flatten_steps(array: np.ndarray) -> np.ndarray:
    """Flatten a per-control-tick saved array into a per-sim-step array."""
    arr = np.asarray(array)
    if arr.ndim <= 2:
        return arr
    return arr.reshape(-1, arr.shape[-1])


def flatten_time(array: np.ndarray | None, num_steps: int) -> np.ndarray | None:
    """Flatten saved time values when present."""
    if array is None:
        return None
    arr = np.asarray(array).reshape(-1)
    if len(arr) < num_steps:
        return None
    return arr[:num_steps]


def joint_qpos_width(model: mujoco.MjModel, joint_id: int) -> int:
    """Return qpos width for a MuJoCo joint."""
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def find_named_joint_qpos_addr(model: mujoco.MjModel, names: list[str]) -> int | None:
    """Return qpos address for the first existing named joint."""
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            return int(model.jnt_qposadr[joint_id])
    return None


def find_object_root_qpos_addr(model: mujoco.MjModel, embodiment_type: str) -> int:
    """Find the qpos address of the object root free joint."""
    body_names = ["bottom", "object"]
    if embodiment_type == "right":
        body_names.extend(["right_object", "right_object_collision"])
    elif embodiment_type == "left":
        body_names.extend(["left_object", "left_object_collision"])
    elif embodiment_type == "bimanual":
        body_names.extend(["right_object", "left_object", "right_object_collision"])

    for body_name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0 or model.body_jntnum[body_id] < 1:
            continue
        joint_id = int(model.body_jntadr[body_id])
        if joint_qpos_width(model, joint_id) == 7:
            return int(model.jnt_qposadr[joint_id])

    # Last-resort fallback matching current MJWP right/left object layout.
    if model.nq >= 7:
        return int(model.nq - 7)
    raise ValueError("Could not find object root free joint in the MuJoCo model.")


def find_object_joint_qpos_addr(
    model: mujoco.MjModel,
    eval_task: str,
    root_qpos_addr: int,
) -> int | None:
    """Find optional articulated object joint qpos address for evaluator tasks."""
    if eval_task not in TASKS_WITH_OBJECT_JOINT:
        return None

    task_candidates = {
        "notebook": ["notebook_joint", "object_joint"],
        "scissors": ["scissors_joint", "object_joint"],
        "real_scissors": ["scissors_joint", "object_joint"],
    }
    qpos_addr = find_named_joint_qpos_addr(model, task_candidates.get(eval_task, []))
    if qpos_addr is not None:
        return qpos_addr

    min_addr = root_qpos_addr + 7
    for joint_id in range(model.njnt):
        addr = int(model.jnt_qposadr[joint_id])
        if addr < min_addr or joint_qpos_width(model, joint_id) != 1:
            continue
        return addr

    raise ValueError(
        f"Evaluator task {eval_task!r} requires object_joint, but no articulated "
        "object joint was found in the MuJoCo model."
    )


def object_state_from_qpos(
    qpos: np.ndarray,
    root_qpos_addr: int,
    joint_qpos_addr: int | None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Extract object root pose and optional object joint from one qpos row."""
    pose = np.asarray(qpos[root_qpos_addr : root_qpos_addr + 7], dtype=np.float64)
    if pose.shape[0] != 7:
        raise ValueError("Object root qpos slice does not contain 7 values.")
    quat = pose[3:7].copy()
    quat_norm = np.linalg.norm(quat)
    if quat_norm > 0.0:
        quat /= quat_norm
    joint = None
    if joint_qpos_addr is not None:
        joint = float(np.asarray(qpos[joint_qpos_addr]).reshape(-1)[0])
    return pose[:3].copy(), quat, joint


def reference_indices(times: np.ndarray | None, num_steps: int, sim_dt: float, max_len: int) -> np.ndarray:
    """Map saved MJWP step times to reference qpos indices."""
    if times is None:
        indices = np.arange(1, num_steps + 1, dtype=np.int64)
    else:
        indices = np.rint(times / sim_dt).astype(np.int64)
    return np.clip(indices, 0, max_len - 1)


def load_reference_qpos(config: Config) -> np.ndarray:
    """Load reference qpos with load_data-compatible interpolation and padding."""
    raw_data = np.load(config.data_path)
    qpos_ref = torch.from_numpy(raw_data["qpos"]).to(torch.float32)
    if config.ref_dt > config.sim_dt:
        qpos_ref = interp(qpos_ref.unsqueeze(0), config.ref_steps).squeeze(0)
    else:
        downsample_factor = int(config.sim_dt / config.ref_dt)
        qpos_ref = qpos_ref[::downsample_factor]

    pad_steps = config.horizon_steps + config.ctrl_steps
    if pad_steps > 0:
        qpos_ref = torch.cat([qpos_ref, qpos_ref[-1:].repeat(pad_steps, 1)], dim=0)
    return qpos_ref.cpu().numpy()


def build_eval_trajectory(
    result_npz_path: Path,
    config: Config,
    eval_task: str,
    max_eval_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Convert a saved MJWP result .npz into eval_recorded_traj.py's format."""
    result = np.load(result_npz_path)
    if "qpos" not in result:
        raise KeyError(f"{result_npz_path} does not contain a 'qpos' array.")
    qpos_sim = flatten_steps(result["qpos"])
    if max_eval_steps is not None:
        qpos_sim = qpos_sim[:max_eval_steps]
    times = flatten_time(
        result["time"] if "time" in result else None,
        qpos_sim.shape[0],
    )
    if times is not None and max_eval_steps is not None:
        times = times[:max_eval_steps]

    qpos_ref_np = load_reference_qpos(config)
    indices = reference_indices(
        times,
        qpos_sim.shape[0],
        config.sim_dt,
        qpos_ref_np.shape[0],
    )

    model = mujoco.MjModel.from_xml_path(config.model_path)
    root_qpos_addr = find_object_root_qpos_addr(model, config.embodiment_type)
    joint_qpos_addr = find_object_joint_qpos_addr(model, eval_task, root_qpos_addr)

    traj: list[dict[str, Any]] = []
    for sim_qpos, ref_idx in zip(qpos_sim, indices, strict=False):
        ref_qpos = qpos_ref_np[ref_idx]
        object_pos, object_quat, object_joint = object_state_from_qpos(
            sim_qpos,
            root_qpos_addr,
            joint_qpos_addr,
        )
        ref_object_pos, ref_object_quat, ref_object_joint = object_state_from_qpos(
            ref_qpos,
            root_qpos_addr,
            joint_qpos_addr,
        )
        traj.append(
            {
                "object_pos": object_pos,
                "object_quat": object_quat,
                "object_joint": object_joint,
                "ref_object_pos_sim": ref_object_pos,
                "ref_object_quat_sim": ref_object_quat,
                "ref_object_joint": ref_object_joint,
            }
        )
    return traj


def serializable_config(config: Config) -> dict[str, Any]:
    """Return compact JSON-serializable config metadata."""
    data = asdict(config)
    noise_scale = data.pop("noise_scale", None)
    if isinstance(noise_scale, torch.Tensor):
        data["noise_scale_shape"] = list(noise_scale.shape)
        data["noise_scale_device"] = str(noise_scale.device)
    else:
        data["noise_scale"] = None
    return data


def append_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append one row to the evaluation CSV, writing a header if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def make_csv_row(
    *,
    args: argparse.Namespace,
    config: Config,
    eval_task: str,
    err: float,
    success: bool,
    num_steps: int,
    result_npz_path: Path,
    run_command: str,
    hydra_overrides: list[str],
    use_mink: bool,
    elapsed_s: float,
    dexrl_data_dir: Path,
) -> dict[str, Any]:
    """Build the CSV payload for one evaluated run."""
    config_json = json.dumps(serializable_config(config), sort_keys=True, default=str)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "task": config.task,
        "eval_task": eval_task,
        "robot_type": config.robot_type,
        "embodiment_type": config.embodiment_type,
        "dataset_name": config.dataset_name,
        "data_id": config.data_id,
        "error_m": err,
        "error_mm": err * 1000.0,
        "success": bool(success),
        "num_steps": num_steps,
        "result_npz_path": str(result_npz_path),
        "reference_data_path": config.data_path,
        "model_path": config.model_path,
        "output_dir": config.output_dir,
        "csv_path": str(args.csv_path),
        "sim_dt": config.sim_dt,
        "ctrl_dt": config.ctrl_dt,
        "ref_dt": config.ref_dt,
        "horizon": config.horizon,
        "horizon_steps": config.horizon_steps,
        "ctrl_steps": config.ctrl_steps,
        "max_sim_steps": config.max_sim_steps,
        "num_samples": config.num_samples,
        "max_num_iterations": config.max_num_iterations,
        "temperature": config.temperature,
        "terminal_rew_scale": config.terminal_rew_scale,
        "pos_rew_scale": config.pos_rew_scale,
        "rot_rew_scale": config.rot_rew_scale,
        "base_pos_rew_scale": config.base_pos_rew_scale,
        "base_rot_rew_scale": config.base_rot_rew_scale,
        "joint_rew_scale": config.joint_rew_scale,
        "vel_rew_scale": config.vel_rew_scale,
        "object_pos_threshold": config.object_pos_threshold,
        "object_rot_threshold": config.object_rot_threshold,
        "use_mink": use_mink,
        "skip_run": args.skip_run,
        "elapsed_s": elapsed_s,
        "max_eval_steps": args.max_eval_steps,
        "run_command": run_command,
        "hydra_overrides": " ".join(hydra_overrides),
        "eval_module_dir": str(args.eval_module_dir),
        "dexrl_data_dir": str(dexrl_data_dir),
        "parameters_json": config_json,
    }


def main() -> None:
    """Run one MJWP eval and append metrics to CSV."""
    args, raw_run_args = parse_args()
    run_args, use_mink = prepare_run_args(args.seed, raw_run_args)
    hydra_overrides = [arg for arg in run_args if arg != "--mink"]
    config = build_config(args.config_yaml, hydra_overrides, use_mink)
    eval_task = args.eval_task or infer_eval_task(config.task)

    dexrl_data_dir = configure_dexrl_data_dir(args.dexrl_data_dir)
    eval_fns = import_reference_evaluator(args.eval_module_dir)
    if eval_task not in eval_fns:
        raise KeyError(f"No evaluator registered for task {eval_task!r}")

    started_wall_time = time.time()
    run_command = ""
    elapsed_s = 0.0
    if not args.skip_run:
        run_command, elapsed_s = run_mjwp(args, run_args)

    if args.result_npz is not None:
        result_npz_path = args.result_npz
        if not result_npz_path.exists():
            raise FileNotFoundError(result_npz_path)
    else:
        started = None if args.skip_run else started_wall_time
        result_npz_path = find_result_npz(Path(config.output_dir), started)

    traj = build_eval_trajectory(
        result_npz_path,
        config,
        eval_task,
        max_eval_steps=args.max_eval_steps,
    )
    kwargs = {"verbose": False} if eval_task in VERBOSE_TASKS else {}
    err, success = eval_fns[eval_task](traj, eval_task, **kwargs)

    row = make_csv_row(
        args=args,
        config=config,
        eval_task=eval_task,
        err=float(err),
        success=bool(success),
        num_steps=len(traj),
        result_npz_path=result_npz_path,
        run_command=run_command,
        hydra_overrides=hydra_overrides,
        use_mink=use_mink,
        elapsed_s=elapsed_s,
        dexrl_data_dir=dexrl_data_dir,
    )
    append_csv_row(args.csv_path, row)

    print(f"Result: {result_npz_path}")
    print(f"Task: {eval_task}")
    print(f"Error: {float(err):.6f} m ({float(err) * 1000.0:.2f} mm)")
    print(f"Success: {bool(success)}")
    print(f"Steps: {len(traj)}")
    print(f"CSV: {args.csv_path}")


if __name__ == "__main__":
    main()
