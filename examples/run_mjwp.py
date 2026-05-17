# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A standalone script to run DIAL MPC with Mujoco + Warp

Up to now, domain randomization is not supported. Will add it later.

Author: Chaoyi Pan
Date: 2025-08-11
"""

from __future__ import annotations

import time
from pathlib import Path

import os
import hydra
os.environ["MUJOCO_GL"] = "egl"
import imageio
import loguru
import mujoco
import numpy as np
import torch
from omegaconf import DictConfig


from spider.config import Config, process_config
from spider.interp import get_slice
from spider.io import load_data
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.simulators.mjwp import (
    copy_sample_state,
    get_qpos,
    get_qvel,
    get_reward,
    get_terminate,
    get_terminal_reward,
    get_trace,
    load_env_params,
    load_state,
    save_env_params,
    save_state,
    setup_env,
    setup_mj_model,  # mjwp specific
    step_env,
    sync_env,
)
from spider.viewers import render_image, setup_renderer, setup_viewer, update_viewer
from spider.viewers.rerun_viewer import log_frame


def _value_at_opt_step(info: dict, key: str) -> float:
    """Read an optimizer summary value from the last real optimization step."""
    value = np.asarray(info[key])
    opt_step = int(np.asarray(info.get("opt_steps", [value.shape[0]])).reshape(-1)[0])
    opt_idx = max(0, min(opt_step - 1, value.shape[0] - 1))
    return float(np.asarray(value[opt_idx]).reshape(-1)[0])


def save_reward_plot(info_list: list[dict], output_path: str) -> None:
    """Save a reward-over-time plot from per-control-tick optimizer infos."""
    if len(info_list) == 0 or "rew_mean" not in info_list[0]:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = []
    for step_idx, info in enumerate(info_list):
        if "time" in info:
            times.append(float(np.asarray(info["time"]).reshape(-1)[-1]))
        else:
            times.append(float(step_idx))
    times = np.asarray(times)

    reward_keys = ["rew_mean"]
    reward_keys.extend(
        sorted(
            key
            for key in info_list[0].keys()
            if key.endswith("_rew_mean") and key != "rew_mean"
        )
    )

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    for key in reward_keys:
        values = np.asarray([_value_at_opt_step(info, key) for info in info_list])
        label = "overall reward" if key == "rew_mean" else key.removesuffix("_mean")
        ax.plot(times, values, linewidth=2.0 if key == "rew_mean" else 1.4, label=label)

    ax.set_title("MJWP rewards over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    loguru.logger.info(f"Saved reward plot to {output_path}")


def main(config: Config):
    """Run the SPIDER using MuJoCo Warp backend"""
    # process config, set defaults and derived fields
    config = process_config(config)

    # load reference data (already interpolated and extended)
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(
        config, config.data_path
    )
    print(f"Loaded reference data from {config.data_path}")
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)
    config.max_sim_steps = (
        config.max_sim_steps
        if config.max_sim_steps > 0
        else qpos_ref.shape[0] - config.horizon_steps - config.ctrl_steps
    )

    # setup env with initial state from first sim qpos
    env = setup_env(config, ref_data)

    # setup mujoco (for viewer only)
    mj_model = setup_mj_model(config)
    mj_data = mujoco.MjData(mj_model)
    mj_data_ref = mujoco.MjData(mj_model)
    mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
    mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0
    images = []
    object_trace_site_ids = []
    robot_trace_site_ids = []
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is not None:
            if name.startswith("trace"):
                if "object" in name:
                    object_trace_site_ids.append(sid)
                else:
                    robot_trace_site_ids.append(sid)
    config.trace_site_ids = object_trace_site_ids + robot_trace_site_ids

    # setup env params
    env_params_list = []
    if config.num_dr == 0:
        xy_offset_list = [0.0]
        pair_margin_list = [0.0]
    else:
        xy_offset_list = np.linspace(
            config.xy_offset_range[0], config.xy_offset_range[1], config.num_dr
        )
        pair_margin_list = np.linspace(
            config.pair_margin_range[0], config.pair_margin_range[1], config.num_dr
        )
    for i in range(config.max_num_iterations):
        env_params = []
        for j in range(config.num_dr):
            env_params.append(
                {"xy_offset": xy_offset_list[j], "pair_margin": pair_margin_list[j]}
            )
        env_params_list.append(env_params)
    config.env_params_list = env_params_list

    # setup viewer and renderer
    run_viewer = setup_viewer(config, mj_model, mj_data)
    renderer = setup_renderer(config, mj_model)

    # setup optimizer
    rollout = make_rollout_fn(
        step_env,
        save_state,
        load_state,
        get_reward,
        get_terminal_reward,
        get_terminate,
        get_trace,
        save_env_params,
        load_env_params,
        copy_sample_state,
    )

    optimize_once = make_optimize_once_fn(rollout)
    optimize = make_optimize_fn(optimize_once)

    # initial controls
    ctrls = ctrl_ref[: config.horizon_steps]
    # buffers for saving info and trajectory
    info_list = []

    # run viewer + control loop
    t_start = time.perf_counter()
    print("starting loop")
    with run_viewer() as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()

            # optimize using future reference window at control-rate (+1 lookahead)
            sim_step = int(np.round(mj_data.time / config.sim_dt))
            ref_slice = get_slice(
                ref_data, sim_step + 1, sim_step + config.horizon_steps + 1
            )
            ctrls, infos = optimize(config, env, ctrls, ref_slice)
            print("Finished optimizing")

            # step environment for ctrl_steps
            step_info = {"qpos": [], "qvel": [], "time": [], "ctrl": []}
            for i in range(config.ctrl_steps):
                # option 1: use mujoco step
                # mj_data.ctrl[:] = ctrls[i].detach().cpu().numpy()
                # mujoco.mj_step(mj_model, mj_data)
                # option 2: use warp step
                step_env(config, env, ctrls[i : i + 1])
                mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
                mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
                mj_data.ctrl[:] = ctrls[i].detach().cpu().numpy()
                mj_data.time += config.sim_dt
                if config.save_video and renderer is not None:
                    if i % int(np.round(config.render_dt / config.sim_dt)) == 0:
                        mj_data_ref.qpos[:] = (
                            qpos_ref[sim_step + i].detach().cpu().numpy()
                        )
                        image = render_image(
                            config, renderer, mj_model, mj_data, mj_data_ref
                        )
                        images.append(image)
                if "rerun" in config.viewer:
                    # manually log the state
                    log_frame(
                        mj_data,
                        sim_time=mj_data.time,
                        viewer_body_entity_and_ids=config.viewer_body_entity_and_ids,
                    )
                step_info["qpos"].append(mj_data.qpos.copy())
                step_info["qvel"].append(mj_data.qvel.copy())
                step_info["time"].append(mj_data.time)
                step_info["ctrl"].append(mj_data.ctrl.copy())
            for k in step_info:
                step_info[k] = np.stack(step_info[k], axis=0)
            infos.update(step_info)
            # sync env state
            sync_env(config, env, mj_data)

            # receding horizon update
            sim_step = int(np.round(mj_data.time / config.sim_dt))
            prev_ctrl = ctrls[config.ctrl_steps :]
            new_ctrl = ctrl_ref[
                sim_step + prev_ctrl.shape[0] : sim_step
                + prev_ctrl.shape[0]
                + config.ctrl_steps
            ]
            ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)

            # sync viewer state and render
            mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
            mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
            mj_data_ref.qpos[:] = qpos_ref[sim_step].detach().cpu().numpy()
            update_viewer(config, viewer, mj_model, mj_data, mj_data_ref, infos)

            # progress
            t1 = time.perf_counter()
            rtr = config.ctrl_dt / (t1 - t0)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {t1 - t0:.4f}s, sim_steps: {sim_step}/{config.max_sim_steps}, opt_steps: {infos['opt_steps'][0]}",
                end="\r",
            )

            # record info/trajectory at control tick
            # rule out "trace"
            info_list.append({k: v for k, v in infos.items() if k != "trace_sample"})

            if sim_step >= config.max_sim_steps:
                break

        t_end = time.perf_counter()
        print(f"Total time: {t_end - t_start:.4f}s")
    save_name = f"num_samples{config.num_samples}"
    # save retargeted trajectory
    if config.save_info and len(info_list) > 0:
        info_aggregated = {}
        for k in info_list[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in info_list], axis=0)
        np.savez(f"{config.output_dir}/{save_name}_trajectory_mjwp.npz", **info_aggregated)
        loguru.logger.info(f"Saved info to {config.output_dir}/{save_name}_trajectory_mjwp.npz")

    # save video
    if config.save_video and len(images) > 0:
        video_path = f"{config.output_dir}/{save_name}_visualization_mjwp.mp4"
        imageio.mimsave(
            video_path,
            images,
            fps=int(1 / config.render_dt),
        )
        loguru.logger.info(f"Saved video to {video_path}")

    # save reward plot
    if len(info_list) > 0:
        save_reward_plot(
            info_list,
            f"{config.output_dir}/visualization_mjwp.png",
        )

    return


@hydra.main(version_base=None, config_path="config", config_name="default")
def run_main(cfg: DictConfig) -> None:
    # Convert DictConfig to Config dataclass, handling special fields
    config_dict = dict(cfg)

    # Handle special conversions
    if "noise_scale" in config_dict and config_dict["noise_scale"] is None:
        config_dict.pop("noise_scale")  # Let the default factory handle it

    # Convert lists to tuples where needed
    if "pair_margin_range" in config_dict:
        config_dict["pair_margin_range"] = tuple(config_dict["pair_margin_range"])
    if "xy_offset_range" in config_dict:
        config_dict["xy_offset_range"] = tuple(config_dict["xy_offset_range"])

    config = Config(**config_dict)
    main(config)


if __name__ == "__main__":
    run_main()
