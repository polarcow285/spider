"""
Convert the custom mocap demo data to mjwp format.

Input format: h5 file
Keys: ['mano_joint_coords', 'wrist_pos', 'wrist_quat', 'wrist_rot_mat', 'obj_pos', 'obj_quat']
      : shape=(271, 21, 3), dtype=float32
    wrist_pos: shape=(271, 3), dtype=float32
    wrist_quat: shape=(271, 4), dtype=float64
    wrist_rot_mat: shape=(271, 3, 3), dtype=float32
    obj_pos: shape=(271, 3), dtype=float64
    obj_quat: shape=(271, 4), dtype=float64d

Output format: npz file 'trajectory_keypoint.npz'
Keys: ['qpos_wrist_right', 'qpos_finger_right', 'qpos_wrist_left', 'qpos_finger_left', 'qpos_obj_right', 'qpos_obj_left', 'contact', 'contact_pos']
    qpos_wrist_right: shape=(T, 7), dtype=float32
    qpos_finger_right: shape=(T, 5, 7), dtype=float32 (5 fingertips)
    qpos_obj_right: shape=(T, 7), dtype=float32

Example usage:
    python spider/process_datasets/custom.py --task hammer --embodiment-type right --dataset-dir example_datasets
    python spider/process_datasets/custom.py --task screwdriver --scale 1.5 --embodiment-type right --dataset-dir example_datasets

"""
import json
import os
from contextlib import contextmanager

import h5py
import numpy as np
import tyro
import loguru
import spider
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
from spider.io import get_processed_data_dir

# visual mesh per object, relative to processed/custom/assets/objects; None -> box primitive
OBJECT_MESHES = {
    "screwdriver": "screwdriver/screwdriver.obj",
    "hammer": "hammer/hammer_simplified/hammer_simplified.obj",
    "eraser": None,
}
ERASER_BOX_HALF_SIZE = [0.025, 0.0625, 0.015]  # from regrind assets/eraser/eraser.xml

def main(
    dataset_dir: str = "../../example_datasets",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    robot_type: str | None = None,  # set only if a robot needs its own demo
    scale: float = 1.0,  # e.g. 1.5 -> raw/custom/screwdriver_1_5x
    show_viewer: bool = True,
    save_video: bool = False,
    start_idx: int = 0,
):
    dataset_dir = os.path.abspath(dataset_dir)
    scale_suffix = "" if scale == 1.0 else f"_{scale:g}x".replace(".", "_")
    variant = f"{task}{scale_suffix}"
    file_path = f"{dataset_dir}/raw/custom/{variant}/demo_30fps.h5"
    output_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name="custom",
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=variant,
        data_id=0,
    )
    os.makedirs(output_dir, exist_ok=True)

    # task info
    mesh_rel = OBJECT_MESHES[task]
    right_object_mesh_dir = os.path.join(
        dataset_dir,
        "processed",
        "custom",
        "assets",
        "objects",
        task,
    )
    task_info = {
        "task": task,
        "dataset_name": "custom",
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": 0,
        "right_object_mesh_dir": right_object_mesh_dir,
        "left_object_mesh_dir": None,
        "ref_dt": 0.02,
    }

    # read data
    with h5py.File(file_path, "r") as f:
        # Standard MANO order: [Wrist, Thumb(1-4), Index(5-8), Middle(9-12), Ring(13-16), Pinky(17-20)]
        mano_keypoints = f["mano_joint_coords"][:]   # (T, 21, 3)
        wrist_pos = f["wrist_pos"][:]                # (T, 3)
        wrist_quat = f["wrist_quat"][:]              # (T, 4)
        obj_pos = f["obj_pos"][:]                    # (T, 3)
        obj_quat = f["obj_quat"][:]                  # (T, 4)
    N = mano_keypoints.shape[0]

    tip_ids = [16, 17, 18, 19, 20]
    unit_quat = np.array([1, 0, 0, 0])
    qpos_wrist_right = np.zeros((N, 7))
    qpos_finger_right = np.zeros((N, 5, 7))
    qpos_obj_right = np.zeros((N, 7))
    qpos_wrist_left = np.zeros((N, 7))
    qpos_finger_left = np.zeros((N, 5, 7))
    qpos_obj_left = np.zeros((N, 7))

    for j, tip_id in enumerate(tip_ids):
        qpos_finger_right[:, j, :3] = mano_keypoints[:, tip_id, :]
        qpos_finger_right[:, j, 3:] = unit_quat



    qpos_wrist_right = np.concatenate([wrist_pos, wrist_quat], axis=1).astype(np.float32)  # (T, 7)

    # ---- Object (right) ----
    qpos_obj_right = np.concatenate([obj_pos, obj_quat], axis=1).astype(np.float32)  # (T, 7)

    keypoints_name = "trajectory_keypoints.npz" if robot_type is None else f"trajectory_keypoints_{robot_type}.npz"
    np.savez(
        f"{output_dir}/{keypoints_name}",
        qpos_wrist_right=qpos_wrist_right[start_idx:],
        qpos_finger_right=qpos_finger_right[start_idx:],
        qpos_obj_right=qpos_obj_right[start_idx:],
        qpos_wrist_left=qpos_wrist_left[start_idx:],
        qpos_finger_left=qpos_finger_left[start_idx:],
        qpos_obj_left=qpos_obj_left[start_idx:],
    )
    loguru.logger.info(f"Saved qpos to {output_dir}/{keypoints_name}")

    task_info_path = f"{output_dir}/../task_info.json"
    with open(task_info_path, "w") as f:
        json.dump(task_info, f, indent=2)
    loguru.logger.info(f"Saved task_info to {task_info_path}")

    qpos_list = np.concatenate(
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
    # visualize
    mj_spec = mujoco.MjSpec.from_file(f"{spider.ROOT}/assets/mano/empty_scene.xml")
     # add right object to body "right_object"
    object_right_handle = mj_spec.worldbody.add_body(
        name="right_object",
        mocap=True,
    )
    object_right_handle.add_site(
        name="right_object",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.02, 0.03],
        rgba=[1, 0, 0, 1],
        group=0,
    )

    if embodiment_type in ["right", "bimanual"]:
        if mesh_rel is None:
            object_right_handle.add_geom(
                name="right_object",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[s * scale for s in ERASER_BOX_HALF_SIZE],
                group=0,
                condim=1,
            )
        else:
            mj_spec.add_mesh(
                name="right_object",
                file=os.path.join(dataset_dir, "processed", "custom", "assets", "objects", mesh_rel),
                scale=[scale] * 3,
            )
            object_right_handle.add_geom(
                name="right_object",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="right_object",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                group=0,
                condim=1,
            )
    # add left object to body "left_object"
    object_left_handle = mj_spec.worldbody.add_body(
        name="left_object",
        mocap=True,
    )
    object_left_handle.add_site(
        name="left_object",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.02, 0.03],
        rgba=[0, 1, 0, 1],
        group=0,
    )

    mj_model = mj_spec.compile()
    mj_data = mujoco.MjData(mj_model)
    rate_limiter = RateLimiter(30.0)
    if show_viewer:
        run_viewer = lambda: mujoco.viewer.launch_passive(mj_model, mj_data)
    else:

        @contextmanager
        def run_viewer():
            yield type(
                "DummyViewer",
                (),
                {
                    "is_running": lambda: True,
                    "sync": lambda: None,
                    "cam": mujoco.MjvCamera(),
                },
            )

    if save_video:
        import imageio

        mj_model.vis.global_.offwidth = 720
        mj_model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(mj_model, height=480, width=720)
        images = []

    print("\n=== Mocap bodies in model ===")

    for i in range(mj_model.nbody):
        body = mj_model.body(i)

        if body.mocapid[0] != -1:
            print(
                "body:",
                body.name,
                "mocap_id:",
                body.mocapid[0],
            )
    with run_viewer() as gui:
        cnt = 0
        contact_seq = np.zeros((N, 10))
        while gui.is_running():
            mj_data.mocap_pos[:] = qpos_list[cnt, :, :3]
            mj_data.mocap_quat[:] = qpos_list[cnt, :, 3:]
            mujoco.mj_step(mj_model, mj_data)
            cnt = (cnt + 1) % N
            if save_video:
                renderer.update_scene(mj_data, gui.cam)
                img = renderer.render()
                images.append(img)
            if cnt == (N - 1):
                if save_video:
                    imageio.mimsave(f"{output_dir}/visualization.mp4", images, fps=120)
                    loguru.logger.info(f"Saved video to {output_dir}/visualization.mp4")
                if not show_viewer:
                    break
            if show_viewer:
                gui.sync()
                rate_limiter.sleep()

if __name__ == "__main__":
    tyro.cli(main)
