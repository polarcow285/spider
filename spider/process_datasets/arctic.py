"""
Convert the arctic mocap demo data to mjwp format.

Input format: h5 file
Keys: ['mano_faces', 'mano_joint_coord','mano_verts','meta','qpos']
    qpos: shape=(161, 21, 3), dtype=float32
        where
        QPOS_INDEX = {
        "object_pos": np.arange(0, 3),
        "object_quat": np.arange(3, 7),
        "object_joint": np.arange(7, 8),
        "hand_wrist_pos": np.arange(8, 11),
        "hand_wrist_rot": np.arange(11, 14),
        "hand_index": np.arange(14, 18),
        "hand_middle": np.arange(18, 22),
        "hand_ring": np.arange(22, 26),
        "hand_thumb": np.arange(26, 30),

        # convenience
        "object_pose": np.arange(0, 7),
        "hand": np.arange(8, 30),
        "fingers": np.arange(14, 30),  # hand excluding wrist joints

    }
    'mano_joint_coord': shape=(161, 21, 3), dtype=float32


Output format: npz file 'trajectory_keypoint.npz'
Keys: ['qpos_wrist_right', 'qpos_finger_right', 'qpos_wrist_left', 'qpos_finger_left', 'qpos_obj_right', 'qpos_obj_left', 'contact', 'contact_pos']
    qpos_wrist_right: shape=(T, 7), dtype=float32
    qpos_finger_right: shape=(T, 5, 7), dtype=float32 (5 fingertips)
    qpos_obj_right: shape=(T, 7), dtype=float32
    qpos_obj_arti: shape (T, 1), dtype=float32

Example usage: python spider/process_datasets/arctic.py --task scissors --embodiment-type right --dataset-dir example_datasets

"""
import io
import json
import os
import h5py
import numpy as np
import tyro
import loguru
import spider
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
from spider.io import get_processed_data_dir
from scipy.spatial.transform import Rotation as R

def main(
    dataset_dir: str = "../../example_datasets",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    show_viewer: bool = True,
    save_video: bool = False,
    start_idx: int = 0,
):
    dataset_dir = os.path.abspath(dataset_dir)
    # file_path = f"{dataset_dir}/raw/custom/{task}_{embodiment_type}.h5"
    file_path = f"{dataset_dir}/raw/arctic/demo.h5"
    output_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name="arctic",
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=0,
    )
    os.makedirs(output_dir, exist_ok=True)


    # read data
    with h5py.File(file_path, "r") as f:
        # Standard MANO order: [Wrist, Thumb(1-4), Index(5-8), Middle(9-12), Ring(13-16), Pinky(17-20)]
        qpos = f["qpos"][:]   # (T, 30)
        mano_keypoints = f["mano_joint_coord"][:]  # (T, 21, 3)
    wrist_pos = qpos[:, 8:11]  # (T, 3)
    # scipy converts to xyzw by default
    wrist_quat = R.from_euler('XYZ', qpos[:, 11:14]).as_quat()  # (T, 4)
    wrist_quat = wrist_quat[:, [3, 0, 1, 2]]  # convert to wxyz
    obj_pos = qpos[:, 0:3]  # (T, 3)
    obj_quat = qpos[:, 3:7]  # (T, 4)
    obj_arti = qpos[:, 7:8]  # (T, 1)

    N = qpos.shape[0]

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

    np.savez(
        f"{output_dir}/trajectory_keypoints.npz",
        qpos_wrist_right=qpos_wrist_right[start_idx:],
        qpos_finger_right=qpos_finger_right[start_idx:],
        qpos_obj_right=qpos_obj_right[start_idx:],
        qpos_wrist_left=qpos_wrist_left[start_idx:],
        qpos_finger_left=qpos_finger_left[start_idx:],
        qpos_obj_left=qpos_obj_left[start_idx:],
        obj_arti=obj_arti[start_idx:],
    )
    loguru.logger.info(f"Saved qpos to {output_dir}/trajectory_keypoints.npz")

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
        top_mesh_path = os.path.join(
            dataset_dir,
            "processed",
            "arctic",
            "assets",
            "objects",
            "scissors",
            "top.obj",
        )

        bottom_mesh_path = os.path.join(
            dataset_dir,
            "processed",
            "arctic",
            "assets",
            "objects",
            "scissors",
            "bottom.obj",
        )

        mj_spec.add_mesh(
            name="scissors_top",
            file=top_mesh_path,
            scale=[0.002, 0.002, 0.002],
        )

        mj_spec.add_mesh(
            name="scissors_bottom",
            file=bottom_mesh_path,
            scale=[0.002, 0.002, 0.002],
        )

        scissors_root = object_right_handle.add_body(
            name="scissors_root",
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
        )

        # --------------------------------------------------
        # Bottom half (fixed relative to root)
        # --------------------------------------------------
        scissors_bottom = scissors_root.add_body(
            name="scissors_bottom",
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
        )

        scissors_bottom.add_geom(
            name="scissors_bottom_geom",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="scissors_bottom",
            rgba=[0.7, 0.7, 0.7, 1],
            condim=1,
        )

        # --------------------------------------------------
        # Top half (hinged)
        # --------------------------------------------------
        scissors_top = scissors_bottom.add_body(
            name="scissors_top",
            pos=[0, 0, 0],   # hinge origin
            quat=[1, 0, 0, 0],
        )

        scissors_top.add_joint(
            name="scissors_hinge",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 0, -1],   # adjust axis
            pos=[0, 0, 0],    # hinge pivot
            range=[0, 0.5],
        )

        scissors_top.add_geom(
            name="scissors_top_geom",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="scissors_top",
            rgba=[0.3, 0.3, 0.3, 1],
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
            hinge_qpos_addr = mj_model.joint("scissors_hinge").qposadr[0]
            mj_data.mocap_pos[:] = qpos_list[cnt, :, :3]
            mj_data.mocap_quat[:] = qpos_list[cnt, :, 3:]
            # scissors articulation
            mj_data.qpos[hinge_qpos_addr] = obj_arti[cnt, 0]
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
