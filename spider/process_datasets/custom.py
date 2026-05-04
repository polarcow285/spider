"""
Convert the custom mocap demo data to mjwp format.

Input format: h5 file
Keys: ['mano_joint_coords', 'wrist_pos', 'wrist_quat', 'wrist_rot_mat', 'obj_pos', 'obj_quat']
    mano_joint_coords: shape=(271, 21, 3), dtype=float32
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
    'fps': 120.0,                  # Frame rate
    'task_name': 'screwdriver',

"""
import io
import json
import os
import h5py
import numpy as np
import tyro
import loguru
from spider.io import get_processed_data_dir

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
    file_path = f"{dataset_dir}/raw/custom/zed_mocap_demo_0318_screwdriver_1.5x.h5"
    output_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name="custom",
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=0,
    )
    os.makedirs(output_dir, exist_ok=True)

    # task info
    task_info = {
        "task": task,
        "dataset_name": "custom",
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": 0,
        "right_object_mesh_dir": None,
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

    tip_ids = [4, 8, 12, 16, 20]
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

    # ---- Contact placeholders ----
    contact = np.zeros((N,), dtype=np.float32)
    contact_pos = np.zeros((N, 3), dtype=np.float32)

    np.savez(
        f"{output_dir}/trajectory_keypoints.npz",
        qpos_wrist_right=qpos_wrist_right[start_idx:],
        qpos_finger_right=qpos_finger_right[start_idx:],
        qpos_obj_right=qpos_obj_right[start_idx:],
        qpos_wrist_left=qpos_wrist_left[start_idx:],
        qpos_finger_left=qpos_finger_left[start_idx:],
        qpos_obj_left=qpos_obj_left[start_idx:],
    )
    loguru.logger.info(f"Saved qpos to {output_dir}/trajectory_keypoints.npz")

if __name__ == "__main__":
    tyro.cli(main)
