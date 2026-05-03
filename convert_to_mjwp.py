"""
Convert the custom mocap demo data to mjwp format.

Input format: h5 file
Keys: ['mano_keypoints_3d', 'wrist_pos', 'wrist_quat', 'wrist_rot_mat', 'obj_pos', 'obj_quat']
    mano_keypoints_3d: shape=(271, 21, 3), dtype=float32
    wrist_pos: shape=(271, 3), dtype=float32
    wrist_quat: shape=(271, 4), dtype=float64
    wrist_rot_mat: shape=(271, 3, 3), dtype=float32
    obj_pos: shape=(271, 3), dtype=float64
    obj_quat: shape=(271, 4), dtype=float64d

Output format: npz file 'trajectory_keypoint.npz'
Keys: ['qpos_wrist_right', 'qpos_finger_right', 'qpos_wrist_left', 'qpos_finger_left', 'qpos_obj_right', 'qpos_obj_left', 'contact', 'contact_pos']
    qpos_wrist_right: shape=(T, 7), dtype=float32
    qpos_finger_right: shape=(T, 21, 3), dtype=float32
    qpos_obj_right: shape=(T, 7), dtype=float32
    'fps': 120.0,                  # Frame rate
    'task_name': 'screwdriver',

"""
import io
import json
import os
import h5py
import numpy as np
import argparse

def convert_h5_to_mjwp(input_path, output_path):
    with h5py.File(input_path, "r") as f:
        mano_keypoints = f["mano_keypoints_3d"][:]   # (T, 21, 3)
        wrist_pos = f["wrist_pos"][:]                # (T, 3)
        wrist_quat = f["wrist_quat"][:]              # (T, 4)
        obj_pos = f["obj_pos"][:]                    # (T, 3)
        obj_quat = f["obj_quat"][:]                  # (T, 4)

    T = mano_keypoints.shape[0]

    # ---- Right hand ----
    qpos_wrist_right = np.concatenate([wrist_pos, wrist_quat], axis=1).astype(np.float32)  # (T, 7)
    qpos_finger_right = mano_keypoints.astype(np.float32)  # (T, 21, 3)

    # ---- Object (right) ----
    qpos_obj_right = np.concatenate([obj_pos, obj_quat], axis=1).astype(np.float32)  # (T, 7)

    # ---- Left hand/object (not provided → zeros) ----
    qpos_wrist_left = np.zeros((T, 7), dtype=np.float32)
    qpos_finger_left = np.zeros((T, 21, 3), dtype=np.float32)
    qpos_obj_left = np.zeros((T, 7), dtype=np.float32)

    # ---- Contact placeholders ----
    contact = np.zeros((T,), dtype=np.float32)
    contact_pos = np.zeros((T, 3), dtype=np.float32)

    # ---- Save ----
    np.savez(
        output_path,
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_wrist_left=qpos_wrist_left,
        qpos_finger_left=qpos_finger_left,
        qpos_obj_right=qpos_obj_right,
        qpos_obj_left=qpos_obj_left,
        contact=contact,
        contact_pos=contact_pos,
        fps=120.0,
        task_name="screwdriver",
    )

    print(f"Saved MJWP trajectory to: {output_path}")
    print(f"T = {T}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert mocap H5 to MJWP NPZ format")
    parser.add_argument("input", type=str, help="Path to input .h5 file")
    parser.add_argument(
        "--output",
        type=str,
        default="trajectory_keypoint.npz",
        help="Output .npz file",
    )

    args = parser.parse_args()
    convert_h5_to_mjwp(args.input, args.output)

# usage: python convert_to_mjwp.py /home/nl455/spider/example_datasets/raw/custom/zed_mocap_demo_0318_screwdriver_1.5x.h5 output.npz /home/nl455/spider/example_datasets/processed/custom/trajectory_keypoint.npz
