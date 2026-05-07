"""
Visualizes the following input custom demo data using matplotlib
Input format: h5 file
Keys: ['mano_joint_coords', 'wrist_pos', 'wrist_quat', 'wrist_rot_mat', 'obj_pos', 'obj_quat']
      : shape=(271, 21, 3), dtype=float32
    wrist_pos: shape=(271, 3), dtype=float32
    wrist_quat: shape=(271, 4), dtype=float64
    wrist_rot_mat: shape=(271, 3, 3), dtype=float32
    obj_pos: shape=(271, 3), dtype=float64
    obj_quat: shape=(271, 4), dtype=float64d
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Load H5 data
# -----------------------------
file_path = "/home/nl455/spider/example_datasets/raw/custom/zed_mocap_demo_0318_screwdriver_1.5x.h5"

with h5py.File(file_path, "r") as f:
    mano = f["mano_joint_coords"][:]   # (T, 21, 3)
    wrist = f["wrist_pos"][:]          # (T, 3)
    obj = f["obj_pos"][:]              # (T, 3)

T = mano.shape[0]

EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

# -----------------------------
# GLOBAL BOUNDS (KEY FIX)
# -----------------------------
all_points = np.concatenate([
    mano.reshape(-1, 3),
    wrist,
    obj
], axis=0)

mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)

center = (mins + maxs) / 2
max_range = (maxs - mins).max() / 2

# -----------------------------
# FIGURE
# -----------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

def set_fixed_axes():
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.set_box_aspect([1, 1, 1])  # keeps aspect ratio fixed

set_fixed_axes()

TIP_IDS = [16, 17, 18, 19, 20]

# -----------------------------
# UPDATE
# -----------------------------
def update(frame):
    ax.cla()  # clear plot only, NOT autoscale logic

    hand = mano[frame]
    w = wrist[frame]
    o = obj[frame]

    tip_points = hand[TIP_IDS]
    other_ids = [i for i in range(21) if i not in TIP_IDS]
    other_points = hand[other_ids]

    ax.scatter(
        other_points[:, 0],
        other_points[:, 1],
        other_points[:, 2],
        c="blue",
        s=20,
        label="joints"
    )
    ax.scatter(
        tip_points[:, 0],
        tip_points[:, 1],
        tip_points[:, 2],
        c="red",
        s=40,
        label="fingertips"
    )

    ax.scatter(w[0], w[1], w[2], c="black", s=60, label="wrist")

    ax.scatter(o[0], o[1], o[2], c="green", s=60)

    # for i, j in EDGES:
    #     ax.plot(
    #         [hand[i,0], hand[j,0]],
    #         [hand[i,1], hand[j,1]],
    #         [hand[i,2], hand[j,2]],
    #         c="black",
    #         linewidth=1
    #     )

    # IMPORTANT: re-apply fixed axes every frame
    set_fixed_axes()

    ax.set_title(f"Frame {frame}")

anim = FuncAnimation(fig, update, frames=T, interval=30)
plt.show()
