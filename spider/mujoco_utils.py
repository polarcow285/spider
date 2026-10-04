# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utils for mujoco.

Author: Chaoyi Pan
Date: 2025-11-01
"""

from contextlib import contextmanager

import mujoco
import mujoco.viewer
import numpy as np


def get_viewer(show_viewer: bool, model: mujoco.MjModel, data: mujoco.MjData):
    if show_viewer:
        run_viewer = lambda: mujoco.viewer.launch_passive(model, data)
    else:
        cam = mujoco.MjvCamera()
        cam.type = 2
        cam.fixedcamid = 0

        @contextmanager
        def run_viewer():
            yield type(
                "DummyViewer",
                (),
                {"is_running": lambda: True, "sync": lambda: None, "cam": 0},
            )

    return run_viewer


def compute_table_z_offset(
    model_path: str,
    obj_pose: np.ndarray,
    obj_body: str = "bottom",
    table_geom: str = "table_geom",
) -> float:
    """Z shift that rests the object (at obj_pose = [pos, quat_wxyz]) on top of the scene's table."""
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, table_geom)
    if table_id == -1:
        raise ValueError(f"No '{table_geom}' geom in {model_path}; pass z_offset explicitly")
    root = model.body(obj_body).id
    qadr = model.jnt_qposadr[model.body_jntadr[root]]
    data.qpos[qadr : qadr + 7] = obj_pose
    mujoco.mj_kinematics(model, data)
    table_top = data.geom_xpos[table_id][2] + model.geom_size[table_id][2]

    # geoms in the object's subtree; prefer collision geoms since those touch the table
    geoms = []
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        while b not in (0, root):
            b = model.body_parentid[b]
        if b == root:
            geoms.append(g)
    collision = [g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
    lowest = np.inf
    for g in collision or geoms:
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            pts = model.mesh_vert[adr : adr + num]
        else:  # corners of the geom's local bounding box
            center, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
            signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
            pts = center + signs * half
        world = data.geom_xpos[g] + pts @ data.geom_xmat[g].reshape(3, 3).T
        lowest = min(lowest, world[:, 2].min())
    return float(table_top - lowest)
