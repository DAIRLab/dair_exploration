#!/usr/bin/env python3

"""
Test basic GUI visualization tools
"""

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
import mujoco
from mujoco import mjx

from dair_exploration.gui_utils import MJXMeshcatVisualizer
from dair_exploration.file_utils import get_config


def test_gui():
    """GUI Test"""
    mj_model = mujoco.MjModel.from_xml_path(get_config("default.mjcf"))
    mjx_model = mjx.put_model(mj_model)
    print("Running initial sim compilation...")
    step_jit = jax.jit(mjx.step)
    mjx_data = step_jit(mjx_model, mjx.make_data(mjx_model))
    data_list = [mjx_data]
    print("Starting Meshcat...")
    vis = MJXMeshcatVisualizer(mjx_model, mjx_data)
    print("Simulating 100 steps...")
    for _ in range(100):
        data_list.append(step_jit(mjx_model, data_list[-1]))
    print("Writing to Meshcat...")
    traj = {
        "true-geom": [
            (jnp.array([0.0, 0.0, idx * 0.01]), Rotation.from_euler("z", 0.01 * idx))
            for idx in range(len(data_list))
        ]
    }
    vis.update_visuals(mjx_model, data_list, traj)
    input("Done! Press enter to exit...")
    del vis


if __name__ == "__main__":
    test_gui()
