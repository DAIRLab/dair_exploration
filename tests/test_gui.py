#!/usr/bin/env python3

"""
Test basic GUI visualization tools
"""
import time

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
import mujoco
from mujoco import mjx
import numpy as np

from dair_exploration.gui_utils import MJXMeshcatVisualizer
from dair_exploration.file_utils import get_config


def test_gui():
    """GUI Test"""
    mj_model = mujoco.MjModel.from_xml_path(get_config("default.mjcf"))
    mjx_model = mjx.put_model(mj_model)
    dt = float(mj_model.opt.timestep)
    nstep = int(2.0 / dt)
    mj_data = mujoco.MjData(mj_model)
    print("Running initial sim compilation...", end="", flush=True)
    step_jit = jax.jit(mjx.step)
    start = time.time()
    mjx_data = step_jit(mjx_model, mjx.make_data(mjx_model))
    print(f"done in {time.time()-start}s")
    print("Starting Meshcat...")
    vis = MJXMeshcatVisualizer(mjx_model, mjx_data)

    xposes = np.zeros((nstep,) + mj_data.xpos.shape)
    xmats = np.zeros((nstep,) + mj_data.xmat.shape)
    print("Simulating 2s on CPU...", end="", flush=True)
    start = time.time()
    for idx in range(nstep):
        mujoco.mj_step(mj_model, mj_data)
        xposes[idx] = mj_data.xpos
        xmats[idx] = mj_data.xmat
    print(f"done in {time.time()-start}s")
    data_list_cpu = [
        mjx_data.replace(
            xpos=xposes[0].reshape(mjx_data.xpos.shape),
            xmat=xmats[0].reshape(mjx_data.xmat.shape),
        )
    ]
    for idx in range(1, nstep):
        data_list_cpu.append(
            data_list_cpu[-1].replace(
                xpos=xposes[idx].reshape(mjx_data.xpos.shape),
                xmat=xmats[idx].reshape(mjx_data.xmat.shape),
            )
        )
    print("Writing to Meshcat...")
    vis.update_visuals(mjx_model, data_list_cpu, None)
    input("Done! Press enter to test JAX sim...")

    data_list = [mjx_data]
    print("Simulating 2s...", end="", flush=True)
    start = time.time()
    for _ in range(nstep):
        data_list.append(step_jit(mjx_model, data_list[-1]))
    print(f"done in {time.time()-start}s")

    print("Writing to Meshcat...")
    true_traj = {
        "true-geom": [
            (jnp.array([0.0, 0.0, idx * 0.01]), Rotation.from_euler("z", 0.01 * idx))
            for idx in range(len(data_list))
        ]
    }
    vis.update_visuals(mjx_model, data_list, true_traj)

    input("Press enter to visualize...")
    vis.sweep(dt=dt)
    input("Done! Press enter to exit...")
    del vis


if __name__ == "__main__":
    test_gui()
