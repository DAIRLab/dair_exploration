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

from dair_exploration.gui_util import MJXMeshcatVisualizer
from dair_exploration.file_util import get_config
from dair_exploration import mjx_util


def test_gui():
    """GUI Test"""
    mj_model = mujoco.MjModel.from_xml_path(get_config("default.mjcf"))
    mjx_model = mjx.put_model(mj_model)
    dt = float(mj_model.opt.timestep)
    nstep = int(2.0 / dt)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_step(mj_model, mj_data)
    mjx_data = mjx.put_data(mj_model, mj_data)
    ctrl = jnp.zeros((nstep, len(mjx_data.ctrl)))
    ctrl_cpu = np.zeros((nstep, len(mjx_data.ctrl)))
    print("Running initial sim compilation...", end="", flush=True)
    start = time.time()
    compiled_diffsim = (
        jax.jit(mjx_util.diffsim)
        .trace(
            mjx_model,
            mjx_data,
            # Use empty dtype/shape struct to avoid caching
            jax.ShapeDtypeStruct((nstep, len(mjx_data.ctrl)), jnp.dtype("float32")),
        )
        .lower()
        .compile()
    )
    print(f"done in {time.time()-start}s")
    print("Starting Meshcat...")
    vis = MJXMeshcatVisualizer(mjx_model, mjx_data)
    xposes = np.zeros((nstep,) + mj_data.xpos.shape)
    xmats = np.zeros((nstep,) + mj_data.xmat.shape)
    print("Simulating 2s on CPU...", end="", flush=True)
    start = time.time()
    for idx in range(nstep):
        mj_data.ctrl = ctrl_cpu[idx]
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

    print("Simulating 2s...", end="", flush=True)
    start = time.time()
    data_stacked = compiled_diffsim(mjx_model, mjx_data, ctrl)
    print(f"done in {time.time()-start}s")
    print("Unstacking data...", end="", flush=True)
    start = time.time()
    data_list = mjx_util.data_unstack(data_stacked)
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
