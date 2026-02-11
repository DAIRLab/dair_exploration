#!/usr/bin/env python3

"""
Test EIG calculation
"""
import time

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from dair_exploration.file_util import get_config, enable_jax_cache
from dair_exploration import mjx_util
from dair_exploration import exploration


def test_exploration():
    """Exploration Algorithm Test"""
    # Tests can be arbitrarily long
    # pylint: disable=too-many-locals
    enable_jax_cache()
    mj_model = mujoco.MjModel.from_xml_path(get_config("default.mjcf"))
    mjx_model = mjx.put_model(mj_model)
    dt = float(mj_model.opt.timestep)
    nstep = int(2.0 / dt)
    print("Sim compilation...", end="", flush=True)
    start = time.time()
    mjx_data = mjx_util.jit_step(mjx_model, mjx.make_data(mjx_model))
    print(f"done in {time.time()-start}s")
    mjx_data2 = mjx_util.jit_step(mjx_model, mjx_data)
    ctrl1 = jnp.zeros((100, nstep, len(mjx_data.ctrl)))
    ctrl2 = jax.random.uniform(jax.random.key(0), ctrl1.shape)

    params = mjx_util.populate_parameter_dict(mjx_model, {"object-geom": ["size"]})
    traj_qpos_params = {
        "object-geom": mjx_data.qpos[
            mjx_util.qposidx_from_geom_name(mjx_model, "object-geom")
        ]
    }
    traj_qpos_params2 = {
        "object-geom": mjx_data2.qpos[
            mjx_util.qposidx_from_geom_name(mjx_model, "object-geom")
        ]
    }
    contact_ids = mjx_util.contactids_from_collision_geoms(
        mjx_model, mjx_data, ["spherebot1-geom", "spherebot2-geom"], ["object-geom"]
    )

    jit_info = jax.jit(
        jax.vmap(exploration.expected_info, in_axes=(0, None, None, None, None, None))
    )
    print("Expected Info w/ ctrl1 (zeros)...", end="", flush=True)
    start = time.time()
    jit_info(ctrl1, params, traj_qpos_params, mjx_data, mjx_model, contact_ids)
    print(f"done in {time.time()-start}s")
    print("Expected Info w/ ctrl2 (random)...", end="", flush=True)
    start = time.time()
    jit_info(ctrl2, params, traj_qpos_params2, mjx_data2, mjx_model, contact_ids)
    print(f"done in {time.time()-start}s")


if __name__ == "__main__":
    test_exploration()
