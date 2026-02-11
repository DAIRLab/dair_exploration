#!/usr/bin/env python3

"""
Test EIG calculation
"""
import time

import jax
import jax.numpy as jnp
from mujoco import mjx

from dair_exploration.file_util import enable_jax_cache
from dair_exploration import mjx_util
from dair_exploration import exploration
from dair_exploration.learning import LearnedModel


def test_exploration():
    """Exploration Algorithm Test"""
    # Tests can be arbitrarily long
    # pylint: disable=too-many-locals
    enable_jax_cache()
    learned_model = LearnedModel("default.mjcf", {"object-geom": ["size"]})
    mjx_model = learned_model.active_model
    dt = float(mjx_model.opt.timestep)
    nstep = int(2.0 / dt)
    print("Sim compilation...", end="", flush=True)
    start = time.time()
    mjx_data = mjx_util.jit_step(mjx_model, mjx.make_data(mjx_model))
    print(f"done in {time.time()-start}s")
    mjx_data2 = mjx_util.jit_step(mjx_model, mjx_data)
    ctrl1 = jnp.zeros((100, nstep, len(mjx_data.ctrl)))
    ctrl2 = jax.random.uniform(jax.random.key(0), ctrl1.shape)

    params = learned_model.params
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

    # Redundantly write to model
    # Should NOT cause a re-trace! Test w/ breakpoint()
    mjx_model = LearnedModel.write_params_to_model(params, mjx_model)

    # Modify nparray in model
    # SHOULD cause a re-trace! Test w/ breakpoint()
    # import numpy as np
    # test = np.copy(mjx_model.mesh_vert)
    # test[0, 0] = 5.0
    # mjx_model = mjx_model.replace(mesh_vert=test)

    print("Expected Info w/ ctrl2 (random)...", end="", flush=True)
    start = time.time()
    jit_info(ctrl2, params, traj_qpos_params2, mjx_data2, mjx_model, contact_ids)
    print(f"done in {time.time()-start}s")


if __name__ == "__main__":
    test_exploration()
