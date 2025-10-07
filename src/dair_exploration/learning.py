#!/usr/bin/env python3

"""
Learning Functions
"""

import jax
import jax.numpy as jnp
from mujoco import mjx

@jax.vmap(in_axes)
def _loss_vimp_


def loss_vimp(
    params: dict[str, dict[str, jax.Array]],
    traj_params: dict[str, tuple[jax.Array, jax.Array]],
    data: dict[str, jax.Array],
    base_model: mjx.Model,
) -> jax.Array:
	# TODO: write params to base_model
	# TODO: combine traj_params / data into trajectory
	# TODO: 
    pass
