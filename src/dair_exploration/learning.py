#!/usr/bin/env python3

"""
Learning Functions
"""

import jax
import jax.numpy as jnp
from mujoco import mjx


def loss_vimp(
    params: dict[str, dict[str, jax.Array]],
    base_model: mjx.Model,
) -> jax.Array:
    pass
