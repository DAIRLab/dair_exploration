#!/usr/bin/env python3

"""
Learning Functions
"""

from functools import partial

import jax
import jax.numpy as jnp
from mujoco import mjx
import numpy as np

from dair_exploration import mjx_util


## Function to compute outputs from parameters
def get_outputs(
    params: dict[str, dict[str, jax.Array]],
    qpos: jax.Array,
    data: mjx.Data,
    base_model: mjx.Model,
    contact_ids: np.ndarray,
) -> dict[str, jax.Array]:
    """Compute outputs (phi and normals) from data"""
    # write pose and params to model/data
    param_data = data.replace(qpos=qpos)
    param_model = mjx_util.write_params_to_model(base_model, params)

    forward_data = mjx_util.jit_forward(param_model, param_data)

    return {
        "phi": forward_data._impl.contact.dist[..., contact_ids],
        "normal": forward_data._impl.contact.frame[..., contact_ids, 0, :],
    }


def jit_get_outputs():
    """JIT of the above"""
    return jax.jit(jax.vmap(get_outputs, in_axes=(None, 0, 0, None, None)))


def expected_info(
    ctrl: jax.Array,
    params: dict[str, dict[str, jax.Array]],
    traj_qpos_params: dict[str, jax.Array],
    base_data: mjx.Data,
    base_model: mjx.Model,
    contact_ids: np.ndarray,
) -> jax.Array:
    """Calculate expected info

    Args:
        ctrl: (traj_len, n_control)
        params: geometry params
        traj_qpos_params: learned current position params
        base_model: model from initial mcjf/urdf
        base_data: data that contains initial qpos/qvel of all objects
    Returns:
        n_params x n_params expected info
    """

    jit_outputs = jit_get_outputs()
    jit_jac_outputs = jax.jit(jax.jacrev(jit_outputs, argnums=(0, 1)))

    # Get current position
    data_current = mjx_util.write_qpos_to_data(base_model, base_data, traj_qpos_params)

    # Sim Forward
    data_stacked = mjx_util.diffsim(base_model, data_current, ctrl)

    # Get Outputs (unnecessary)
    # outputs = jit_outputs(params, data_stacked.qpos, data_stacked, base_model, contact_ids)

    # Get Jacobian w.r.t. outputs
    output_jacs = jit_jac_outputs(
        params, data_stacked.qpos, data_stacked, base_model, contact_ids
    )

    return output_jacs
