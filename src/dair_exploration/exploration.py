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
    outputs = jit_outputs(
        params, data_stacked.qpos, data_stacked, base_model, contact_ids
    )

    # Get Jacobian w.r.t. outputs
    output_jacs = jit_jac_outputs(
        params, data_stacked.qpos, data_stacked, base_model, contact_ids
    )
    qpos_keys = np.concatenate(
        [
            mjx_util.qposidx_from_geom_name(base_model, key)
            for key in traj_qpos_params.keys()
        ]
    )

    ## Handle Phi
    phi_jacs = output_jacs["phi"]
    # Handle w.r.t. state, assume jac == identity, note phi_jacs[1] is block diagonal on timesteps
    phi_qpos_jac = jnp.sum(phi_jacs[1][..., qpos_keys], axis=-2)
    # Handle w.r.t. geometry / physics params
    phi_geom_jac = jnp.concat(jax.tree.flatten(phi_jacs[0])[0], axis=-1)

    # Create multiplier
    # TODO: make hyperparameters
    phi_nominal = 0.005  # m
    phi_ci = 0.05  # Confidence Interval
    phi_alpha = np.log(np.reciprocal(phi_ci) - 1.0) / phi_nominal
    contact_bool = jax.nn.sigmoid(
        -phi_alpha * outputs["phi"]
    )  # simgoid = 1/(1+exp(-x))
    phi_mult = contact_bool - jnp.square(
        contact_bool
    )  # exp(x)/(1+exp(x))^2 = sigmoid(-x) - sigmoid(-x)^2

    # Create phi info
    phi_param_jac = jnp.concat([phi_qpos_jac, phi_geom_jac], axis=-1)
    phi_param_flat_jac = phi_param_jac.reshape(-1, phi_param_jac.shape[-1])
    phi_info = phi_param_flat_jac.T @ (phi_mult.reshape(-1, 1) * phi_param_flat_jac)

    ## Handle Normal
    normal_jacs = output_jacs["normal"]
    # Handle w.r.t. state, assume jac == identity, note phi_jacs[1] is block diagonal on timesteps
    normal_qpos_jac = jnp.sum(normal_jacs[1][..., qpos_keys], axis=-2)
    # Handle w.r.t. geometry / physics params
    normal_geom_jac = jnp.concat(jax.tree.flatten(normal_jacs[0])[0], axis=-1)

    # Create multiplier
    # TODO: make hyperparameters
    normal_var = 0.01519224261
    normal_mult = contact_bool * jnp.reciprocal(normal_var)

    # Create normal info
    normal_param_jac = jnp.concat([normal_qpos_jac, normal_geom_jac], axis=-1)
    normal_param_flat_jac = normal_param_jac.reshape(-1, normal_param_jac.shape[-1])
    normal_info = normal_param_flat_jac.T @ (
        jnp.repeat(normal_mult[..., None], 3, axis=-1).reshape(-1, 1)
        * normal_param_flat_jac
    )

    return phi_info + normal_info
