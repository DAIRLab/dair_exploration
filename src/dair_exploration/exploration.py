#!/usr/bin/env python3

"""
Exploration w/ EIG Functions
"""

from dataclasses import dataclass
from enum import Enum
import operator

import gin
import jax
import jax.numpy as jnp
from mujoco import mjx
import numpy as np

from dair_exploration import mjx_util
from dair_exploration.learning import LearnedModel


@gin.constants_from_enum
class InfoStyle(Enum):
    """Which style of info calculation to use"""

    IDENTITY = 0  # treat dx[t]/dx[t+1] = identity
    DIFFSIM = 1  # use diffsim, get info w.r.t. first timestep
    SAMPLING = 2  # use sampling, get info w.r.t. last timestep


@gin.configurable
@dataclass(frozen=True)
class InfoHyperparameters:  # pylint: disable=too-many-instance-attributes
    """Class to specify info hyperparameters"""

    # Loss Weights
    phi_nominal: float = 0.002  # m, distance where p(contact_measured) drops below CI
    phi_ci: float = 0.05  # Confidence Interval (0,1) for above
    normal_var: float = (
        0.01519224261  # cos(radians) [default 10 degrees], variance of cos(normal angle deviation)
    )

    # Computation Parameters
    epsilon: float = 1e-8
    style: InfoStyle = InfoStyle.IDENTITY


## Function to compute outputs from measurements
def get_outputs_from_measurements(
    params: tuple[dict[str, dict[str, jax.Array]], dict[str, dict[str, jax.Array]]],
    measurements: dict[str, dict[str, jax.Array]],
    base_model: mjx.Model,
) -> dict[str, jax.Array]:
    """Compute outputs (phi and normals) from data"""
    # write pose and params to model/data
    param_model = LearnedModel.write_params_to_model(params[0], base_model)
    pose_traj = {
        geom_name: measurements[geom_name]["position"]
        for geom_name in measurements.keys()
        if isinstance(measurements[geom_name], dict)
        and "position" in measurements[geom_name]
    } | {geom_name: params[1][geom_name]["position"] for geom_name in params[1].keys()}
    vel_traj = {
        geom_name: measurements[geom_name]["velocity"]
        for geom_name in measurements.keys()
        if isinstance(measurements[geom_name], dict)
        and "velocity" in measurements[geom_name]
    } | {geom_name: params[1][geom_name]["velocity"] for geom_name in params[1].keys()}
    # Forward will compute all physical parameters and distances
    forward_data = mjx_util.jit_forward(
        param_model,
        mjx_util.write_qvel_to_data(
            param_model,
            mjx_util.write_qpos_to_data(
                param_model, mjx.make_data(param_model), pose_traj
            ),
            vel_traj,
        ),
    )

    obj_geom_names = params[1].keys()
    assert len(obj_geom_names) == 1  # Only 1 object supported
    contact_masks = {
        geom_name: mjx_util.contactids_from_collision_geoms(
            param_model, [geom_name], obj_geom_names
        )
        for geom_name in measurements.keys()
        if isinstance(measurements[geom_name], dict)
        and "contact_normal_W" in measurements[geom_name]
    }
    phis = {
        geom_name: jnp.sum(
            forward_data.contact.dist * jnp.abs(contact_mask[jnp.newaxis, :]),
            axis=-1,
            keepdims=True,
        )
        for geom_name, contact_mask in contact_masks.items()
    }
    normals = {
        geom_name: jnp.mean(
            jnp.sum(
                contact_mask[jnp.newaxis, :, jnp.newaxis]
                * forward_data.contact.frame[..., 0, :],
                axis=-2,
                keepdims=True,
            ),
            axis=-2,
        )
        for geom_name, contact_mask in contact_masks.items()
    }

    return {
        "phi": phis,
        "normal": normals,
    }


vmap_get_outputs_from_measurements = jax.jit(
    jax.vmap(get_outputs_from_measurements, in_axes=((None, 0), 0, None))
)

jac_get_outputs_from_measurements = jax.jit(
    jax.jacrev(vmap_get_outputs_from_measurements)
)


def observed_info(
    params: tuple[dict[str, dict[str, jax.Array]], dict[str, jax.Array]],
    measurements: dict[str, dict[str, jax.Array]],
    base_model: mjx.Model,
    hyperparams: InfoHyperparameters,
) -> jax.Array:
    """Calculate observed info

    Args:
        ctrl: (traj_len, n_control)
        params: geometry params
        measurements: contact and robot proprioception and control data,
                        as a full trajectory (not a list of trajectories)
        base_model: model from initial mcjf/urdf
    Returns:
        n_params x n_params observed info
    """
    # TODO: add other styles
    assert hyperparams.style == InfoStyle.IDENTITY

    n_geom = jax.tree.reduce(operator.add, jax.tree.map(jnp.size, params[0]))
    n_q = jax.tree.reduce(
        operator.add,
        jax.tree.map(
            lambda leaf: leaf.shape[-1],
            [params[1][name]["position"] for name in params[1].keys()],
        ),
    )
    assert len(params[1].keys()) == 1  # Only 1 object supported
    obj_geom_name = list(params[1].keys())[0]

    # Get phi/normal outputs and parameter jacobians
    outputs = vmap_get_outputs_from_measurements(params, measurements, base_model)
    jacs = jac_get_outputs_from_measurements(params, measurements, base_model)

    ## Compute Phi Info
    phis = jax.tree.reduce(
        lambda leaf1, leaf2: jnp.concat([leaf1, leaf2], axis=0), outputs["phi"]
    )  # (n_T*n_contacts, 1, 1)
    phi_alpha = (
        np.log(np.reciprocal(hyperparams.phi_ci) - 1.0) / hyperparams.phi_nominal
    )
    contact_bool = jax.nn.sigmoid(-phi_alpha * phis)  # simgoid = 1/(1+exp(-x))
    phi_mult = contact_bool - jnp.square(
        contact_bool
    )  # exp(x)/(1+exp(x))^2 = sigmoid(-x) - sigmoid(-x)^2
    # Handle w.r.t. geometry / physics params
    phi_geom_jac = jnp.stack(
        jax.tree.flatten([jacs["phi"][name][0] for name in jacs["phi"].keys()])[0],
        axis=0,
    ).reshape(
        -1, 1, n_geom
    )  # (n_T*n_contacts, 1, n_geom)
    # Handle w.r.t. state, assume jac == identity, note that (... n_T, ..., n_T, :) is block-diagonal
    phi_pose_jac = jnp.sum(
        jnp.stack(
            jax.tree.flatten(
                [
                    jacs["phi"][name][1][obj_geom_name]["position"]
                    for name in jacs["phi"].keys()
                ]
            )[0],
            axis=0,
        ),
        axis=-2,
    ).reshape(
        -1, 1, n_q
    )  # (n_T*n_contacts, 1, n_geom)
    phi_param_jac = jnp.concat(
        [phi_pose_jac, phi_geom_jac], axis=-1
    )  # (n_T*n_contacts, 1, n_param = n_q + n_geom)
    phi_info = jnp.swapaxes(phi_param_jac, -2, -1) @ phi_mult @ phi_param_jac

    ## Handle Normal
    # Handle w.r.t. state, assume jac == identity, note phi_jacs[1] is block diagonal on timesteps
    normal_pose_jac = jnp.sum(
        jnp.stack(
            jax.tree.flatten(
                [
                    jacs["normal"][name][1][obj_geom_name]["position"]
                    for name in jacs["normal"].keys()
                ]
            )[0],
            axis=0,
        ),
        axis=-2,
    ).reshape(-1, 3, n_q)
    # Handle w.r.t. geometry / physics params
    normal_geom_jac = jnp.stack(
        jax.tree.flatten([jacs["normal"][name][0] for name in jacs["normal"].keys()])[
            0
        ],
        axis=0,
    ).reshape(-1, 3, n_geom)
    normal_param_jac = jnp.concat(
        [normal_pose_jac, normal_geom_jac], axis=-1
    )  # (n_T*n_contacts, 3, n_param = n_q + n_geom)

    # Create multiplier
    normal_mult = (
        contact_bool * jnp.reciprocal(hyperparams.normal_var) * jnp.eye(3)
    )  # (n_T*n_contacts, 3, 3)

    # Create normal info
    normal_info = (
        jnp.swapaxes(normal_param_jac, -2, -1) @ normal_mult @ normal_param_jac
    )

    return phi_info + normal_info


## Function to compute outputs from diffsim data
def get_outputs_from_data(
    params: dict[str, dict[str, jax.Array]],
    qpos: jax.Array,
    data: mjx.Data,
    base_model: mjx.Model,
    obj_geom_names: list[str],
    collision_geom_names: list[str],
) -> dict[str, jax.Array]:
    """Compute outputs (phi and normals) from data"""
    assert len(obj_geom_names) == 1
    # write pose and params to model/data
    param_data = data.replace(qpos=qpos)
    param_model = LearnedModel.write_params_to_model(params, base_model)

    forward_data = mjx_util.jit_forward(param_model, param_data)

    contact_masks = {
        geom_name: mjx_util.contactids_from_collision_geoms(
            param_model, [geom_name], obj_geom_names
        )
        for geom_name in collision_geom_names
    }
    phis = {
        geom_name: jnp.sum(
            forward_data.contact.dist * jnp.abs(contact_mask[jnp.newaxis, :]),
            axis=-1,
            keepdims=True,
        )
        for geom_name, contact_mask in contact_masks.items()
    }
    normals = {
        geom_name: jnp.mean(
            jnp.sum(
                contact_mask[jnp.newaxis, :, jnp.newaxis]
                * forward_data.contact.frame[..., 0, :],
                axis=-2,
                keepdims=True,
            ),
            axis=-2,
        )
        for geom_name, contact_mask in contact_masks.items()
    }

    return {
        "phi": phis,
        "normal": normals,
    }


vmap_get_outputs_from_data = jax.jit(
    jax.vmap(get_outputs_from_data, in_axes=(None, 0, 0, None, None, None)),
    static_argnums=(-1, -2),
)

jac_get_outputs_from_data = jax.jit(
    jax.jacrev(vmap_get_outputs_from_data, argnums=(0, 1)),
    static_argnums=(-1, -2),
)


def expected_info(
    ctrl: jax.Array,
    params: tuple[dict[str, dict[str, jax.Array]], dict[str, jax.Array]],
    collision_geom_names: list[str],
    base_model: mjx.Model,
    hyperparams: InfoHyperparameters,
) -> jax.Array:
    """Calculate expected info

    Args:
        ctrl: (traj_len, n_control)
        params: (geometry params, learned trajectory param *final q*)
        base_model: model from initial mcjf/urdf
        hyperparams
    Returns:
        n_params x n_params expected info
    """
    # TODO: add other styles
    assert hyperparams.style == InfoStyle.IDENTITY

    # Get current position
    data_current = mjx_util.write_qpos_to_data(
        base_model, mjx.make_data(base_model), params[1]
    )

    # Sim Forward
    data_stacked = mjx_util.diffsim(base_model, data_current, ctrl, stacked=True)

    # Get Outputs
    outputs = vmap_get_outputs_from_data(
        params[0],
        data_stacked.qpos,
        data_stacked,
        base_model,
        frozenset(params[1].keys()),
        frozenset(collision_geom_names),
    )

    # Get Jacobian w.r.t. outputs
    output_jacs = jac_get_outputs_from_data(
        params[0],
        data_stacked.qpos,
        data_stacked,
        base_model,
        frozenset(params[1].keys()),
        frozenset(collision_geom_names),
    )

    breakpoint()

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
