#!/usr/bin/env python3

"""
Exploration w/ EIG Functions
"""

from dataclasses import dataclass
from enum import Enum
import operator
from typing import Any, Optional

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


## Functions to compute outputs from measurements
def get_outputs_from_measurements(
    params: tuple[dict[str, dict[str, jax.Array]], dict[str, dict[str, jax.Array]]],
    measurements: dict[str, dict[str, jax.Array]],
    base_model: mjx.Model,
) -> dict[str, jax.Array]:
    """Compute outputs (phi and normals) given measurements"""
    # write pose and params to model/data
    param_model = LearnedModel.write_params_to_model(
        params[0], base_model, needs_sim=False
    )
    # Forward will compute all physical parameters and distances
    forward_data = mjx_util.jit_forward(
        param_model,
        mjx_util.write_qpos_qvel_to_data(
            param_model, mjx.make_data(param_model), measurements | params[1]
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


@jax.jit
def get_outputs_from_sim(
    params: tuple[dict[str, dict[str, jax.Array]], dict[str, jax.Array]],
    measurements: dict[str, dict[str, jax.Array]],
    base_model: mjx.Model,
) -> dict[str, jax.Array]:
    """Compute outputs (phi and normals) through diffsim

    NOTE: assumes params[1] is only the starting position
    NOTE: measurements should contain object params as well!
    """
    # write pose and params to model/data
    param_model = LearnedModel.write_params_to_model(
        params[0], base_model, needs_sim=True
    )
    first_meas = jax.tree.map(lambda leaf: leaf[0, ...], measurements)
    start_data = mjx_util.write_qpos_to_data(
        param_model,
        mjx_util.write_qpos_qvel_to_data(
            param_model, mjx.make_data(base_model), first_meas
        ),
        params[1],
    )

    # Sim Forward
    step_data = mjx_util.diffsim_overwrite(
        param_model,
        start_data,
        measurements["ctrl"],
        measurements,
        stacked=True,
        keep_grad=True,
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
            step_data.contact.dist * jnp.abs(contact_mask[jnp.newaxis, :]),
            axis=-1,
            keepdims=True,
        )[
            ..., jnp.newaxis, :
        ]  # Unsqueeze -2
        for geom_name, contact_mask in contact_masks.items()
    }
    normals = {
        geom_name: jnp.mean(
            jnp.sum(
                contact_mask[jnp.newaxis, :, jnp.newaxis]
                * step_data.contact.frame[..., 0, :],
                axis=-2,
                keepdims=True,
            ),
            axis=-2,
        )[
            ..., jnp.newaxis, :
        ]  # Unsqueeze -2
        for geom_name, contact_mask in contact_masks.items()
    }

    geom_posvels = mjx_util.extract_geom_qposvel_from_data(
        param_model, step_data, tuple(params[1].keys())
    )
    final_pose = {
        geom_name: geom_posvels[geom_name]["position"][-1, ...]
        for geom_name in geom_posvels
    }

    return {"phi": phis, "normal": normals, "final_pose": final_pose}


# Diffsim needs forward-mode (and it is faster with long graphs)
# see https://github.com/google-deepmind/mujoco/issues/2259
jac_get_outputs_from_sim = jax.jit(jax.jacfwd(get_outputs_from_sim))


## Functions to compute info from Jacobians
def _info_from_jacs(
    outputs: Any,
    jacs: Any,
    hyperparams: InfoHyperparameters,
) -> jax.Array:
    """Compute observed info from given outputs and output jacobians w.r.t. parameters
    Identity-Style Only
    """
    assert len(jacs["phi"][list(jacs["phi"])[0]][0].keys()) == 1
    obj_geom_name = list(jacs["phi"][list(jacs["phi"])[0]][0].keys())[0]

    n_geom = jax.tree.reduce(
        operator.add,
        jax.tree.map(
            lambda leaf: leaf.shape[-1],
            jacs["phi"][list(jacs["phi"])[0]][0][obj_geom_name],
        ),
    )
    if hyperparams.style == InfoStyle.IDENTITY:
        n_q = jacs["phi"][list(jacs["phi"])[0]][1][obj_geom_name]["position"].shape[-1]
    elif hyperparams.style == InfoStyle.DIFFSIM:
        n_q = jacs["phi"][list(jacs["phi"])[0]][1][obj_geom_name].shape[-1]
    else:
        raise NotImplementedError(f"Style {hyperparams.style} not implemented.")

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

    # Handle w.r.t. state
    if hyperparams.style == InfoStyle.IDENTITY:
        # Assume jac == identity, note that (... n_T, ..., n_T, :) is block-diagonal
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
    elif hyperparams.style == InfoStyle.DIFFSIM:
        # Jac already w.r.t. start state
        phi_pose_jac = jnp.stack(
            [jacs["phi"][name][1][obj_geom_name] for name in jacs["phi"].keys()], axis=0
        ).reshape(-1, 1, n_q)
    else:
        raise NotImplementedError(f"Style {hyperparams.style} not implemented.")

    phi_param_jac = jnp.concat(
        [phi_pose_jac, phi_geom_jac], axis=-1
    )  # (n_T*n_contacts, 1, n_param = n_q + n_geom)
    phi_info = jnp.swapaxes(phi_param_jac, -2, -1) @ phi_mult @ phi_param_jac

    ## Handle Normal
    # Handle w.r.t. state
    if hyperparams.style == InfoStyle.IDENTITY:
        # Assume jac == identity, note that (... n_T, ..., n_T, :) is block-diagonal
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
    elif hyperparams.style == InfoStyle.DIFFSIM:
        # Jac already w.r.t. start state
        normal_pose_jac = jnp.stack(
            [jacs["normal"][name][1][obj_geom_name] for name in jacs["normal"].keys()],
            axis=0,
        ).reshape(-1, 3, n_q)
    else:
        raise NotImplementedError(f"Style {hyperparams.style} not implemented.")
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

    return jnp.sum(phi_info + normal_info, axis=0)


def observed_info(
    params: tuple[dict[str, dict[str, jax.Array]], dict[str, dict[str, jax.Array]]],
    measurements: dict[str, dict[str, jax.Array]],
    base_model: mjx.Model,
    hyperparams: InfoHyperparameters,
) -> tuple[jax.Array, Optional[jax.Array]]:
    """Calculate observed info

    Args:
        ctrl: (traj_len, n_control)
        params: geometry params
        measurements: contact and robot proprioceptio/home/ethankg/Workspace/test/data/traj_01.pkln and control data,
                        as a full trajectory (not a list of trajectories)
        base_model: model from initial mcjf/urdf
    Returns:
        n_params x n_params observed info
        Optional: the jacobian of the final pose w.r.t. the geometry and initial pose
    """
    jac_final_pose = None
    if hyperparams.style == InfoStyle.IDENTITY:

        # Get phi/normal outputs and parameter jacobians
        outputs = vmap_get_outputs_from_measurements(params, measurements, base_model)
        jacs = jac_get_outputs_from_measurements(params, measurements, base_model)
    elif hyperparams.style == InfoStyle.DIFFSIM:
        outputs = get_outputs_from_sim(
            (
                params[0],
                {
                    geom_name: params[1][geom_name]["position"][0, ...]
                    for geom_name in params[1]
                },
            ),
            measurements | params[1],
            base_model,
        )
        jacs = jac_get_outputs_from_sim(
            (
                params[0],
                {
                    geom_name: params[1][geom_name]["position"][0, ...]
                    for geom_name in params[1]
                },
            ),
            measurements | params[1],
            base_model,
        )
        jac_final_pose = jacs["final_pose"]
    else:
        raise NotImplementedError(f"Style {hyperparams.style} not implemented.")

    return _info_from_jacs(outputs, jacs, hyperparams), jac_final_pose


def _expected_info_identity(
    ctrl: jax.Array,
    params: tuple[dict[str, dict[str, jax.Array]], dict[str, jax.Array]],
    robot_geom_names: tuple,
    base_model: mjx.Model,
    hyperparams: InfoHyperparameters,
) -> jax.Array:
    """Calculate expected info using the identity style"""

    assert hyperparams.style == InfoStyle.IDENTITY

    # Get current position of object and robot
    data_current = mjx_util.write_qpos_to_data(
        base_model,
        mjx.make_data(base_model),
        params[1]
        | {
            geom_name: ctrl[0, 3 * idx : 3 * (idx + 1)]
            for idx, geom_name in enumerate(robot_geom_names)
        },
    )

    # Sim Forward
    data_stacked = mjx_util.diffsim(base_model, data_current, ctrl, stacked=True)

    # Extract parameters
    qpos_params = mjx_util.extract_geom_qposvel_from_data(
        base_model, data_stacked, frozenset(params[1].keys())
    )
    robot_pose = mjx_util.extract_geom_qposvel_from_data(
        base_model, data_stacked, robot_geom_names
    )
    for val in robot_pose.values():
        val["contact_normal_W"] = jnp.zeros_like(
            val["position"]
        )  # Mark as making contact

    # Get Outputs
    outputs = vmap_get_outputs_from_measurements(
        (params[0], qpos_params), robot_pose, base_model
    )
    jacs = jac_get_outputs_from_measurements(
        (params[0], qpos_params), robot_pose, base_model
    )

    return _info_from_jacs(outputs, jacs, hyperparams)


def _expected_info_diffsim(
    ctrl: jax.Array,
    params: tuple[dict[str, dict[str, jax.Array]], dict[str, jax.Array]],
    robot_geom_names: tuple,
    base_model: mjx.Model,
    hyperparams: InfoHyperparameters,
) -> jax.Array:
    """Calculate expected info using the identity style"""
    # TODO: Implement
    assert hyperparams.style == InfoStyle.DIFFSIM


def expected_info(
    ctrl: jax.Array,
    params: tuple[dict[str, dict[str, jax.Array]], dict[str, jax.Array]],
    robot_geom_names: tuple,
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
    if hyperparams.style == InfoStyle.IDENTITY:
        return _expected_info_identity(
            ctrl, params, robot_geom_names, base_model, hyperparams
        )
    else:
        raise NotImplementedError(f"Style {hyperparams.style} not implemented.")
