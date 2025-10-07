#!/usr/bin/env python3

"""
Utilities for Mujoco XLA / JAX
"""

import jax

from mujoco import mjx


## Naming Utilities
# pylint: disable=missing-function-docstring
def bodyid_from_body_name(model: mjx.Model, name: str) -> int:
    return model.name_bodyadr.tolist().index(model.names.find(f"{name}\0".encode()))


def geomid_from_geom_name(model: mjx.Model, name: str) -> int:
    return model.name_geomadr.tolist().index(model.names.find(f"{name}\0".encode()))


def bodyid_from_geomid(model: mjx.Model, geomid: int) -> int:
    return model.body_rootid[model.geom_bodyid[geomid]]


def qposadr_from_bodyid(model: mjx.Model, bodyid: int) -> int:
    return model.jnt_qposadr[model.body_jntadr[bodyid]]


def qveladr_from_bodyid(model: mjx.Model, bodyid: int) -> int:
    return model.jnt_dofadr[model.body_jntadr[bodyid]]


def qposadr_from_geom_name(model: mjx.Model, name: str) -> int:
    return qposadr_from_bodyid(
        model, bodyid_from_geomid(model, geomid_from_geom_name(model, name))
    )


def qveladr_from_geom_name(model: mjx.Model, name: str) -> int:
    return qveladr_from_bodyid(
        model, bodyid_from_geomid(model, geomid_from_geom_name(model, name))
    )


# pylint: enable=missing-function-docstring


## Parameter Utilities
def populate_parameter_dict(base_model: mjx.Model, param_names: dict[str, list[str]]):
    """
    Populates a parameter dictionary with values from the base model.

    Args:
        base_model: model with the initial guess
        param_names: maps geom_name to list of parameters
    Returns:
        dict[str, dict[str, jax.Array]] of populated parameters
    """
    # TODO: Add Body Parameters (Mass / COM / Inertia)
    ret = {}
    for geom_name in param_names.keys():
        ret[geom_name] = {}
        geomid = geomid_from_geom_name(base_model, geom_name)
        for param in param_names[geom_name]:
            if param == "size":
                ret[geom_name][param] = base_model.geom_size[geomid].copy()
            elif param == "friction":
                ret[geom_name][param] = base_model.geom_friction[geomid].copy()
            elif param == "friction.sliding":
                ret[geom_name][param] = base_model.geom_friction[geomid, 0].copy()
            else:
                raise NotImplementedError(f"No implementation for parameter {param}")
    return ret


def write_params_to_model(
    base_model: mjx.Model, params: dict[str, dict[str, jax.Array]]
):
    """Returns a model with the parameters replaced."""
    model = base_model
    for geom_name in params.keys():
        geomid = geomid_from_geom_name(base_model, geom_name)
        for param_name, param in params[geom_name]:
            if param_name == "size":
                model.geom_size = model.geom_size.at[geomid].set(param)
            elif param_name == "friction":
                model.geom_friction = model.geom_friction.at[geomid].set(param)
            elif param_name == "friction.sliding":
                model.geom_friction = model.geom_friction.at[geomid, 0].set(param)
            else:
                raise NotImplementedError(
                    f"No implementation for parameter {param_name}"
                )
    return model


## Compiled base functions
@jax.jit
def jit_step(model: mjx.Model, data: mjx.Data):
    """Simulation Step"""
    return mjx.step(model, data)


@jax.jit
@jax.vmap(in_axes=(None, 0))
def jit_vmap_forward(model: mjx.Model, data: mjx.Data):
    """Simulation Step"""
    return mjx.forward(model, data)


## Diff Sim
def diffsim_overwrite(
    model: mjx.Model,
    init_data: mjx.Data,
    ctrl: jax.Array,
    qpos_overwrite: dict[str, jax.Array],
    qvel_overwrite: dict[str, jax.Array],
) -> list[mjx.Data]:
    """Simulate from init_data

    Params:
        ctrl: (n_timesteps, n_ctrl)
        qpos/qvel_overwrite: geom_name -> (n_timesteps, 3)

    Returns:
        list of new data objects from simulation
    """

    ret_list = [init_data]

    for timestep in range(ctrl.shape[0]):
        old_data = ret_list[-1]
        new_qpos = old_data.qpos
        new_qvel = old_data.qvel
        for geom_name in qpos_overwrite.keys():
            qposadr = qposadr_from_geom_name(model, geom_name)
            new_qpos = new_qpos.at[qposadr : qposadr + 3].set(
                qpos_overwrite[geom_name][timestep]
            )
        for geom_name in qvel_overwrite.keys():
            qveladr = qveladr_from_geom_name(model, geom_name)
            new_qvel = new_qvel.at[qveladr : qveladr + 3].set(
                qvel_overwrite[geom_name][timestep]
            )
        ret_list.append(
            jit_step(
                model,
                old_data.replace(qpos=new_qpos, qvel=new_qvel, ctrl=ctrl[timestep]),
            )
        )

    return ret_list[1:]


def diffsim(
    model: mjx.Model,
    init_data: mjx.Data,
    ctrl: jax.Array,
) -> list[mjx.Data]:
    """Simulate from init_data

    Params:
        ctrl: (n_timesteps, n_ctrl)

    Returns:
        list of new data objects from simulation
    """
    ret_list = [init_data]

    for timestep in range(ctrl.shape[0]):
        ret_list.append(jit_step(model, ret_list[-1].replace(ctrl=ctrl[timestep])))

    return ret_list[1:]
