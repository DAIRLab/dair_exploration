#!/usr/bin/env python3

"""
Differentiable Simulation using Mujoco XLA / JAX
"""

import jax
from mujoco import mjx


# Mujoco Utilities
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


# Compiled base functions
@jax.jit
def jit_step(model: mjx.Model, data: mjx.Data):
    """Simulation Step"""
    return mjx.step(model, data)


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
