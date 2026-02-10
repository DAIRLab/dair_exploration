#!/usr/bin/env python3

"""
Utilities for Mujoco XLA / JAX
"""

import jax

from mujoco import mjx
import numpy as np


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


def qposidx_from_geom_name(model: mjx.Model, name: str) -> int:
    bodyid = bodyid_from_geomid(model, geomid_from_geom_name(model, name))
    qposids = []
    # Loop through all joints in body
    for jntid in range(
        model.body_jntadr[bodyid], model.body_jntadr[bodyid] + model.body_jntnum[bodyid]
    ):
        nq = (
            1 if model.jnt_type[jntid] > 1 else (4 if model.jnt_type[jntid] == 1 else 7)
        )
        qposids.extend(
            list(range(model.jnt_qposadr[jntid], model.jnt_qposadr[jntid] + nq))
        )

    return np.array(qposids)


def qvelidx_from_geom_name(model: mjx.Model, name: str) -> int:
    bodyid = bodyid_from_geomid(model, geomid_from_geom_name(model, name))
    qvelids = []
    # Loop through all joints in body
    for jntid in range(
        model.body_jntadr[bodyid], model.body_jntadr[bodyid] + model.body_jntnum[bodyid]
    ):
        nv = (
            1 if model.jnt_type[jntid] > 1 else (3 if model.jnt_type[jntid] == 1 else 6)
        )
        qvelids.extend(
            list(range(model.jnt_dofadr[jntid], model.jnt_dofadr[jntid] + nv))
        )

    return np.array(qvelids)


# pylint: enable=missing-function-docstring


def contactids_from_collision_geoms(
    base_model: mjx.Model,
    base_data: mjx.Data,
    sensor_geoms: list[str],
    object_geoms: list[str],
) -> np.ndarray:
    """Return list of contact ids that correspond with a sensor contacting an object"""
    sensor_geomids = [geomid_from_geom_name(base_model, name) for name in sensor_geoms]
    object_geomids = [geomid_from_geom_name(base_model, name) for name in object_geoms]

    # MJX requires access via _impl
    # pylint: disable=protected-access

    return np.where(
        np.logical_or(
            np.logical_and(
                np.isin(base_data._impl.contact.geom1, sensor_geomids),
                np.isin(base_data._impl.contact.geom2, object_geomids),
            ),
            np.logical_and(
                np.isin(base_data._impl.contact.geom1, object_geomids),
                np.isin(base_data._impl.contact.geom2, sensor_geomids),
            ),
        ),
    )[0]


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
) -> mjx.Model:
    """Returns a model with the parameters replaced."""
    model = base_model
    for geom_name in params.keys():
        geomid = geomid_from_geom_name(base_model, geom_name)
        for param_name, param in params[geom_name].items():
            if param_name == "size":
                model = model.replace(geom_size=model.geom_size.at[geomid].set(param))
            elif param_name == "friction":
                model = model.replace(
                    geom_friction=model.geom_friction.at[geomid].set(param)
                )
            elif param_name == "friction.sliding":
                model = model.replace(
                    geom_friction=model.geom_friction.at[geomid, 0].set(param)
                )
            else:
                raise NotImplementedError(
                    f"No implementation for parameter {param_name}"
                )
    return model


def write_qpos_to_data(
    base_model: mjx.Model, base_data: mjx.Data, traj_qpos: dict[str, jax.Array]
) -> mjx.Data:
    """Write a qpos parameter to MJX data object in a jax-traceable way"""
    ret_data = base_data
    for key, val in traj_qpos.items():
        ret_data = ret_data.replace(
            qpos=ret_data.qpos.at[qposidx_from_geom_name(base_model, key)].set(val)
        )
    return ret_data


def write_qvel_to_data(
    base_model: mjx.Model, base_data: mjx.Data, traj_qvel: dict[str, jax.Array]
) -> mjx.Data:
    """Write a qvel parameter to MJX data object in a jax-traceable way"""
    ret_data = base_data
    for key, val in traj_qvel.items():
        ret_data = ret_data.replace(
            qvel=ret_data.qvel.at[qvelidx_from_geom_name(base_model, key)].set(val)
        )
    return ret_data


## Compiled base functions
@jax.jit
def jit_step(model: mjx.Model, data: mjx.Data):
    """Simulation Step"""
    return mjx.step(model, data)


@jax.jit
def jit_forward(model: mjx.Model, data: mjx.Data):
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


def data_unstack(data: mjx.Data) -> list[mjx.Data]:
    """Unstack a data object with a batch dimension into a list of mjx datas"""
    leaves, treedef = jax.tree.flatten(data)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


@jax.jit
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

    def _sim_step(carry_data, ctrl):
        """Inner sim step"""
        ret_data = jit_step(model, carry_data.replace(ctrl=ctrl))
        return (ret_data, ret_data)

    _, data_stacked = jax.lax.scan(_sim_step, init_data, ctrl)

    return data_stacked
