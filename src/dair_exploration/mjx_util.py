#!/usr/bin/env python3

"""
Utilities for Mujoco XLA / JAX
"""

from functools import partial
from typing import Union

import jax
import jax.numpy as jnp

from mujoco import mjx
import numpy as np

from dair_exploration.jax_util import overwrite_keep_gradient


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


def qposidx_from_geom_name(model: mjx.Model, name: str) -> np.ndarray:
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


def qposidx_from_geom_names(model: mjx.Model, names: list[str]) -> np.ndarray:
    return np.concatenate([qposidx_from_geom_name(model, name) for name in names])


def qvelidx_from_geom_name(model: mjx.Model, name: str) -> np.ndarray:
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


def qvelidx_from_geom_names(model: mjx.Model, names: list[str]) -> np.ndarray:
    return np.concatenate([qvelidx_from_geom_name(model, name) for name in names])


# pylint: enable=missing-function-docstring


def contactids_from_geoms(
    base_model: mjx.Model,
    object_geoms: list[str],
) -> np.ndarray:
    """Return mask (n_contactids) of contactids where:
    (0 == no contact, 1 == contact)
    """
    object_geomids = [geomid_from_geom_name(base_model, name) for name in object_geoms]

    base_data = jit_forward(base_model, mjx.make_data(base_model))

    mask = jnp.logical_or(
        jnp.isin(base_data.contact.geom1, jnp.array(object_geomids)),
        jnp.isin(base_data.contact.geom2, jnp.array(object_geomids)),
    ).astype(float)

    return mask


def contactids_from_collision_geoms(
    base_model: mjx.Model,
    sensor_geoms: list[str],
    object_geoms: list[str],
) -> np.ndarray:
    """Return mask (n_contactids) of contactids where:
    (0 == no contact, -1 == to sensor, +1 == from sensor)
    """
    # TODO: confirm that this works for stacked data
    sensor_geomids = [geomid_from_geom_name(base_model, name) for name in sensor_geoms]
    object_geomids = [geomid_from_geom_name(base_model, name) for name in object_geoms]

    base_data = jit_forward(base_model, mjx.make_data(base_model))

    from_mask = jnp.logical_and(
        jnp.isin(base_data.contact.geom1, jnp.array(object_geomids)),
        jnp.isin(base_data.contact.geom2, jnp.array(sensor_geomids)),
    ).astype(float)
    to_mask = jnp.logical_and(
        jnp.isin(base_data.contact.geom2, jnp.array(object_geomids)),
        jnp.isin(base_data.contact.geom1, jnp.array(sensor_geomids)),
    ).astype(float)

    return from_mask - to_mask


## Parameter Utilities
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


def write_qpos_qvel_to_data(
    base_model: mjx.Model,
    base_data: mjx.Data,
    traj_qpos_qvel: dict[str, dict[str, jax.Array]],
) -> mjx.Data:
    """Write qvel and qpos parameters to MJX data object in a jax-traceable way"""
    ret_data = base_data
    for key, val in traj_qpos_qvel.items():
        if not isinstance(val, dict):
            continue
        if "position" not in val:
            continue
        if "velocity" not in val:
            continue
        ret_data = ret_data.replace(
            qpos=ret_data.qpos.at[qposidx_from_geom_name(base_model, key)].set(
                val["position"]
            ),
            qvel=ret_data.qvel.at[qvelidx_from_geom_name(base_model, key)].set(
                val["velocity"]
            ),
        )
    return ret_data


def extract_geom_qposvel_from_data(
    base_model: mjx.Model,
    data: mjx.Data,
    geoms: Union[frozenset, tuple],
) -> dict[str, dict[str, jax.Array]]:
    """Inverse of write_qpos_qvel_to_data"""
    ret = {
        geom_name: {
            "position": None,
            "velocity": None,
        }
        for geom_name in geoms
    }

    for geom_name in geoms:
        ret[geom_name]["position"] = data.qpos[
            ..., qposidx_from_geom_name(base_model, geom_name)
        ]
        ret[geom_name]["velocity"] = data.qvel[
            ..., qvelidx_from_geom_name(base_model, geom_name)
        ]

    return ret


## Compiled base functions
@jax.jit
def jit_step(model: mjx.Model, data: mjx.Data):
    """Simulation Step"""
    return mjx.step(model, data)


@jax.jit
def jit_forward(model: mjx.Model, data: mjx.Data):
    """Forward Step"""
    return mjx.forward(model, data)


@jax.jit
@partial(jax.vmap, in_axes=(None, 0))
def jit_vmap_kinematics(model: mjx.Model, data: mjx.Data):
    """Kinematics Step"""
    return mjx.kinematics(model, data)


## Diff Sim
@partial(jax.jit, static_argnames=["stacked", "keep_grad"])
def diffsim_overwrite(
    model: mjx.Model,
    init_data: mjx.Data,
    ctrl: jax.Array,
    posvel_overwrite: dict[str, dict[str, jax.Array]],
    stacked: bool = False,
    keep_grad: bool = False,
) -> list[mjx.Data]:
    """Simulate from init_data

    Params:
        ctrl: (n_timesteps, n_ctrl)
        posvel_overwrite: geom_name ->
            {"position": (n_timesteps, n_q), "velocity": (n_timesteps, n_v)}
        stacked: if True, leave final data stacked
        keep_grad: if True, keep gradient through the overwrite step

    Returns:
        list of new data objects from simulation
    """

    def _sim_step(carry_data, in_x):
        """Inner sim step"""
        ctrl, posvel = in_x
        new_qpos = carry_data.qpos
        new_qvel = carry_data.qvel
        for geom_name in posvel.keys():
            if not isinstance(posvel[geom_name], dict):
                continue
            qposidx = qposidx_from_geom_name(model, geom_name)
            qvelidx = qvelidx_from_geom_name(model, geom_name)
            new_qpos = new_qpos.at[qposidx].set(posvel[geom_name]["position"])
            new_qvel = new_qvel.at[qvelidx].set(posvel[geom_name]["velocity"])

        if keep_grad:
            new_qpos = overwrite_keep_gradient(carry_data.qpos, new_qpos)
            new_qvel = overwrite_keep_gradient(carry_data.qvel, new_qvel)

        ret_data = jit_step(
            model,
            carry_data.replace(ctrl=ctrl, qpos=new_qpos, qvel=new_qvel),
        )
        return (ret_data, ret_data)

    _, data_stacked = jax.lax.scan(_sim_step, init_data, (ctrl, posvel_overwrite))

    return data_stacked if stacked else data_unstack(data_stacked)


@jax.jit
def data_unstack(data: mjx.Data) -> list[mjx.Data]:
    """Unstack a data object with a batch dimension into a list of mjx datas"""
    leaves, treedef = jax.tree.flatten(data)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


@partial(jax.jit, static_argnames="stacked")
def diffsim(
    model: mjx.Model,
    init_data: mjx.Data,
    ctrl: jax.Array,
    stacked: bool = False,
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

    return data_stacked if stacked else data_unstack(data_stacked)
