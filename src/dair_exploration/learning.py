#!/usr/bin/env python3

"""Utility classes/functions for managing learning

The main contents of this file are as follows:

    * Class to hold and manage learnable parameters
"""
from collections.abc import Sequence
from enum import Enum

import gin
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from dair_exploration import file_util, mjx_util, data_util


@gin.configurable
class LearnedModel:
    """Geometric and intertial model parameters"""

    _params: dict[str, dict[str, jax.Array]] = None
    r"""Current object model parameters"""
    _active_spec: mujoco.MjSpec = None
    r"""Mujoco spec corresponding to current model parameters"""
    _active_model: mjx.Model = None
    r"""Cache of the active model associated with the current parameters"""

    def __init__(self, model_file: str, param_spec: dict[str, list[str]]):
        spec = mujoco.MjSpec.from_file(file_util.get_config(model_file).as_posix())
        # Populate params dict
        # TODO: add mass, CoM, inertia
        self._params = {}
        for geom_name in param_spec.keys():
            self._params[geom_name] = {}
            for param in param_spec[geom_name]:
                if param == "size":
                    if spec.geom(geom_name).type == mujoco.mjtGeom.mjGEOM_MESH:
                        # Mesh object
                        self._params[geom_name][param] = jnp.asarray(
                            spec.mesh(spec.geom(geom_name).meshname).uservert
                        )
                    elif spec.geom(geom_name).type == mujoco.mjtGeom.mjGEOM_BOX:
                        # Cuboid
                        self._params[geom_name][param] = jnp.asarray(
                            spec.geom(geom_name).size
                        )
                    else:
                        raise NotImplementedError(
                            f"'size' not supported for type {spec.geom(geom_name).type}"
                        )
                elif param == "friction":
                    self._params[geom_name][param] = jnp.asarray(
                        spec.geom(geom_name).friction
                    )
                elif param == "friction.sliding":
                    self._params[geom_name][param] = jnp.asarray(
                        spec.geom(geom_name).friction[0]
                    )
                else:
                    raise NotImplementedError(
                        f"No implementation for parameter {param}"
                    )
        self._active_spec = spec
        self._active_model = mjx.put_model(self._active_spec.compile())

    def write_to_file(self, model_name: str = "out"):
        """Write current spec to file"""
        file_util.write_text(
            self._active_spec.to_xml(), "learning", f"model_{model_name}.mjcf"
        )

    @property
    def params(self) -> dict[str, dict[str, jax.Array]]:
        """Get current parameters"""
        return self._params

    @params.setter
    def params(self, value) -> None:
        """Set parameters and update spec and active model"""
        assert jax.tree.structure(self._params) == jax.tree.structure(
            value
        ), "Can't change parameter tree structure"
        self._params = value

        # Write params to spec
        for geom_name in self._params.keys():
            for param_name in self._params[geom_name].keys():
                if param_name == "size":
                    # if (
                    #     self._active_spec.geom(geom_name).type
                    #     == mujoco.mjtGeom.mjGEOM_MESH
                    # ):
                    #     # Mesh object
                    #     self._active_spec.mesh(
                    #         self._active_spec.geom(geom_name).meshname
                    #     ).uservert = np.array(self._params[geom_name][param_name])
                    # TODO: add mesh support
                    if (
                        self._active_spec.geom(geom_name).type
                        == mujoco.mjtGeom.mjGEOM_BOX
                    ):
                        # Cuboid
                        self._active_spec.geom(geom_name).size = np.array(
                            self._params[geom_name][param_name]
                        )
                    else:
                        raise NotImplementedError(
                            "'size' not supported for type "
                            f"{self._active_spec.geom(geom_name).type}"
                        )
                elif param_name == "friction":
                    self._active_spec.geom(geom_name).friction = np.array(
                        self._params[geom_name][param_name]
                    )
                elif param_name == "friction.sliding":
                    self._active_spec.geom(geom_name).friction[0] = np.array(
                        self._params[geom_name][param_name]
                    )
                else:
                    raise NotImplementedError(
                        f"No implementation for parameter {param_name}"
                    )

        # Compile spec to model
        self._active_model = mjx.put_model(self._active_spec.compile())

    @property
    def active_model(self) -> mjx.Model:
        """Get model associated with current parameters"""
        return self._active_model

    @staticmethod
    def write_params_to_model(
        params: dict[str, dict[str, jax.Array]], model: mjx.Model
    ) -> mjx.Model:
        """Write parameter dictionary to the active model in a jax-traceable fashion"""
        for geom_name in params.keys():
            geomid = mjx_util.geomid_from_geom_name(model, geom_name)
            for param_name, param in params[geom_name].items():
                if param_name == "size":
                    # TODO: add mesh support
                    assert model.geom_type[geomid] == mujoco.mjtGeom.mjGEOM_BOX
                    model = model.replace(
                        geom_size=model.geom_size.at[geomid].set(param)
                    )
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


class TrajParamKey(Enum):
    """Key for the parameter dictionary"""

    Q0 = 0
    TRAJQ = 1
    TRAJV = 2


@gin.configurable(denylist=["base_model"])
class LearnedTrajectory(Sequence):
    """Learned Trajectory Parameters

    dict["object_geom_name", dict[(Q0/Q/V), jax.Array]
    Where for trajectory X[t], the states are:
    (Q0[t]/0, Q[t]/V[t], Q0[t+1]/0)
    """

    _params: dict[str, dict[TrajParamKey, list[jax.Array]]]
    r"""Current object trajectory parameters"""
    _fixed_len: int
    r"""Fixed length for each trajectory"""
    _base_model: mjx.Model
    r"""Base model used to help populate qpos/qvel"""

    def __init__(
        self, base_model: mjx.Model, geom_names: list[str], fixed_len: int = 0
    ):
        self._base_model = base_model
        self._fixed_len = fixed_len
        self._params = {}
        assert len(geom_names) > 0
        for geom_name in geom_names:
            self._params[geom_name] = {}
            qposids = mjx_util.qposidx_from_geom_name(base_model, geom_name)

            # Initialize empty or Qpos0
            self._params[geom_name][TrajParamKey.TRAJQ] = []
            self._params[geom_name][TrajParamKey.TRAJV] = []
            self._params[geom_name][TrajParamKey.Q0] = [
                jnp.expand_dims(base_model.qpos0[qposids], axis=0)
            ]

    @property
    def params(self):
        """Raw parameter object"""
        return self._params

    def __len__(self):
        """Number of trajectory parameters stored"""
        return len(next(iter(self._params.values()))[TrajParamKey.TRAJQ])

    def __getitem__(self, idx: int) -> jax.Array:
        """Return a contiguous trajectory, if fixed_len pad the beginning w/ (Q0/0)"""
        ret = {}
        for geom_name, geom_traj in self._params.items():
            pad_len = (
                self._fixed_len - (len(geom_traj[TrajParamKey.TRAJQ]) + 2)
                if self._fixed_len > 0
                else 0
            )
            ret[geom_name] = {}
            ret[geom_name]["position"] = jnp.concatenate(
                [
                    jnp.repeat(
                        geom_traj[TrajParamKey.Q0][idx],
                        1 + pad_len,
                        axis=0,
                    ),
                    geom_traj[TrajParamKey.TRAJQ],
                    geom_traj[TrajParamKey.Q0][idx + 1],
                ]
            )
            n_v = geom_traj[TrajParamKey.TRAJV][0].shape[-1]
            ret[geom_name]["velocity"] = jnp.concatenate(
                [
                    jnp.repeat(jnp.zeros((1, len(n_v))), 1 + pad_len, axis=0),
                    geom_traj[TrajParamKey.TRAJV],
                    jnp.zeros((1, len(n_v))),
                ]
            )
        return ret

    def write_to_file(self, traj_name: str = "out"):
        """Write current spec to file"""
        file_util.write_object(self._params, "learning", f"traj_{traj_name}.pkl")


@gin.constants_from_enum
class LossStyle(Enum):
    """Loss Style"""

    DIFFSIM = 0
    VIMP = 1


@jax.jit
def loss_diffsim(
    params: tuple[dict[str, dict[str, jax.Array]], dict[str, jax.Array]],
    measurements: dict[str, dict[str, jax.Array]],
    active_model: mjx.Model,
) -> jax.Array:
    """Diffsim loss function for training

    Parameters:
        * params: tuple of (model_params, traj_param)
        * * where traj_param is specifically q0 for each learnable geometry
        * measurements: contact and robot proprioception and control data,
        * * as a full trajectory (not a list of trajectories)
        * active_model: the mjx model in which to create data and write params
    """
    pass


@gin.configurable(allowlist=["loss_style"])
def train_epochs(
    learned_model: LearnedModel,
    learned_traj: LearnedTrajectory,
    measurements: data_util.TrajectorySet,
    n_epochs: int,
    epoch_start: int = 0,
    loss_style: LossStyle = LossStyle.DIFFSIM,
) -> None:  # TODO: return loss statistics
    """Train and update the learned model and trajectory on the measurements.

        * Initialize optax (gin-configured).
        Foreach epoch in range(epoch_start, epoch_start + n_epochs):
        ** Pull parameters from learned model/traj
        ** Call gin-configured loss function and gradient (jax-traceably)
        ** Use optax to update parameters
        ** Write new parameters to the learned model / traj
        ** Write new parameters to file
        ** TODO: Record loss statistics and write to file
        * If Ctrl-C is called, finish current epoch and return

    Return (TODO) loss statistics; Learned Model/Trajectory are mutated.
    """
    loss_fn = loss_diffsim  # TODO: switch based on lossstyle
