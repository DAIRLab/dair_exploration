#!/usr/bin/env python3

"""Utility classes/functions for managing learning

The main contents of this file are as follows:

    * Class to hold and manage learnable parameters
"""
import gin
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from dair_exploration import file_util, mjx_util


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


# class LearnedTrajectory:
#     pass


# def loss_vimp(
#     params: dict[str, dict[str, jax.Array]],
#     traj_params: dict[str, tuple[jax.Array, jax.Array]],
#     data: dict[str, jax.Array],
#     base_model: mjx.Model,
# ) -> jax.Array:
#     # TODO: write params to base_model
#     # TODO: combine traj_params / data into trajectory
#     # TODO:
#     pass
