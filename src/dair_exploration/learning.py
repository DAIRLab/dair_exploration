#!/usr/bin/env python3

"""Utility classes/functions for managing learning

The main contents of this file are as follows:

    * Class to hold and manage learnable parameters
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import operator
import time
from typing import Optional, Any

import gin
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import optax
from mpax import create_qp, raPDHG

from dair_exploration import file_util, mjx_util, data_util
from dair_exploration.gui_util import MJXMeshcatVisualizer


@gin.configurable
class LearnedModel:
    """Geometric and intertial model parameters"""

    _params: dict[str, dict[str, jax.Array]] = None
    r"""Current object model parameters"""
    _active_spec: mujoco.MjSpec = None
    r"""Mujoco spec corresponding to current model parameters"""
    _base_model: mjx.Model = None
    r"""Cache of the initial model"""
    _min_size: float = 1e-3
    r"""Minimum size for cuboid"""

    def __init__(
        self, model_file: str, param_spec: dict[str, list[str]], min_size: float = 1e-3
    ):
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
                        self._params[geom_name][param] = jnp.maximum(
                            jnp.asarray(spec.geom(geom_name).size),
                            min_size
                            * jnp.ones_like(jnp.asarray(spec.geom(geom_name).size)),
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
        self._base_model = mjx.put_model(self._active_spec.compile())

    def write_to_file(self, model_name: str = "out"):
        """Write current spec to file"""
        file_util.write_text(
            self._active_spec.to_xml(), "learning", f"model_{model_name}.xml"
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
                        # Cuboid, clamp to min_size
                        self._params[geom_name][param_name] = jnp.maximum(
                            self._params[geom_name][param_name],
                            self._min_size
                            * jnp.ones_like(self._params[geom_name][param_name]),
                        )
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

    @property
    def base_model(self):
        """Model with initial parameters, use for training to avoid re-JIT"""
        return self._base_model

    @property
    def active_model(self) -> mjx.Model:
        """Get model associated with current parameters"""
        return mjx.put_model(self._active_spec.compile())

    @staticmethod
    def write_params_to_model(
        params: dict[str, dict[str, jax.Array]],
        model: mjx.Model,
        needs_sim: bool = True,
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
        # Margin > Gap can be useful for non-sim computations to generate
        # "Inactive" Contacts
        if not needs_sim:
            model = model.replace(geom_gap=jnp.zeros_like(model.geom_gap))
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

    def extend_traj(self, n_timesteps: int):
        """Extend the learned trajectory by a known number of timesteps."""
        for geom_name, geom_traj in self._params.items():
            last_pose = geom_traj[TrajParamKey.Q0][-1]
            geom_traj[TrajParamKey.Q0].append(last_pose)
            geom_traj[TrajParamKey.TRAJQ].append(
                jnp.repeat(
                    last_pose,
                    n_timesteps - 2,
                    axis=0,
                )
            )
            zero_velocity = jnp.zeros(
                (1, len(mjx_util.qvelidx_from_geom_name(self._base_model, geom_name)))
            )
            geom_traj[TrajParamKey.TRAJV].append(
                jnp.repeat(
                    zero_velocity,
                    n_timesteps - 2,
                    axis=0,
                )
            )

    @property
    def init_q(self) -> dict[str, jax.Array]:
        """Initial position"""
        return {
            geom_name: geom_traj[TrajParamKey.Q0][0].squeeze()
            for geom_name, geom_traj in self._params.items()
        }

    @init_q.setter
    def init_q(self, value) -> None:
        """New initial position"""
        for geom_name, qpos0 in value.items():
            assert len(qpos0) == self._params[geom_name][TrajParamKey.Q0][0].shape[-1]
            self._params[geom_name][TrajParamKey.Q0][0] = jnp.expand_dims(qpos0, axis=0)

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
                self._fixed_len - (len(geom_traj[TrajParamKey.TRAJQ][idx]) + 2)
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
                    geom_traj[TrajParamKey.TRAJQ][idx],
                    geom_traj[TrajParamKey.Q0][idx + 1],
                ]
            )
            n_v = geom_traj[TrajParamKey.TRAJV][0].shape[-1]
            ret[geom_name]["velocity"] = jnp.concatenate(
                [
                    jnp.repeat(jnp.zeros((1, n_v)), 1 + pad_len, axis=0),
                    geom_traj[TrajParamKey.TRAJV][idx],
                    jnp.zeros((1, n_v)),
                ]
            )
        return ret

    def __setitem__(self, idx: int, new_traj: dict[str, dict[str, jax.Array]]):
        """Write parameter array from (padded) input to corresponding trajectory"""
        for geom_name, geom_traj in self._params.items():
            pad_len = (
                self._fixed_len - (len(geom_traj[TrajParamKey.TRAJQ][idx]) + 2)
                if self._fixed_len > 0
                else 0
            )
            assert (
                len(new_traj[geom_name]["position"])
                == len(new_traj[geom_name]["velocity"])
                == len(geom_traj[TrajParamKey.TRAJQ][idx]) + 2 + pad_len
            )
            geom_traj[TrajParamKey.Q0][idx] = jnp.mean(
                new_traj[geom_name]["position"][: (pad_len + 1), ...],
                axis=0,
                keepdims=True,
            )
            geom_traj[TrajParamKey.TRAJQ][idx] = new_traj[geom_name]["position"][
                (pad_len + 1) : -1
            ]
            geom_traj[TrajParamKey.TRAJV][idx] = new_traj[geom_name]["velocity"][
                (pad_len + 1) : -1
            ]
            geom_traj[TrajParamKey.Q0][idx + 1] = new_traj[geom_name]["position"][-1:]

    def get_full_trajectory(self):
        """Concat all elements into a single full trajectory"""
        ret = {}
        for geom_name, geom_traj in self._params.items():
            ret[geom_name] = {}
            ret[geom_name]["position"] = jnp.concatenate(
                [
                    jnp.concatenate(
                        [
                            geom_traj[TrajParamKey.Q0][idx],
                            geom_traj[TrajParamKey.TRAJQ][idx],
                        ]
                    )
                    for idx in range(len(geom_traj[TrajParamKey.TRAJQ]))
                ]
                + [geom_traj[TrajParamKey.Q0][-1]]
            )
            n_v = geom_traj[TrajParamKey.TRAJV][0].shape[-1]
            ret[geom_name]["velocity"] = jnp.concatenate(
                [
                    jnp.concatenate(
                        [
                            jnp.zeros((1, n_v)),
                            geom_traj[TrajParamKey.TRAJV][idx],
                        ]
                    )
                    for idx in range(len(geom_traj[TrajParamKey.TRAJV]))
                ]
                + [jnp.zeros((1, n_v))]
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


@gin.configurable
@dataclass(frozen=True)
class LearningHyperparameters:
    """Class to specify loss hyperparameters"""

    # Switches
    sim_overwrite: bool = True  # Whether to overwrite sim or just do open-loop control

    # Loss Weights
    phi_nominal: float = 0.002  # m, distance where p(contact_measured) drops below CI
    phi_ci: float = 0.05  # Confidence Interval (0,1) for above
    normal_var: float = (
        0.01519224261  # cos(radians) [default 10 degrees], variance of cos(normal angle deviation)
    )
    w_pen: float = 0.0  # cost/m
    w_q_pred: float = 1e0  # cost/J
    w_v_pred: float = 1e0  # cost/J
    w_comp: float = 1e0  # cost/J
    w_diss: float = 1e0  # cost/J
    w_elas: float = 1e0  # cost/J

    # Regularizers
    reg_grounded: float = 1e1  # multiplier on distance from ground a T=0
    ground_geom: str = "ground-geom"

    # Computation Parameters
    epsilon: float = 1e-8


def _get_measurement_loss_and_outputs(
    model: mjx.Model,
    data_stacked: mjx.Data,
    measurements: dict[str, dict[str, jax.Array]],
    obj_geom_names: list[str],
    hyperparams: LearningHyperparameters,
) -> Any:
    """Return the measurement loss and phis/normals for a given stack of data against the measurements."""
    assert len(obj_geom_names) == 1  # Only 1 object supported
    contact_masks = {
        geom_name: mjx_util.contactids_from_collision_geoms(
            model, [geom_name], obj_geom_names
        )
        for geom_name in measurements.keys()
        if "contact_normal_W" in measurements[geom_name]
    }
    phis = {
        geom_name: jnp.sum(
            data_stacked.contact.dist * jnp.abs(contact_mask[jnp.newaxis, :]),
            axis=-1,
            keepdims=True,
        )
        for geom_name, contact_mask in contact_masks.items()
    }
    normals = {
        geom_name: jnp.mean(
            jnp.sum(
                contact_mask[jnp.newaxis, :, jnp.newaxis]
                * data_stacked.contact.frame[..., 0, :],
                axis=-2,
                keepdims=True,
            ),
            axis=-2,
        )
        for geom_name, contact_mask in contact_masks.items()
    }

    # Compare outputs to measurements for loss
    loss = {}
    contact_bools = {
        geom_name: jnp.round(
            jnp.linalg.norm(
                measurements[geom_name]["contact_normal_W"], axis=-1, keepdims=True
            )
        )
        for geom_name in measurements.keys()
        if "contact_normal_W" in measurements[geom_name]
    }
    meas_normals = {
        geom_name: measurements[geom_name]["contact_normal_W"]
        for geom_name in measurements.keys()
        if "contact_normal_W" in measurements[geom_name]
    }

    phi_alpha = (
        np.log(np.reciprocal(hyperparams.phi_ci) - 1.0) / hyperparams.phi_nominal
    )

    loss["meas_normal"] = jax.tree.map(
        lambda normal, meas_normal, contact_bool, normal_var=hyperparams.normal_var: 0.5
        * contact_bool
        * jnp.reciprocal(normal_var)
        * (1.0 - jnp.sum(normal * meas_normal, axis=-1, keepdims=True)),
        normals,
        meas_normals,
        contact_bools,
    )

    loss["meas_contact"] = jax.tree.map(
        lambda phi, contact_bool, phi_alpha=phi_alpha: (contact_bool - 1.0)
        * phi_alpha
        * phi
        - jax.nn.log_sigmoid(
            -(phi_alpha * phi)
        ),  # -logsigmoid(-x) == log(1+exp(x)), more numerically stable
        phis,
        contact_bools,
    )

    # Add Penetration Loss, any contact_id for the learned object
    dist_pen = jnp.maximum(
        -data_stacked.contact.dist
        * mjx_util.contactids_from_geoms(model, obj_geom_names)[jnp.newaxis, :],
        jnp.zeros_like(data_stacked.contact.dist),
    )
    loss["pen"] = hyperparams.w_pen * jnp.sum(dist_pen, axis=-1)

    # Add ground regularizer
    # Note: collision_convex.py adjusted so non-unique witness points aren't set to 1.
    loss["reg_grounded"] = hyperparams.reg_grounded * jnp.sum(
        jnp.abs(
            (
                data_stacked.contact.dist
                * mjx_util.contactids_from_collision_geoms(
                    model, [hyperparams.ground_geom], obj_geom_names
                )
            )[0, ...]
        )
    )

    return loss, {"phis": phis, "normals": normals, "dist_pen": dist_pen}


@gin.configurable(allowlist=["hyperparams"])
def loss_vimp(
    params: tuple[dict[str, dict[str, jax.Array]], dict[str, jax.Array]],
    measurements: dict[str, dict[str, jax.Array]],
    base_model: mjx.Model,
    hyperparams: LearningHyperparameters,
) -> tuple[float, Any]:
    """Violation-Implicit Loss for Training

    Parameters:
        * params: tuple of (model_params, traj_param)
        * * where traj_param is specifically q0 for each learnable geometry
        * measurements: contact and robot proprioception and control data,
        * * as a full trajectory (not a list of trajectories)
        * base_model: the mjx model in which to create data and write params

    Returns:
        * Loss (scalar)
        * Auxiliary data (e.g. individual loss / regularization terms)
    """
    # Write params to model and data objects
    model = LearnedModel.write_params_to_model(params[0], base_model, needs_sim=False)
    pose_traj = {
        geom_name: measurements[geom_name]["position"]
        for geom_name in measurements.keys()
        if "position" in measurements[geom_name]
    } | {geom_name: params[1][geom_name]["position"] for geom_name in params[1].keys()}
    vel_traj = {
        geom_name: measurements[geom_name]["velocity"]
        for geom_name in measurements.keys()
        if "velocity" in measurements[geom_name]
    } | {geom_name: params[1][geom_name]["velocity"] for geom_name in params[1].keys()}
    # Forward will compute all physical parameters and distances
    mjx_data = jax.vmap(mjx.forward, in_axes=(None, 0))(
        model,
        jax.vmap(
            mjx_util.write_qvel_to_data,
            in_axes=(None, 0, {geom_name: 0 for geom_name in pose_traj.keys()}),
        )(
            model,
            jax.vmap(
                mjx_util.write_qpos_to_data,
                in_axes=(None, None, {geom_name: 0 for geom_name in pose_traj.keys()}),
            )(model, mjx.make_data(model), pose_traj),
            vel_traj,
        ),
    )
    ### Prediction: Velocity Term (q_v_pred)
    # Exclude robot predictions
    obj_qvel_idx = mjx_util.qvelidx_from_geom_names(model, list(params[1].keys()))
    delassus_objonly = (
        mjx_data.efc_J[1:, :, obj_qvel_idx]
        @ jnp.linalg.inv(mjx_data.qM[1:, obj_qvel_idx, :][..., obj_qvel_idx])
        @ jnp.moveaxis(mjx_data.efc_J[1:, :, obj_qvel_idx], -1, -2)
    )  # (n_t-1, n_l, n_l)
    non_contact_acceleration = (
        jnp.linalg.inv(mjx_data.qM)
        @ jnp.expand_dims(
            mjx_data.qfrc_passive
            + mjx_data.qfrc_actuator
            + mjx_data.qfrc_applied
            - mjx_data.qfrc_bias,
            axis=-1,
        )
    ).squeeze()[
        1:, :
    ]  # (n_t-1, n_v)
    delta_t = jnp.expand_dims(
        measurements["time"][1:] - measurements["time"][:-1], axis=-1
    )  # (n_t-1, 1)
    obj_delta_v = mjx_data.qvel[1:, obj_qvel_idx] - (
        mjx_data.qvel[:-1, obj_qvel_idx]
        + non_contact_acceleration[..., obj_qvel_idx] * delta_t
    )  # (n_t-1, n_v)

    qp_v_pred = delassus_objonly + hyperparams.epsilon * jnp.eye(
        delassus_objonly.shape[-1]
    )  # (n_t-1, n_l, n_l)
    q_v_pred = (
        -mjx_data.efc_J[1:, :, obj_qvel_idx] @ jnp.expand_dims(obj_delta_v, axis=-1)
    ).squeeze(
        axis=-1
    )  # (n_t-1, n_l)
    const_v_pred = (
        0.5
        * jnp.expand_dims(obj_delta_v, axis=-2)
        @ mjx_data.qM[1:, obj_qvel_idx, :][..., obj_qvel_idx]
        @ jnp.expand_dims(obj_delta_v, axis=-1)
    )  # (n_t-1, 1, 1)

    ### Complementarity (q_comp)
    # Note: efc pyramids are *always* contiguous and align with contact.dist
    n_contact = mjx_data.contact.dist.shape[-1]
    efc_to_normal = jax.scipy.linalg.block_diag(
        *([jnp.ones(4)] * n_contact)
    )  # Sum over pyramid to get normal, (n_c, n_l)
    q_comp = mjx_data.contact.dist[1:] @ efc_to_normal  # (n_t-1, n_l)

    ### Inelasticity (q_elas)
    normal_velocities = (
        efc_to_normal
        @ mjx_data.efc_J[1:, :, obj_qvel_idx]
        @ jnp.expand_dims(mjx_data.qvel[1:, obj_qvel_idx], axis=-1)
    )  # (n_t-1, n_c, 1)

    q_elas = (
        jnp.clip(normal_velocities.squeeze(axis=-1), a_min=0.0) @ efc_to_normal
    )  # (n_t-1, n_l)

    ### Max Power Dissipation (q_diss)
    # Assume friction is constant across n_t
    # See: https://mujoco.readthedocs.io/en/stable/_images/contact_frame.svg
    # Only m_1 and m_2 (no torsion), across all contacts
    efc_to_tangent = jax.scipy.linalg.block_diag(
        *(
            jnp.array(
                [
                    [
                        mjx_data.contact.friction[0, :, 0],
                        -mjx_data.contact.friction[0, :, 0],
                        jnp.zeros_like(mjx_data.contact.friction[0, :, 0]),
                        jnp.zeros_like(mjx_data.contact.friction[0, :, 0]),
                    ],
                    [
                        jnp.zeros_like(mjx_data.contact.friction[0, :, 0]),
                        jnp.zeros_like(mjx_data.contact.friction[0, :, 0]),
                        mjx_data.contact.friction[0, :, 1],
                        -mjx_data.contact.friction[0, :, 1],
                    ],
                ]
            ).transpose([2, 0, 1])
        )
    )  # (n_tan, n_l)
    sliding_velocities = (
        efc_to_tangent
        @ mjx_data.efc_J[1:, :, obj_qvel_idx]
        @ jnp.expand_dims(mjx_data.qvel[1:, obj_qvel_idx], axis=-1)
    ).squeeze(
        axis=-1
    )  # (n_t-1, n_tan)
    # Need non-0 norm for grad calculation, hence add eps to norm()
    sliding_speeds = jnp.linalg.norm(
        sliding_velocities.reshape(-1, n_contact, 2), axis=-1
    )  # (n_t-1, n_c)

    q_diss = jnp.hstack([sliding_speeds, sliding_velocities]) @ jnp.vstack(
        [efc_to_normal, efc_to_tangent]
    )  # (n_t-1, n_c + n_tan) * (n_c + n_tan, n_l) = (n_t-1, n_l)

    # Run QP optimization
    # Envelope theorem guarantees that gradient of loss w.r.t. parameters
    # can ignore the gradient of the impulses w.r.t. the QCQP parameters.
    qp_final = jax.lax.stop_gradient(hyperparams.w_v_pred * qp_v_pred)
    q_final = jax.lax.stop_gradient(
        hyperparams.w_v_pred * q_v_pred
        + hyperparams.w_comp * q_comp
        + hyperparams.w_diss * q_diss
        + hyperparams.w_elas * q_elas
    )
    impulses_raw = jax.vmap(
        lambda Q_final, q_final: (
            raPDHG()
            .optimize(
                create_qp(
                    Q=Q_final,
                    c=q_final.T,
                    A=jnp.zeros((0, q_final.shape[-1])),
                    b=jnp.zeros(0),
                    G=jnp.eye(q_final.shape[-1]),
                    h=jnp.zeros(q_final.shape[-1]),
                    l=jnp.zeros(q_final.shape[-1]),
                    u=jnp.full((q_final.shape[-1],), jnp.inf),
                ),
            )
            .primal_solution
        )
    )(qp_final, q_final)
    impulses = jnp.nan_to_num(jnp.clip(impulses_raw, a_min=0.0))  # (n_t-1, n_l)

    # Record contactnets loss terms
    loss = {}
    ### Loss: Prediction: Velocity (loss_v_pred)
    loss["v_pred"] = hyperparams.w_v_pred * (
        0.5 * impulses[..., jnp.newaxis, :] @ qp_v_pred @ impulses[..., jnp.newaxis]
        + impulses[..., jnp.newaxis, :] @ q_v_pred[..., jnp.newaxis]
        + const_v_pred
    ).squeeze(
        axis=[-1, -2]
    )  # (n_t-1,)

    ### Loss: Complementarity (loss_comp)
    loss["comp"] = hyperparams.w_comp * (
        impulses[..., jnp.newaxis, :] @ q_comp[..., jnp.newaxis]
    ).squeeze(
        axis=[-1, -2]
    )  # (n_t-1,)

    ### Loss: Max Power Dissipation (loss_diss)
    loss["diss"] = hyperparams.w_diss * (
        impulses[..., jnp.newaxis, :] @ q_diss[..., jnp.newaxis]
    ).squeeze(
        axis=[-1, -2]
    )  # (n_t-1,)

    ### Loss: Inelasticity (loss_elas)
    loss["diss"] = hyperparams.w_elas * (
        impulses[..., jnp.newaxis, :] @ q_elas[..., jnp.newaxis]
    ).squeeze(
        axis=[-1, -2]
    )  # (n_t-1,)

    ### Loss: Prediction: Position (loss_q_pred)
    # Want to access internal position euler integrator for MJX
    # pylint: disable=protected-access
    q_pred = jax.vmap(
        lambda qpos, qvel, delta_t, model=model: mjx._src.scan.flat(
            model,
            lambda *args, delta_t=delta_t: mjx._src.forward._integrate_pos(
                *args, dt=delta_t
            ),
            "jqv",
            "q",
            model.jnt_type,
            qpos,
            qvel,
        )
    )(mjx_data.qpos[:-1], mjx_data.qvel[1:], delta_t.squeeze(axis=-1))
    obj_qpos_idx = mjx_util.qposidx_from_geom_names(model, list(params[1].keys()))
    # TODO: have more general configuration difference rather than L2 norm
    loss["q_pred"] = hyperparams.w_q_pred * jnp.sum(
        (q_pred[..., obj_qpos_idx] - mjx_data.qpos[1:, obj_qpos_idx]) ** 2, axis=-1
    )

    ### Measurement Losses (Contact Bool / Normal)
    # And Regularization (Penetration / Grounded)
    loss_meas, outputs = _get_measurement_loss_and_outputs(
        model, mjx_data, measurements, params[1].keys(), hyperparams
    )

    # TODO: NO loss should be negative, make sure of that

    return (
        jax.tree.reduce(operator.add, jax.tree.map(jnp.sum, loss | loss_meas)),
        {
            "loss": loss | loss_meas,
            "outputs": outputs,
            "data": mjx_data,
        },
    )


@gin.configurable(allowlist=["hyperparams"])
def loss_diffsim(
    params: tuple[dict[str, dict[str, jax.Array]], dict[str, jax.Array]],
    measurements: dict[str, dict[str, jax.Array]],
    base_model: mjx.Model,
    hyperparams: LearningHyperparameters,
) -> tuple[float, Any]:
    """Diffsim loss function for training

    Parameters:
        * params: tuple of (model_params, traj_param)
        * * where traj_param is specifically q0 for each learnable geometry
        * measurements: contact and robot proprioception and control data,
        * * as a full trajectory (not a list of trajectories)
        * base_model: the mjx model in which to create data and write params

    Returns:
        * Loss (scalar)
        * Auxiliary data (e.g. individual loss / regularization terms)
    """
    # Write params to model and data objects
    model = LearnedModel.write_params_to_model(params[0], base_model, needs_sim=True)
    init_data = mjx_util.write_qpos_to_data(
        model,
        mjx.make_data(model),
        dict(
            {
                geom_name: measurements[geom_name]["position"][0, :]
                for geom_name in measurements.keys()
                if "position" in measurements[geom_name]
            },
            **(params[1]),
        ),
    )

    # Run diffsim
    data_sim = (
        mjx_util.diffsim_overwrite(
            model,
            init_data,
            measurements["ctrl"],
            {
                geom_name: measurements[geom_name]
                for geom_name in measurements.keys()
                if "position" in measurements[geom_name]
            },
            stacked=True,
        )
        if hyperparams.sim_overwrite
        else mjx_util.diffsim(
            model,
            init_data,
            measurements["ctrl"],
            stacked=True,
        )
    )
    # mjx.Data w/ leading T dimension

    # Compute outputs (phi, normal)
    loss, outputs = _get_measurement_loss_and_outputs(
        model, data_sim, measurements, params[1].keys(), hyperparams
    )

    return (
        jax.tree.reduce(operator.add, jax.tree.map(jnp.sum, loss)),
        {
            "loss": loss,
            "outputs": outputs,
            "data": data_sim,
        },
    )


# Lots of configuration (using gin) for training
@gin.configurable(allowlist=["loss_style", "optimizer_cls", "vis_update"])
def train_epochs(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    learned_model: LearnedModel,
    learned_traj: LearnedTrajectory,
    dataset: data_util.TrajectorySet,
    n_epochs: int,
    epoch_start: int = 0,
    gui_vis: Optional[MJXMeshcatVisualizer] = None,
    loss_style: LossStyle = LossStyle.DIFFSIM,
    optimizer_cls: optax.GradientTransformation = optax.adam,
    vis_update: int = 0,
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

    # Configure loss, params, and optimizer
    # loss_fn = jax.jit(jax.value_and_grad(loss_diffsim, has_aux=True))
    # Diffsim needs forward-mode (and it is faster with long graphs)
    # see https://github.com/google-deepmind/mujoco/issues/2259
    if loss_style == LossStyle.DIFFSIM:
        n_batch = 1
        loss_fn = jax.jit(
            jax.jacfwd(loss_diffsim, has_aux=True), static_argnames=["hyperparams"]
        )
    elif loss_style == LossStyle.VIMP:
        n_batch = len(dataset)
        loss_fn = jax.jit(
            jax.grad(loss_vimp, has_aux=True), static_argnames=["hyperparams"]
        )
    else:
        raise NotImplementedError(f"Loss Style {loss_style} not supported.")

    def get_learning_params(batch_idx):
        return (
            learned_model.params,
            (
                learned_traj.init_q
                if loss_style == LossStyle.DIFFSIM
                else learned_traj[batch_idx]
            ),
        )

    def get_data(batch_idx):
        return (
            dataset.full_trajectory()
            if loss_style == LossStyle.DIFFSIM
            else dataset[batch_idx]
        )

    def set_learning_params(params, batch_idx):
        if loss_style == LossStyle.DIFFSIM:
            learned_model.params, learned_traj.init_q = params
        else:
            learned_model.params, learned_traj[batch_idx] = params

    optimizer = optimizer_cls()
    opt_state = optimizer.init(get_learning_params(0))

    hyperparams = LearningHyperparameters()

    for epoch in range(epoch_start, epoch_start + n_epochs):
        start = time.time()

        aux_list = []

        for batch_idx in range(n_batch):
            # Compute Grad + auxiliary data
            grads, aux = loss_fn(
                get_learning_params(batch_idx),
                get_data(batch_idx),
                learned_model.base_model,
                hyperparams,
            )
            aux_list.append(aux)

            # Gradient Step
            updates, opt_state = optimizer.update(grads, opt_state)
            set_learning_params(
                optax.apply_updates(get_learning_params(batch_idx), updates), batch_idx
            )

        # Print data
        loss_total = jax.tree.reduce(
            operator.add, jax.tree.map(jnp.sum, [aux["loss"] for aux in aux_list])
        )
        print(f"{epoch:04d} ({time.time()-start:6.4f}s): Loss ({loss_total:6.4f})")

        # Visualization / File updates
        if vis_update > 0 and epoch % vis_update == 0:
            print("\t Writing to File...")
            learned_model.write_to_file(f"{epoch:04d}")
            learned_traj.write_to_file(f"{epoch:04d}")
            file_util.write_object(aux_list, "learning", f"aux_{epoch:04d}.pkl")
            if gui_vis is not None:
                print("\t Visualizing...")
                vis_data = None
                if loss_style == LossStyle.DIFFSIM:
                    vis_data = aux_list[0]["data"]
                elif loss_style == LossStyle.VIMP:
                    full_traj = learned_traj.get_full_trajectory()
                    pose_traj = {
                        geom_name: full_traj[geom_name]["position"]
                        for geom_name in full_traj.keys()
                    }
                    vel_traj = {
                        geom_name: full_traj[geom_name]["velocity"]
                        for geom_name in full_traj.keys()
                    }
                    vis_data = jax.vmap(
                        mjx_util.write_qvel_to_data,
                        in_axes=(
                            None,
                            0,
                            {geom_name: 0 for geom_name in pose_traj.keys()},
                        ),
                    )(
                        learned_model.active_model,
                        jax.vmap(
                            mjx_util.write_qpos_to_data,
                            in_axes=(
                                None,
                                None,
                                {geom_name: 0 for geom_name in pose_traj.keys()},
                            ),
                        )(
                            learned_model.active_model,
                            mjx.make_data(learned_model.active_model),
                            pose_traj,
                        ),
                        vel_traj,
                    )
                    # Make sure xpos and qpos are in sync
                    vis_data = jax.jit(jax.vmap(mjx.kinematics, in_axes=(None, 0)))(
                        learned_model.active_model, vis_data
                    )
                # TODO: Debug slow visualization code
                gui_vis.update_visuals(
                    model=learned_model.active_model,
                    data_trajectory=mjx_util.data_unstack(vis_data),
                )
    ## END training loop
