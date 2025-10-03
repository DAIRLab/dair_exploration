#!/usr/bin/env python3

"""Utility functions for Working with the Trifinger

The main contents of this file are as follows:

        * Class to handle LCM communication with the robot
        * Functions and Classes to construct a workspace and take discrete actions within
"""

from dataclasses import dataclass, field
import time
from typing import Any, Optional

import gin
import jax.numpy as jnp
import lcm
import numpy as np
from scipy.spatial.transform import Rotation

from .lcmtypes.dairlib import (
    lcmt_fingertips_position,
    lcmt_object_state,
    lcmt_densetact_measurement_data,
    lcmt_fingertips_target_kinematics,
)


## LCM Service
@gin.configurable
class TrifingerLCMService:
    """
    Command robot and collect data over LCM
    """

    # LCM Elements
    _lcm: lcm.LCM
    _lcm_subs: dict[tuple[str, lcm.LCMSubscription]]
    _raw_data: dict[str, list[Any]]

    _fingertip_geom_names: list[str]
    _object_geom_name: str
    _traj_time_len: float

    def __init__(
        self,
        lcm_channels: dict[str, str],
        fingertip_geom_names: list[str],
        object_geom_name: Optional[str] = None,
        traj_time_len: float = 2.0,
    ):
        self._fingertip_geom_names = fingertip_geom_names
        self._object_geom_name = object_geom_name
        self._traj_time_len = traj_time_len
        self._raw_data = {"fingertip_pose": [], "densetact": [], "object_state": []}

        # Init LCM Subscriptions
        self._lcm = lcm.LCM()
        self._lcm_subs = {}
        self._lcm_subs["fingertips_position"] = (
            lcm_channels["fingertips_position"],
            self._lcm.subscribe(lcm_channels["fingertips_position"], self.sub_handler),
        )
        self._lcm_subs["densetact"] = (
            lcm_channels["densetact"],
            self._lcm.subscribe(lcm_channels["densetact"], self.sub_handler),
        )
        self._lcm_subs["object_state"] = (
            lcm_channels["object_state"],
            self._lcm.subscribe(lcm_channels["object_state"], self.sub_handler),
        )
        self._lcm_subs["fingertips_target"] = (lcm_channels["fingertips_target"], None)
        for sub in self._lcm_subs.values():
            if sub[1] is not None:
                sub[1].set_queue_capacity(
                    1
                )  # to discard everything outside of the handle window

    @property
    def fingertip_geom_names(self):
        """Fingertip body names"""
        return self._fingertip_geom_names

    @property
    def object_geom_name(self):
        """Fingertip body names"""
        return self._object_geom_name

    def sub_handler(self, channel: str, data: Any):
        """
        Write LCM incoming messages to cache
        """
        if channel == self._lcm_subs["fingertips_position"][0]:
            self._raw_data["fingertip_pose"].append(
                lcmt_fingertips_position.decode(data)
            )
        if channel == self._lcm_subs["densetact"][0]:
            self._raw_data["densetact"].append(
                lcmt_densetact_measurement_data.decode(data)
            )
        if channel == self._lcm_subs["object_state"][0]:
            self._raw_data["object_state"].append(lcmt_object_state.decode(data))

    def execute_trajectory(
        self,
        target_state: np.ndarray,
        pos_is_absolute: bool = True,
        no_data: bool = False,
    ) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
        """
        Direct the robot to go to target_state.
        Record all incoming data over the next traj_time_len seconds.
        NOTE: assumes that target_state is in order


        Params:
            target_state: (finger_0q, finger_120q, finger_240q, finger_0v, finger_120v, finger_240v)
            pos_is_absolute: relative or absolute (w.r.t. world frame) motion (default: True)
            no_data: Run command but do not collect data (default: False)

        Returns:
        timestamps size (dtime,)
        dict(fingertip/object_geom_names -> data_dict)
        data_dict has
            position (dtime, 3), velocity (dtime, 3),
            if available: contact_force_C (dtime, 3),
                contact_force_W (dtime, 3), contact_normal_W (dtime, 3)
        """
        # pylint: disable=too-many-locals, too-many-statements
        command = lcmt_fingertips_target_kinematics()
        assert target_state.shape == (len(command.targetPos) + len(command.targetVel),)
        command.utime = int(time.time() * 1e6)
        command.isAbsoluteTargetPos = pos_is_absolute
        command.targetPos[:] = target_state[: len(command.targetPos)]
        command.targetVel[:] = target_state[len(command.targetPos) :]

        # Start with clear data
        for data_list in self._raw_data.values():
            data_list.clear()

        print(f"Sending Command at: {time.time()}")
        self._lcm.publish(self._lcm_subs["fingertips_target"][0], command.encode())
        end_time = time.time() + self._traj_time_len
        while time.time() < end_time:
            self._lcm.handle_timeout(int((end_time - time.time()) * 1e3))
        print(f"Finished at: {time.time()}")
        print(
            f"Collected {[len(data_list) for data_list in self._raw_data.values()]}"
            + " samples."
        )

        # Return empty if not any force data
        ret = {body_name: {} for body_name in self._fingertip_geom_names}
        if no_data or len(self._raw_data["densetact"]) < 1:
            return ret

        assert self._raw_data["densetact"][0].numSensors == len(
            self._fingertip_geom_names
        )
        assert len(self._raw_data["fingertip_pose"]) >= len(self._raw_data["densetact"])

        def is_sorted(a: np.ndarray) -> bool:
            return np.all(a[:-1] <= a[1:])

        densetact_time_s = np.array(
            [
                float(measurement.sensorData[0].timestamp) / 1e6
                for measurement in self._raw_data["densetact"]
            ]
        ).flatten()
        assert is_sorted(densetact_time_s)
        fingerpos_time_s = np.array(
            [
                float(measurement.utime) / 1e6
                for measurement in self._raw_data["fingertip_pose"]
            ]
        ).flatten()
        assert is_sorted(fingerpos_time_s)
        object_time_s = np.array(
            [
                float(measurement.utime) / 1e6
                for measurement in self._raw_data["object_state"]
            ]
        ).flatten()
        assert is_sorted(object_time_s)

        # Interp fingertip data
        fingertip_pos_w = {}
        fingertip_vel_w = {}
        fingertip_force_c = {}
        fingertip_force_w = {}
        fingertip_normal_w = {}
        for body_idx, body_name in enumerate(self._fingertip_geom_names):
            # Position Interpolation
            body_pos = np.array(
                [
                    measurement.curPos[3 * body_idx : 3 * body_idx + 3]
                    for measurement in self._raw_data["fingertip_pose"]
                ]
            )
            assert body_pos.shape == (len(fingerpos_time_s), 3)
            body_pos_interp = np.vstack(
                [
                    np.interp(densetact_time_s, fingerpos_time_s, body_pos[:, idx])
                    for idx in range(3)
                ]
            ).T
            assert body_pos_interp.shape == (len(densetact_time_s), 3)
            fingertip_pos_w[body_name] = body_pos_interp

            # Velocity Interpolation
            body_vel = np.array(
                [
                    measurement.curVel[3 * body_idx : 3 * body_idx + 3]
                    for measurement in self._raw_data["fingertip_pose"]
                ]
            )
            assert body_vel.shape == (len(fingerpos_time_s), 3)
            body_vel_interp = np.vstack(
                [
                    np.interp(densetact_time_s, fingerpos_time_s, body_vel[:, idx])
                    for idx in range(3)
                ]
            ).T
            assert body_vel_interp.shape == (len(densetact_time_s), 3)
            fingertip_vel_w[body_name] = body_vel_interp

            # Quat Interpolation
            body_quat = np.array(
                [
                    measurement.curQuat[4 * body_idx : 4 * body_idx + 4]
                    for measurement in self._raw_data["fingertip_pose"]
                ]
            )
            assert body_quat.shape == (len(fingerpos_time_s), 4)
            body_quat_interp = np.vstack(
                [
                    np.interp(densetact_time_s, fingerpos_time_s, body_quat[:, idx])
                    for idx in range(4)
                ]
            ).T
            assert body_quat_interp.shape == (len(densetact_time_s), 4)
            body_r_bw = Rotation.from_quat(body_quat_interp, scalar_first=True)

            # Record normal and force in world frame
            body_r_cb = Rotation.from_matrix(
                np.stack(
                    [
                        np.array(measurement.sensorData[body_idx].contactFrame)[:3, :3]
                        for measurement in self._raw_data["densetact"]
                    ]
                )
            )
            normal_c = np.broadcast_to(
                np.array([0.0, 0.0, 1.0]), (len(densetact_time_s), 3)
            )
            body_r_cw = body_r_bw.inv() * body_r_cb
            fingertip_normal_w[body_name] = body_r_cw.apply(normal_c)
            # Zero out no contact normal
            finger_in_contact = np.array(
                [
                    measurement.sensorData[body_idx].inContact
                    for measurement in self._raw_data["densetact"]
                ]
            )
            fingertip_normal_w[body_name][~finger_in_contact] = 0.0
            force_c = np.array(
                [
                    (
                        list(measurement.sensorData[body_idx].scaledFriction)
                        + [measurement.sensorData[body_idx].scaledNormal]
                    )
                    for measurement in self._raw_data["densetact"]
                ]
            )
            assert force_c.shape == (len(densetact_time_s), 3)
            fingertip_force_w[body_name] = body_r_cw.apply(force_c)
            fingertip_force_c[body_name] = force_c

        for body_name in self._fingertip_geom_names:
            ret[body_name]["position"] = jnp.array(fingertip_pos_w[body_name])
            ret[body_name]["velocity"] = jnp.array(fingertip_vel_w[body_name])
            ret[body_name]["contact_force_C"] = jnp.array(fingertip_force_c[body_name])
            ret[body_name]["contact_force_W"] = jnp.array(fingertip_force_w[body_name])
            ret[body_name]["contact_normal_W"] = jnp.array(
                fingertip_normal_w[body_name]
            )

        # Interp ground-truth object data
        if len(self._raw_data["object_state"]) > 0:
            obj_name = (
                self._object_geom_name
                if self._object_geom_name is not None
                else self._raw_data["object_state"][0].object_name
            )
            ret[obj_name] = {}
            # Position Interpolation
            num_positions = self._raw_data["object_state"][0].num_positions
            object_pos = np.array(
                [
                    measurement.position[:]
                    for measurement in self._raw_data["object_state"]
                ]
            )
            assert object_pos.shape == (len(object_time_s), num_positions)
            object_pos_interp = np.vstack(
                [
                    np.interp(densetact_time_s, object_time_s, object_pos[:, idx])
                    for idx in range(num_positions)
                ]
            ).T
            assert object_pos_interp.shape == (len(densetact_time_s), num_positions)
            ret[obj_name]["position"] = jnp.array(object_pos_interp)

            # Velocity Interpolation
            num_velocities = self._raw_data["object_state"][0].num_velocities
            object_vel = np.array(
                [
                    measurement.velocity[:]
                    for measurement in self._raw_data["object_state"]
                ]
            )
            assert object_vel.shape == (len(object_time_s), num_velocities)
            object_vel_interp = np.vstack(
                [
                    np.interp(densetact_time_s, object_time_s, object_vel[:, idx])
                    for idx in range(num_velocities)
                ]
            ).T
            assert object_vel_interp.shape == (len(densetact_time_s), num_velocities)
            ret[obj_name]["velocity"] = jnp.array(object_vel_interp)

        # Return
        return jnp.array(densetact_time_s), ret


## Action / Workspace Parameters
@gin.configurable
@dataclass
class Action:
    """Action specification.

    Each action starts at the edge of the workspace at angle (polar, azimuth).
    Each action ends at the intersection of the workspace dome and the Y-Z plane.
    At location (radius where 1 = workspace_radius) and angle off +Y-axis (angle).
    """

    # pylint: disable=too-many-instance-attributes

    name: str = ""

    posx_start_polar: float = 0.0
    posx_start_azimuth: float = 0.0
    posx_end_radius: float = 1.0
    posx_end_angle: float = np.pi / 2

    negx_start_polar: float = 0.0
    negx_start_azimuth: float = 0.0
    negx_end_radius: float = 1.0
    negx_end_angle: float = np.pi / 2

    def __str__(self):
        return f"Action {self.name}"

    def __post_init__(self):
        """Method to check validity of parameters."""
        assert 0.0 <= self.posx_start_polar <= np.pi / 2
        assert -np.pi / 2 <= self.posx_start_azimuth <= np.pi / 2
        assert 0.0 <= self.posx_end_radius <= 1.0
        assert 0.0 <= self.posx_end_angle <= np.pi

        assert 0.0 <= self.negx_start_polar <= np.pi / 2
        assert -np.pi / 2 <= self.negx_start_azimuth <= np.pi / 2
        assert 0.0 <= self.negx_end_radius <= 1.0
        assert 0.0 <= self.negx_end_angle <= np.pi


@gin.configurable
@dataclass
class ActionLibraryParams:
    """Class to specify workspace setup and discrete action space"""

    # pylint: disable=too-many-instance-attributes
    workspace_xy_center: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    """2D Center of the workspace"""
    workspace_z_rot: float = 0.0  # rad
    r"""Rotation of workspace x/y plane about world z axis"""
    workspace_radius: float = 0.15  # m
    r"""Radius of workspace"""
    robot_radius: float = 0.01575  # m
    r"""Radius of robot spheres"""
    fixed_240_w: tuple[float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0)
    )  # m
    r"""Where to keep the trifinger's unused 240deg arm"""

    # Switches
    ground_buffer: bool = True
    r"""Truncate bottom of workspace at robot radius"""
    yplane_buffer: bool = False
    r"""Target separate planes for both robot fingers to guarantee no contact"""

    def __post_init__(self):
        """Method to check validity of parameters."""
        assert self.workspace_radius > 0.0
        assert 0.0 < self.robot_radius < self.workspace_radius
        assert len(self.workspace_xy_center) == 2
        assert len(self.fixed_240_w) == 3


@gin.configurable
def sample_action(
    params: ActionLibraryParams = ActionLibraryParams(),
    library: Optional[list[Action]] = None,
    index: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a straight line action
    Args:
    library: list of discrete possible actions
    """

    # Input Validation
    if index is not None:
        assert library is not None

    fixed_240_traj = np.array(params.fixed_240_w)
    rng = np.random.default_rng()

    # Start in workspace frame
    def sample_finger(
        flip_x: bool = False,
        start_polar: Optional[float] = None,
        start_azimuth: Optional[float] = None,
        end_radius: Optional[float] = None,
        end_angle: Optional[float] = None,
    ):
        flip_factor = -1.0 if flip_x else 1.0
        start_polar = (
            rng.uniform(0.0, np.pi / 2.0) if start_polar is None else start_polar
        )
        start_azimuth = (
            rng.uniform(-np.pi / 2.0, np.pi / 2.0)
            if start_azimuth is None
            else start_azimuth
        )
        start_s = (params.workspace_radius - params.robot_radius) * np.array(
            [
                (np.sin(start_polar) * np.cos(start_azimuth)),
                np.sin(start_polar) * np.sin(start_azimuth),
                np.cos(start_polar),
            ]
        )
        start_s[2] += params.robot_radius if params.ground_buffer else 0.0
        start_s[0] *= flip_factor
        max_radius = params.workspace_radius - params.robot_radius
        end_radius = (
            rng.uniform(0.0, max_radius)
            if end_radius is None
            else end_radius * max_radius
        )
        end_angle = rng.uniform(0.0, np.pi) if end_angle is None else end_angle
        end_s = np.array(
            [
                flip_factor * (params.robot_radius if params.yplane_buffer else 0.0),
                end_radius * np.cos(end_angle),
                (params.robot_radius if params.ground_buffer else 0.0)
                + end_radius * np.sin(end_angle),
            ]
        )
        return (start_s, end_s)

    if library is not None:
        lib_index = (
            (rng.integers(len(library)) % len(library))
            if index is None
            else (index % len(library))
        )
        if len(library[lib_index].name) > 0:
            print(f"Sampled Action: {library[lib_index].name}")
        finger_0_traj = sample_finger(
            False,
            start_polar=library[lib_index].posx_start_polar,
            start_azimuth=library[lib_index].posx_start_azimuth,
            end_radius=library[lib_index].posx_end_radius,
            end_angle=library[lib_index].posx_end_angle,
        )
        finger_120_traj = sample_finger(
            True,
            start_polar=library[lib_index].negx_start_polar,
            start_azimuth=library[lib_index].negx_start_azimuth,
            end_radius=library[lib_index].negx_end_radius,
            end_angle=library[lib_index].negx_end_angle,
        )
    else:
        finger_0_traj = sample_finger(False)
        finger_120_traj = sample_finger(True)

    ret = (np.zeros(18), np.zeros(18))
    z_rot = np.array(
        [
            [np.cos(params.workspace_z_rot), -np.sin(params.workspace_z_rot), 0.0],
            [np.sin(params.workspace_z_rot), np.cos(params.workspace_z_rot), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    xy_trans = np.array(
        [params.workspace_xy_center[0], params.workspace_xy_center[1], 0.0]
    )
    for idx in [0, 1]:
        ret[idx][:3] = z_rot @ finger_0_traj[idx].T + xy_trans
        ret[idx][3:6] = z_rot @ finger_120_traj[idx].T + xy_trans
        ret[idx][6:9] = fixed_240_traj[:]

    return ret
