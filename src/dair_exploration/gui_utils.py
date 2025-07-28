#!/usr/bin/env python3

"""Utility functions for GUI visualizations (including Meshcat)

The main contents of this file are as follows:

        * A class to initatite a comparison visualization in Meshcat between the true and learned trajectory + geometry
"""

import time
from tkinter import Tk, Scale, DoubleVar
from typing import Optional

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from pydrake.geometry import StartMeshcat, Meshcat, Rgba
from pydrake.geometry import Box, Sphere
from jax.scipy.spatial.transform import Rotation


class MJXMeshcatVisualizer:
    """
    Meshcat Comparison Visualization
    """

    _meshcat: Meshcat
    _model: mjx.Model
    _data: list[mjx.Data]
    _trajs: Optional[dict[int, jax.Array]]

    # Visualization
    _root: Tk
    _scale: Scale
    _timestep: DoubleVar

    def __init__(self, model: mjx.Model, init_data: Optional[mjx.Data]) -> None:
        self._meshcat = StartMeshcat()
        self._model = model
        self._data = []
        self._trajs = None
        self.reinit_tk()
        self.update_visuals(
            model, [mjx.make_data(model)] if init_data is None else [init_data], None
        )

    def update_visuals(
        self,
        model: mjx.Model,
        data_trajectory: list[mjx.Data],
        traj_overwrite: Optional[dict[str, list[tuple[jax.Array, Rotation]]]] = None,
    ):
        """Update local parameters used in update()"""

        ### input validation
        assert len(data_trajectory) > 0
        for data in data_trajectory:
            assert data.xpos.shape == (model.nbody, 3)
        if traj_overwrite is not None:
            for key, val in traj_overwrite.items():
                assert (
                    model.names.find(key.encode()) in model.name_geomadr
                ), f"{key} not in model"
                assert len(val) == len(data_trajectory)
                for subval in val:
                    assert len(subval[0]) == 3  # 3D Position
                    # Rotation enforced by type check

        # Parameter Overwrite
        self._model = model
        self._data = data_trajectory
        self._trajs = traj_overwrite
        self._scale.configure(to=float(len(self._data) - 1))
        self.update()

    def reinit_tk(self, new_val=0.0) -> None:
        """Reset scale range to new value"""
        self._root = Tk()
        self._timestep = DoubleVar(value=new_val)
        self._scale = Scale(
            master=self._root,
            variable=self._timestep,
            digits=0,
            from_=0.0,
            to=new_val,
            length=900,
            orient="horizontal",
            resolution=1.0,
            command=self.update,
        )
        self._scale.pack()

    def update(self, _event=None) -> None:
        """
        Update visualization
        """

        # Get current timestep
        timestep = int(self._timestep.get())
        data = self._data[timestep]

        # Loop through all geometries
        for geomid, bodyid in enumerate(self._model.geom_bodyid):
            ### Get Shape
            shape = None
            if self._model.geom_type[geomid] == mujoco.mjtGeom.mjGEOM_BOX:
                # geom_size is half-lengths
                shape = Box(*((2.0 * self._model.geom_size[geomid]).tolist()))
            elif self._model.geom_type[geomid] == mujoco.mjtGeom.mjGEOM_SPHERE:
                # geom_size is radius
                shape = Sphere(float(self._model.geom_size[geomid][0]))
            # TODO: Add ConvexMesh visualization
            if shape is None:
                continue
            ### Other shape types not supported yet

            ### Set Object Geometry
            nameadr = self._model.name_geomadr[geomid]
            name = str(
                self._model.names[
                    nameadr : self._model.names.find(b"\x00", nameadr)
                ].decode()
            )
            rgba = Rgba(*(self._model.geom_rgba[geomid].tolist()))
            self._meshcat.SetObject("/" + name, shape, rgba)

            if self._trajs is not None and name in self._trajs:
                pos = self._trajs[name][timestep][0]
                rot = self._trajs[name][timestep][1].as_matrix()
            else:
                pos = data.xpos[bodyid]
                rot = data.xmat[bodyid]

            ### Set Object Transform
            transform = jnp.eye(4).at[:3, :3].set(rot).at[:3, 3].set(pos)
            self._meshcat.SetTransform("/" + name, transform)

    def sweep(self, dt=0.033) -> None:
        """
        Sweep through the entire trajectory
        """
        end = int(self._scale.config()["to"][-1])

        for timestep in range(end):
            self._timestep.set(timestep)
            self.update()
            time.sleep(dt)
