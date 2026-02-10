#!/usr/bin/env python3

"""Utility functions for GUI visualizations (including Meshcat)"""

import time
from tkinter import Tk, Scale, DoubleVar
from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
import mujoco
from mujoco import mjx
import meshcat
from meshcat import geometry
import numpy as np
import trimesh


class MJXMeshcatVisualizer:
    """
    Meshcat Comparison Visualization
    """

    _meshcat: meshcat.Visualizer
    _model: mjx.Model
    _data: list[mjx.Data]
    _trajs: Optional[dict[int, jax.Array]]

    # Visualization
    _root: Tk
    _scale: Scale
    _timestep: DoubleVar

    def __init__(self, model: mjx.Model, init_data: Optional[mjx.Data]) -> None:
        self._meshcat = meshcat.Visualizer().open()
        print(f"Started Meshcat server at: {self._meshcat.url()}")
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
                shape = geometry.Box((2.0 * self._model.geom_size[geomid]).tolist())
            elif self._model.geom_type[geomid] == mujoco.mjtGeom.mjGEOM_SPHERE:
                # geom_size is radius
                shape = geometry.Sphere(float(self._model.geom_size[geomid][0]))
            elif self._model.geom_type[geomid] == mujoco.mjtGeom.mjGEOM_MESH:
                # Get convex hull using trimesh
                trimesh_mesh = trimesh.Trimesh(
                    vertices=np.asarray(
                        Rotation.from_quat(jnp.roll(self._model.mesh_quat, -1)).apply(
                            # Mujoco wants us to use _impl
                            # pylint: disable-next=protected-access
                            self._model._impl.mesh_convex[0].vert
                        )
                        + self._model.mesh_pos
                    )
                ).convex_hull
                shape = geometry.TriangularMeshGeometry(
                    vertices=trimesh_mesh.vertices, faces=trimesh_mesh.faces
                )
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
            rgba = self._model.geom_rgba[geomid].tolist()
            color = int(
                f"0x{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}",
                16,
            )
            opacity = rgba[3]
            self._meshcat[name].set_object(
                shape, geometry.MeshLambertMaterial(color=color, opacity=opacity)
            )

            if self._trajs is not None and name in self._trajs:
                pos = self._trajs[name][timestep][0]
                rot = self._trajs[name][timestep][1].as_matrix()
            else:
                pos = data.xpos[bodyid]
                rot = data.xmat[bodyid]

            ### Set Object Transform
            transform = jnp.eye(4).at[:3, :3].set(rot).at[:3, 3].set(pos)
            self._meshcat[name].set_transform(np.asarray(transform).astype(np.float64))

    def sweep(self, dt=0.033) -> None:
        """
        Sweep through the entire trajectory
        """
        end = int(self._scale.config()["to"][-1])

        for timestep in range(end):
            start = time.perf_counter()
            self._timestep.set(timestep)
            self.update()
            while time.perf_counter() - start < dt:
                pass

    def draw_action_samples(self, action_knots: np.ndarray):
        """Draw lines representing actions"""

        self.clear_action_samples()

        n_knots = action_knots.shape[0]
        assert action_knots.shape == (n_knots, 2, 18)

        # TODO: make gin-config param
        fingertip_body_names = ["finger_0", "finger_1"]

        for knot_idx in range(n_knots):
            for finger_idx, finger_name in enumerate(fingertip_body_names):
                start_loc = action_knots[
                    knot_idx, 0, (finger_idx * 3) : ((finger_idx + 1) * 3)
                ]
                end_loc = action_knots[
                    knot_idx, 1, (finger_idx * 3) : ((finger_idx + 1) * 3)
                ]
                vertices = np.stack([start_loc, end_loc], axis=1)
                assert vertices.shape == (3, 2), str(vertices.shape)
                draw_geom = geometry.LineSegments(
                    geometry.PointsGeometry(position=vertices.astype(np.float32)),
                    geometry.LineBasicMaterial(
                        color=(0x19E6E6 if finger_idx == 0 else 0xE619E6)
                    ),
                )
                self._meshcat["actions"][f"{knot_idx}"][f"{finger_name}"].set_object(
                    draw_geom
                )

    def clear_action_samples(self):
        """Clear lines from action samples"""
        self._meshcat["actions"].delete()
