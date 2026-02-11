#!/usr/bin/env python3

"""Utility functions for dataset management

The main contents of this file are as follows:

    * Class to hold, pad, and record trajectory data
"""

from typing import Any

import gin

import jax
import jax.numpy as jnp

from dair_exploration.file_util import write_object


@gin.configurable
class TrajectorySet:
    """Holds collected trajectory data"""

    _trajectories: list[Any] = None

    def add_trajectory(self, trajectory: Any) -> None:
        """Add a trajectory to the working set and save to file"""
        # Each data tree should have the same structure
        if self._trajectories is None:
            self._trajectories = []
        else:
            assert jax.tree.structure(self._trajectories[0]) == jax.tree.structure(
                trajectory
            )
        self._trajectories.append(trajectory)

        write_object(
            self._trajectories, "data", f"trajs_{len(self._trajectories):02d}.npz"
        )

    @property
    def trajectories(self) -> list[Any]:
        """Get list of trajectories as a property"""
        return self._trajectories

    def __len__(self):
        """Length == # of trajectories"""
        return len(self._trajectories)

    def full_trajectory(self) -> Any:
        """Get entire trajectory in a single data object"""
        # Pylint doesn't understand the *args expansion includes "tree"
        # pylint: disable=no-value-for-parameter
        return jax.tree.map(
            lambda *xs: jnp.concatenate(xs, axis=0), *self._trajectories
        )
