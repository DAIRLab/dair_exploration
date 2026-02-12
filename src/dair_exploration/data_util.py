#!/usr/bin/env python3

"""Utility functions for dataset management

The main contents of this file are as follows:

    * Class to hold, pad, and record trajectory data
"""

from typing import Any, Optional
from collections.abc import Sequence

import gin

import jax
import jax.numpy as jnp

from dair_exploration.file_util import write_object


@gin.configurable
class TrajectorySet(Sequence):
    """Holds collected trajectory data

    Pads trajectory to desired length, defaults to 0-padding.
    If a dict key is provided, pad with first value instead of 0s.
    """

    _trajectories: list[Any] = []
    r"""Underlying trajectory data"""
    _pad_to: int = 0
    r"""Minimum trajectory length / how long to pad trajectories"""
    _pad_repeat_keys: list[str] = []
    r"""Dict keys to pad with same value instead of 0"""

    def __init__(self, pad_to: int = 0, pad_repeat_keys: Optional[list[str]] = None):
        if pad_repeat_keys is not None:
            self._pad_repeat_keys = pad_repeat_keys
        self._pad_to = pad_to

    def add_trajectory(self, trajectory: Any) -> None:
        """Add a trajectory to the working set and save to file"""
        # Each data tree should have the same structure
        if len(self._trajectories) > 0:
            assert jax.tree.structure(self._trajectories[0]) == jax.tree.structure(
                trajectory
            )
        self._trajectories.append(trajectory)

        write_object(
            self._trajectories, "data", f"traj_{len(self._trajectories):02d}.pkl"
        )

    def __getitem__(self, idx: int) -> Any:
        """Return trajectory item (padded)"""

        def pad(path, leaf):
            pad_len = self._pad_to - len(leaf) if self._pad_to > 0 else 0
            do_repeat_first = any(
                jax.tree_util.DictKey(key) in path for key in self._pad_repeat_keys
            )
            repeat_val = (
                jnp.repeat(leaf[:1, ...], pad_len, axis=0)
                if do_repeat_first
                else jnp.zeros_like(jnp.repeat(leaf[:1, ...], pad_len, axis=0))
            )
            return jnp.insert(leaf, 0, repeat_val, axis=0)

        return jax.tree.map_with_path(pad, self._trajectories[idx])

    @property
    def trajectories(self) -> list[Any]:
        """Get list of trajectories as a property"""
        return self._trajectories

    def __len__(self):
        """Length == # of trajectories"""
        return len(self._trajectories)

    def full_trajectory(self) -> Any:
        """Get entire trajectory in a single data object, unpadded."""
        # Pylint doesn't understand the *args expansion includes "tree"
        # pylint: disable=no-value-for-parameter
        return jax.tree.map(
            lambda *xs: jnp.concatenate(xs, axis=0), *self._trajectories
        )
