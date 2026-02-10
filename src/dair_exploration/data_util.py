#!/usr/bin/env python3

"""Utility functions for dataset management

The main contents of this file are as follows:

    * Class to hold, pad, and record trajectory data
"""


from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Optional, Callable

import gin
import numpy as np
from scipy.spatial.transform import Rotation

import jax
import jax.numpy as jnp

from dair_exploration.file_util import write_object

## Action / Workspace Parameters


@gin.configurable
@dataclass
class TrajectorySet:
    """Holds collected trajectory data"""

    _trajectories: list[Any] = None

    def add_trajectory(trajectory: jax.tree_util.PyTreeDef) -> None:
        """Add a trajectory to the working set and save to file"""
        # Each data tree should have the same structure
        if self._trajectories is None:
            self._trajectories = []
        else:
            assert (
                jax.tree_util.tree_flatten(self._trajectories[0])[1]
                == jax.tree_util.tree_flatten(trajectory)[1]
            )
        self._trajectories.append(trajectory)

        write_object(
            self._trajectories, "data", f"trajs_{len(self._trajectories):02d}.npz"
        )

    def full_trajectory_nested_key():
        pass
