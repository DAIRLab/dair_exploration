#!/usr/bin/env python3

"""Utility functions for JAX"""

from typing import Any, Union

import jax


def overwrite_keep_gradient(old_value: Any, new_value: Any):
    """Return new_value but keep gradient flow from old_value"""

    assert jax.tree.structure(old_value) == jax.tree.structure(new_value)

    def overwrite(old_val: Union[jax.Array, float], new_val: Union[jax.Array, float]):
        return old_val + jax.lax.stop_gradient(new_val - old_val)

    return jax.tree.map(overwrite, old_value, new_value)
