#!/usr/bin/env python3

"""Test the basic functionality of the SVGD observed and expected info implementation."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


### Define Environment
@dataclass(frozen=True)
class TestSVGDHyperparameters:
    """Extra knobs for SVGD sampling + observed-info Monte Carlo size."""

    dyn_speed: float = 0.1
    dyn_var: float = 0.01
    dyn_penalty: float = 100.0

    meas_normal_var: float = 0.1
    meas_phi_alphpa: float = 7000.0


@jax.jit
def logpdf_dynamics(
    params: dict[str, jax.Array],
    x_next: jax.Array,
    x_curr: jax.Array,
    hp: TestSVGDHyperparameters,
) -> float:
    """Log probability density function of the dynamics.
    Use ContactNets, min_{lamb>0} (z_final - (z_curr - SPEED + lamb))^2 + lamb*xt
    Args:
        params: Parameters of the dynamics model.
        x_curr: Current state.
        x_next: Next state.
        hp: Hyperparameters
    Returns:
        Log probability density of the next state given the current state and parameters.
    """
    radius = params["radius"]
    z_curr = x_curr[..., 1]
    lamb = jnp.maximum(0.0, hp.dyn_speed - (z_curr - radius))
    z_pred = z_curr - hp.dyn_speed + lamb
    barrier = lamb * (x_next[..., 1] - radius)
    return -jnp.reciprocal(hp.dyn_var) * (
        0.5
        * (
            jnp.square(x_next[..., 1] - z_pred)
            + jnp.square(x_next[..., 0] - x_curr[..., 0])
        )
        + barrier
    ) - hp.dyn_penalty * (
        jnp.abs(jnp.minimum(z_curr - radius, 0.0))
        + jnp.abs(jnp.minimum(x_next[..., 1] - radius, 0.0))
    )


grad_logpdf_dynamics = jax.jit(jax.grad(logpdf_dynamics))
grad_logpdf_dynamics_with_xT = jax.jit(jax.grad(logpdf_dynamics, argnums=(0, 1)))


@jax.jit
def logpdf_measurement(
    params: dict[str, jax.Array],
    x_curr: jax.Array,
    measurements: dict[str, dict[str, jax.Array]],
    hp: TestSVGDHyperparameters,
) -> jax.Array:
    """Log probability density function of the measurements.

    Args:
        params: Parameters of the dynamics model.
        x_curr: Current state.
        measurements: Dictionary of measurements
            ("object_name" -> ["position", "contact_normal"] (or 0 if not in contact))
        hp: Hyperparameters
    Returns:
        Pytree of log probability densities of measurements given the current state and parameters.
    """
    ret = {}
    for sensor_name in measurements:
        sensor_pos = measurements[sensor_name]["position"]
        meas_normal = measurements[sensor_name]["contact_normal_W"]
        to_center = x_curr - sensor_pos
        phi = jnp.linalg.norm(to_center) - params["radius"]  # Signed distance function
        n_hat = jnp.where(
            phi <= 0.0,
            to_center / jnp.maximum(phi, 1e-8),
            jnp.zeros_like(to_center),
        )
        contact_bool = jnp.clip(jnp.round(jnp.linalg.norm(meas_normal)), 0.0, 1.0)
        normal_term = (
            -0.5
            * contact_bool
            * (
                jnp.reciprocal(hp.meas_normal_var) * (1.0 - jnp.dot(n_hat, meas_normal))
                + jnp.log(2.0 * jnp.pi * hp.meas_normal_var)
            )
        )
        contact_term = (contact_bool - 1.0) * hp.meas_phi_alpha * jnp.square(
            phi
        ) - jax.nn.softplus(hp.meas_phi_alpha * jnp.square(phi))
        ret[sensor_name] = {"normal": normal_term, "contact": contact_term}
    return ret
