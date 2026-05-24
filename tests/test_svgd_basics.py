#!/usr/bin/env python3

"""Test the basic functionality of the SVGD observed and expected info implementation."""

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

import numpy as np

import matplotlib.pyplot as plt


### Define Environment
@dataclass(frozen=True)
class TestSVGDHyperparameters:
    """Extra knobs for SVGD sampling + observed-info Monte Carlo size."""

    dyn_speed: float = 1.0
    dyn_var: float = 0.01
    dyn_penalty: float = 1000.0

    meas_normal_var: float = 0.1
    meas_phi_alpha: float = 7000.0


@partial(jax.jit, static_argnames=["hp"])
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


@partial(jax.jit, static_argnames=["hp"])
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
            to_center / jnp.maximum(jnp.linalg.norm(to_center), 1e-8),
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
        contact_term = (1.0 - contact_bool) * hp.meas_phi_alpha * jnp.square(
            phi
        ) - jax.nn.softplus(hp.meas_phi_alpha * jnp.square(phi))
        ret[sensor_name] = {"normal": normal_term, "contact": contact_term}
    return ret


def _create_env() -> dict[str, Any]:
    """Create a simple environment for testing."""
    params = {"radius": jnp.array(0.5)}
    x_learned = jnp.array([[0.0, 1.0], [0.0, 0.5]])
    measurements = {
        "bottom_left": {
            "position": jnp.array([-0.5, 0.5]),
            "contact_normal_W": jnp.array([1.0, 0.0]),
        }
    }
    return {"params": params, "x_learned": x_learned, "measurements": measurements}


def test_plot_distributions():
    """Test plotting the distributions."""
    hp = TestSVGDHyperparameters()
    env = _create_env()
    params = env["params"]
    x_learned = env["x_learned"]
    measurements = env["measurements"]

    # Create a grid of points to evaluate the logpdfs
    x_grid = jnp.linspace(-2.0, 2.0, 1000)
    y_grid = jnp.linspace(0.0, 2.0, 1000)
    xx, yy = jnp.meshgrid(x_grid, y_grid)
    grid_points = jnp.stack([xx.flatten(), yy.flatten()], axis=-1)

    # Evaluate logpdfs on the grid
    logpdf_dyn = jax.vmap(lambda x: logpdf_dynamics(params, x_learned[-1], x, hp))(
        grid_points
    )
    logpdf_meas = jax.vmap(lambda x: logpdf_measurement(params, x, measurements, hp))(
        grid_points
    )

    # Reshape for plotting
    logpdf_dyn = logpdf_dyn.reshape(xx.shape)
    logpdf_meas_normal = logpdf_meas["bottom_left"]["normal"].reshape(xx.shape)
    logpdf_meas_contact = logpdf_meas["bottom_left"]["contact"].reshape(xx.shape)

    # Plotting code
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.contourf(
        xx,
        yy,
        logpdf_dyn,
        levels=np.linspace(-100, 0.0, 50),
        extend="min",
        cmap="viridis",
    )
    plt.colorbar(label="Log PDF Dynamics")
    plt.scatter(x_learned[:, 0], x_learned[:, 1], color="red", label="Learned States")
    plt.title("Dynamics Log PDF")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.contourf(xx, yy, logpdf_meas_normal, levels=50, cmap="viridis")
    plt.colorbar(label="Log PDF Measurement")
    plt.scatter(x_learned[:, 0], x_learned[:, 1], color="red", label="Learned States")
    plt.title("Normal Measurement Log PDF")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.contourf(
        xx,
        yy,
        logpdf_meas_contact,
        levels=np.linspace(-100, 0.0, 50),
        vmin=-100,
        vmax=0,
        cmap="viridis",
        extend="min",
    )
    plt.colorbar(label="Log PDF Measurement")
    plt.scatter(x_learned[:, 0], x_learned[:, 1], color="red", label="Learned States")
    plt.title("Contact Measurement Log PDF")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_plot_distributions()
