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
    meas_phi_prob1at0: float = 0.99
    meas_phi_prob1at_nominal: float = 0.05
    meas_phi_nominal: float = 0.005

    n_meas_samples: int = 10000
    n_svgd_samples: int = 100


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


def _pdf_nocontact(phi: jax.Array, hp: TestSVGDHyperparameters) -> jax.Array:
    """Probability density function of the contact boolean measurement when not in contact."""
    # See https://www.wolframalpha.com/input?i=solve+2*%28a*sigmoid%28x*c%5E2%29+-+%28a-0.5%29%29%3D1-b+for+x
    meas_phi_alpha = jnp.log(
        2 * hp.meas_phi_prob1at0 / hp.meas_phi_prob1at_nominal - 1
    ) / jnp.abs(hp.meas_phi_nominal)
    # hp.meas_phi_prob1at0 at phi=0 and hp.meas_phi_prob1at_nominal at phi=hp.meas_phi_nominal
    prob_0 = 2.0 * (
        hp.meas_phi_prob1at0 * jax.nn.sigmoid(meas_phi_alpha * jnp.abs(phi))
        - (hp.meas_phi_prob1at0 - 0.5)
    )
    return prob_0


def _pdf_nocontact_standard(phi: jax.Array, hp: TestSVGDHyperparameters) -> jax.Array:
    """Probability density function of the contact boolean measurement when not in contact."""
    meas_shift = jnp.log(-hp.meas_phi_prob1at0 / (hp.meas_phi_prob1at0 - 1.0))
    meas_alpha = (
        meas_shift
        - jnp.log(-hp.meas_phi_prob1at_nominal / (hp.meas_phi_prob1at_nominal - 1.0))
    ) / hp.meas_phi_nominal
    prob_0 = jax.nn.sigmoid(meas_alpha * phi - meas_shift)
    return prob_0


def _logpdf_contact(
    phi: jax.Array | float, contact_bool: jax.Array | float, hp: TestSVGDHyperparameters
) -> jax.Array:
    """Log probability density function of the contact boolean measurement."""
    prob_0 = _pdf_nocontact(phi, hp)
    return jnp.nan_to_num(contact_bool * jnp.log(1.0 - prob_0)) + jnp.nan_to_num(
        (1.0 - contact_bool) * jnp.log(prob_0)
    )


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
        contact_term = (1.0 - contact_bool) * hp.meas_phi_alpha * phi - jax.nn.softplus(
            hp.meas_phi_alpha * phi
        )
        ret[sensor_name] = {"normal": normal_term, "contact": contact_term}
    return ret


def _sample_measurements(
    params: dict[str, jax.Array],
    x_samples: jax.Array,
    measurements: dict[str, dict[str, jax.Array]],
    hp: TestSVGDHyperparameters,
    rng: jax.Array | None = None,
) -> dict[str, dict[str, jax.Array]]:
    """Sample measurements given a set of sampled state and the parameters."""
    if rng is None:
        rng = jax.random.key(0)

    n_x_samples = x_samples.shape[0]
    ret = {}
    for sensor_name in measurements:
        sensor_pos = measurements[sensor_name]["position"]
        to_center = x_samples - sensor_pos
        n_hat = jnp.mean(
            to_center
            / jnp.maximum(jnp.linalg.norm(to_center, axis=-1, keepdims=True), 1e-8),
            axis=0,
        )
        n_hat = n_hat / jnp.maximum(
            jnp.linalg.norm(n_hat), 1e-8
        )  # Normalize to unit vector
        normal_noise = jax.random.multivariate_normal(
            rng,
            mean=n_hat,
            cov=hp.meas_normal_var / n_x_samples * jnp.eye(n_hat.size),
            shape=(hp.n_meas_samples,),
        )
        normal_noise = normal_noise / jnp.maximum(
            jnp.linalg.norm(normal_noise, axis=-1, keepdims=True), 1e-8
        )  # Normalize noise to unit vectors
        phi = jnp.linalg.norm(to_center) - params["radius"]  # Signed distance function
        prob_0 = jax.nn.sigmoid(hp.meas_phi_alpha * jnp.square(phi))
        contact_bool = jax.random.uniform(rng, shape=(hp.n_meas_samples,)) > prob_0
        normal_sample = contact_bool[..., None] * normal_noise
        ret[sensor_name] = {
            "position": jnp.broadcast_to(sensor_pos, normal_sample.shape),
            "contact_normal_W": normal_sample,
        }
    return ret


def _create_env() -> dict[str, Any]:
    """Create a simple environment for testing."""
    params = {"radius": jnp.array(0.5)}
    x_learned = jnp.array([[0.0, 1.0], [0.0, 0.5]])
    measurements = {
        "bottom_left": {
            "position": jnp.array([[-0.5, 0.5], [-0.5, 0.5]]),
            "contact_normal_W": jnp.array([[0.0, 0.0], [1.0, 0.0]]),
        },
        "bottom_right": {
            "position": jnp.array([[0.5, 0.5], [0.5, 0.5]]),
            "contact_normal_W": jnp.array([[0.0, 0.0], [-1.0, 0.0]]),
        },
        "top_left": {
            "position": jnp.array([[-0.5, 1.0], [-0.5, 1.0]]),
            "contact_normal_W": jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        },
        "top_right": {
            "position": jnp.array([[0.5, 1.0], [0.5, 1.0]]),
            "contact_normal_W": jnp.array([[-1.0, 0.0], [0.0, 0.0]]),
        },
    }
    return {"params": params, "x_learned": x_learned, "measurements": measurements}


def test_expected_info_final():
    """Test the expected information computation at the final learned state."""
    hp = TestSVGDHyperparameters()
    env = _create_env()
    params = env["params"]
    x_final = env["x_learned"][-1]
    meas_final = jax.tree.map(lambda leaf: leaf[-1], env["measurements"])
    params_packed = jnp.array([params["radius"], x_final[0], x_final[1]])

    measurements = _sample_measurements(params, x_final.reshape(1, -1), meas_final, hp)

    def logmeas_packed(
        omega: jax.Array, meas: dict[str, dict[str, jax.Array]]
    ) -> jax.Array:
        return logpdf_measurement({"radius": omega[0]}, omega[1:], meas, hp)

    hessian_expinfo = jax.tree.map(
        lambda leaf: -1.0 * leaf, jax.hessian(logmeas_packed)(params_packed, meas_final)
    )
    # c

    breakpoint()


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
    m_final = jax.tree.map(lambda leaf: leaf[-1], measurements)

    # Evaluate logpdfs on the grid
    logpdf_dyn = jax.vmap(lambda x: logpdf_dynamics(params, x_learned[-1], x, hp))(
        grid_points
    )
    logpdf_meas = jax.vmap(lambda x: logpdf_measurement(params, x, m_final, hp))(
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


def test_logpdf_contact():
    """Test the logpdf of the contact measurement."""
    hp = TestSVGDHyperparameters()
    assert jnp.isclose(
        1.0 - _pdf_nocontact(0.0, hp), hp.meas_phi_prob1at0
    ), "PDF at phi=0 should match meas_phi_prob1at0"
    assert jnp.isclose(
        1.0 - _pdf_nocontact(hp.meas_phi_nominal, hp), hp.meas_phi_prob1at_nominal
    ), "PDF at phi=meas_phi_nominal should match meas_phi_prob1at_nominal"
    phi_values = jnp.linspace(-0.01, 0.01, 100)
    pdf_nocontact = jax.vmap(lambda phi: _pdf_nocontact(phi, hp))(phi_values)
    pdf_contact = 1.0 - pdf_nocontact
    logpdf_contact = jax.vmap(lambda phi: _logpdf_contact(phi, 1.0, hp))(phi_values)
    logpdf_nocontact = jax.vmap(lambda phi: _logpdf_contact(phi, 0.0, hp))(phi_values)

    # Test grads and hessians
    grad_logpdf_contact = jax.vmap(lambda phi: jax.grad(_logpdf_contact)(phi, 1.0, hp))(
        phi_values
    )
    hess_logpdf_contact = jax.vmap(
        lambda phi: jax.hessian(_logpdf_contact)(phi, 1.0, hp)
    )(phi_values)
    grad_logpdf_nocontact = jax.vmap(
        lambda phi: jax.grad(_logpdf_contact)(phi, 0.0, hp)
    )(phi_values)
    hess_logpdf_nocontact = jax.vmap(
        lambda phi: jax.hessian(_logpdf_contact)(phi, 0.0, hp)
    )(phi_values)

    # Mean grad should be 0
    mean_grad = (
        grad_logpdf_contact * pdf_contact + grad_logpdf_nocontact * pdf_nocontact
    )
    assert jnp.all(
        jnp.isclose(mean_grad, 0.0, atol=1e-4)
    ), "Mean gradient should be close to 0"

    # Mean Hessian should be close to mean grad squared (Fisher information)
    mean_hess = (
        hess_logpdf_contact * pdf_contact + hess_logpdf_nocontact * pdf_nocontact
    )
    mean_grad_squared = (
        grad_logpdf_contact**2 * pdf_contact + grad_logpdf_nocontact**2 * pdf_nocontact
    )
    assert jnp.all(
        jnp.isclose(-mean_hess, mean_grad_squared)
    ), "Mean Hessian should be close to mean grad squared (Fisher information)"

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(phi_values, logpdf_contact, label="Contact")
    plt.plot(phi_values, logpdf_nocontact, label="No Contact")
    plt.title("Log PDF of Contact Measurement")
    plt.xlabel("Phi")
    plt.ylabel("Log PDF")
    plt.grid()
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(phi_values, pdf_contact, label="Contact")
    plt.plot(phi_values, pdf_nocontact, label="No Contact")
    plt.title("PDF of Contact Measurement")
    plt.xlabel("Phi")
    plt.ylabel("PDF")
    plt.grid()
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(phi_values, mean_grad_squared, label="Fisher Info")
    plt.title("Fisher Info")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    test_logpdf_contact()
