#!/usr/bin/env python3

"""Test the basic functionality of the SVGD observed and expected info implementation."""

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy

from dair_exploration.svgd_marginalization import (
    grad_dyn_wrt_params,
    sample_state_particles_svgd,
    SVGDHyperparameters,
)


### Define Environment
@dataclass(frozen=True)
class TestSVGDHyperparameters:
    """Extra knobs for SVGD sampling + observed-info Monte Carlo size."""

    dyn_speed: float = 1.0
    dyn_var: float = 1e-4
    dyn_penalty: float = 1000.0

    meas_normal_kappa: float = 15.0  # von Mises ~= Gaussian with variance 1/kappa
    meas_phi_prob1at0: float = 0.8
    meas_phi_prob1at_nominal: float = 0.05
    meas_phi_nominal: float = 0.05

    n_meas_samples: int = 10000


@partial(jax.jit, static_argnames=["hp"])
def logpdf_dynamics(
    params: dict[str, jax.Array],
    x_next: jax.Array,
    x_curr: jax.Array,
    hp: TestSVGDHyperparameters,
) -> jax.Array:
    """Log probability density function of the (backwards) dynamics p(x_curr|x_next, params).
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

@partial(jax.jit, static_argnames=["hp"])
def logpdf_prior(
    omega: tuple[Any, Any],
    hp: TestSVGDHyperparameters,
) -> jax.Array:
    """Log prior probability on the params.
    Args:
        params: Parameters of the dynamics model.
        hp: Hyperparameters
    Returns:
        Log probability density of the parameters.
    """
    params = omega[0]
    radius = params["radius"]
    x_final = omega[1]
    return -hp.dyn_penalty * (
        jnp.abs(jnp.minimum(x_final[..., 1] - radius, 0.0))
    )


grad_logpdf_dynamics = jax.jit(jax.grad(logpdf_dynamics))
grad_logpdf_dynamics_with_xT = jax.jit(jax.grad(logpdf_dynamics, argnums=(0, 1)))


@partial(jax.jit, static_argnames=["hp"])
def _pdf_nocontact(phi: jax.Array, hp: TestSVGDHyperparameters) -> jax.Array:
    """Probability density function of the contact boolean measurement when not in contact."""
    # https://www.wolframalpha.com/input?i=solve+2*%28a*sigmoid%28x*c%5E2%29+-+%28a-0.5%29%29%3D1-b+for+x
    meas_phi_alpha = jnp.log(
        2 * hp.meas_phi_prob1at0 / hp.meas_phi_prob1at_nominal - 1
    ) / jnp.abs(hp.meas_phi_nominal)
    # hp.meas_phi_prob1at0 at phi=0 and hp.meas_phi_prob1at_nominal at phi=hp.meas_phi_nominal
    prob_0 = 2.0 * (
        hp.meas_phi_prob1at0 * jax.nn.sigmoid(meas_phi_alpha * jnp.abs(phi))
        - (hp.meas_phi_prob1at0 - 0.5)
    )
    return prob_0


@partial(jax.jit, static_argnames=["hp"])
def _logpdf_contact(
    phi: jax.Array, contact_bool: jax.Array, hp: TestSVGDHyperparameters
) -> jax.Array:
    """Log probability density function of the contact boolean measurement."""
    prob_0 = _pdf_nocontact(phi, hp)
    return jnp.nan_to_num(contact_bool * jnp.log(1.0 - prob_0)) + jnp.nan_to_num(
        (1.0 - contact_bool) * jnp.log(prob_0)
    )


@partial(jax.jit, static_argnames=["hp"])
def _logpdf_normal(
    n_hat: jax.Array, meas_normal: jax.Array, hp: TestSVGDHyperparameters
) -> jax.Array:
    """Log probability density function of the normal measurement (von Mises-Fisher)."""
    n_hat_normal = n_hat / jnp.maximum(
        jnp.linalg.norm(n_hat, axis=-1, keepdims=True), 1e-8
    )
    meas_normal = meas_normal / jnp.maximum(
        jnp.linalg.norm(meas_normal, axis=-1, keepdims=True), 1e-8
    )
    ndim = n_hat_normal.shape[-1]
    assert ndim == meas_normal.shape[-1], "Incompatible dimensions"
    cosn = (
        jnp.clip(n_hat_normal[..., None, :] @ meas_normal[..., None], -1.0, 1.0)
        .squeeze(-1)
        .squeeze(-1)
    )
    return (
        hp.meas_normal_kappa * cosn
        + (ndim / 2.0 - 1) * jnp.log(hp.meas_normal_kappa)
        - jnp.log(
            (2.0 * jnp.pi) ** (ndim / 2.0)
            * scipy.special.iv(ndim / 2.0 - 1, hp.meas_normal_kappa)
        )
    )


@partial(jax.jit, static_argnames=["hp"])
def p_nocontact(
    params: dict[str, jax.Array],
    x_curr: jax.Array,
    measurements: dict[str, dict[str, jax.Array]],
    hp: TestSVGDHyperparameters,
) -> dict[str, jax.Array]:
    """Probability of no contact."""
    # Placeholder for the actual implementation
    ret = {}
    for sensor_name in measurements:
        sensor_pos = measurements[sensor_name]["position"]
        to_center = x_curr - sensor_pos
        phi = jnp.linalg.norm(to_center) - params["radius"]  # Signed distance function
        ret[sensor_name] = _pdf_nocontact(phi, hp)
    return ret


@partial(jax.jit, static_argnames=["hp"])
def logpdf_measurement(
    params: dict[str, jax.Array],
    x_curr: jax.Array,
    measurements: dict[str, dict[str, jax.Array]],
    hp: TestSVGDHyperparameters,
) -> Any:
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
        n_hat = to_center / jnp.maximum(jnp.linalg.norm(to_center), 1e-8)
        contact_bool = jnp.clip(jnp.round(jnp.linalg.norm(meas_normal)), 0.0, 1.0)
        contact_term = _logpdf_contact(phi, contact_bool, hp)
        normal_term = contact_bool * _logpdf_normal(n_hat, meas_normal, hp)
        ret[sensor_name] = contact_term + normal_term
    return ret


def _sample_normals(
    rng: jax.Array,
    x_curr: jax.Array,
    measurements: dict[str, dict[str, jax.Array]],
    hp: TestSVGDHyperparameters,
) -> dict[str, dict[str, jax.Array]]:
    """Sample normals from the measurement distribution.

    Args:
        rng: jax PRNG key
        x_curr: Current state, shape (..., n_state)
        measurements: Dictionary of measurements
            ("object_name" -> ["position", "contact_normal"] (or 0 if not in contact))
        hp: Hyperparameters

    Returns:
        Dictionary of sampled normals for each sensor.
    """
    ret = {}
    for sensor_name in measurements:
        sensor_pos = measurements[sensor_name]["position"]
        to_center = x_curr - sensor_pos
        n_hat = to_center / jnp.maximum(jnp.linalg.norm(to_center), 1e-8)
        ret[sensor_name] = {
            "position": jnp.broadcast_to(sensor_pos, x_curr.shape),
            "contact_normal_W": jax.random.vonmises_fisher(
                rng,
                n_hat,
                hp.meas_normal_kappa,
            ),
        }
    return ret


def _create_env_ground() -> dict[str, Any]:
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

def _create_env_air() -> dict[str, Any]:
    """Create a simple environment for testing."""
    params = {"radius": jnp.array(0.5)}
    x_learned = jnp.array([[0.0, 2.0], [0.0, 1.0]])
    measurements = {
        "bottom_left": {
            "position": jnp.array([[-0.5, 1.0], [-0.5, 1.0]]),
            "contact_normal_W": jnp.array([[0.0, 0.0], [1.0, 0.0]]),
        },
        "bottom_right": {
            "position": jnp.array([[0.5, 1.0], [0.5, 1.0]]),
            "contact_normal_W": jnp.array([[0.0, 0.0], [-1.0, 0.0]]),
        },
        "top_left": {
            "position": jnp.array([[-0.5, 2.0], [-0.5, 2.0]]),
            "contact_normal_W": jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        },
        "top_right": {
            "position": jnp.array([[0.5, 2.0], [0.5, 2.0]]),
            "contact_normal_W": jnp.array([[-1.0, 0.0], [0.0, 0.0]]),
        },
    }
    return {"params": params, "x_learned": x_learned, "measurements": measurements}


def test_expected_info_final():
    """Test the expected information computation at the final learned state."""
    hp = TestSVGDHyperparameters()
    env = _create_env_air()
    params = env["params"]
    x_final = env["x_learned"][-1]
    meas_final = jax.tree.map(lambda leaf: leaf[-1], env["measurements"])
    params_packed = jnp.array([params["radius"], x_final[0], x_final[1]])

    def logpdf_meas_fn(params_packed, meas):
        return logpdf_measurement(
            {"radius": params_packed[0]}, params_packed[1:], meas, hp
        )["bottom_left"]

    meas_samples = _sample_normals(jax.random.key(0),
        x_final[None, ...].repeat(hp.n_meas_samples, axis=0), meas_final, hp
    )
    zero_meas = jax.tree.map(jnp.zeros_like, meas_final)
    for sensor_name in meas_final:
        zero_meas[sensor_name]["position"] = meas_final[sensor_name]["position"]

    # Compute expected value:
    #               (p_nocontact) * value_at_0 + (1-p_nocontact) * expected_value_under_vmf
    # NOTE: don't forget to square *then* take the mean
    print("Calculating expected gradients and Hessians...", flush=True)
    grad_logpdf = jax.grad(logpdf_meas_fn)
    vmap_grad_logpdf = jax.vmap(grad_logpdf, in_axes=(None, 0))
    hess_logpdf = jax.hessian(logpdf_meas_fn)
    vmap_hess_logpdf = jax.vmap(hess_logpdf, in_axes=(None, 0))
    prob_nocontact = p_nocontact(params, x_final, meas_final, hp)["bottom_left"]
    grad_zero = grad_logpdf(params_packed, zero_meas)
    grad_one = vmap_grad_logpdf(params_packed, meas_samples)
    expected_grad = grad_zero * prob_nocontact + jnp.mean(grad_one, axis=0) * (
        1 - prob_nocontact
    )
    expected_neg_hess = -1.0 * (
        hess_logpdf(params_packed, zero_meas) * prob_nocontact
        + jnp.mean(vmap_hess_logpdf(params_packed, meas_samples), axis=0)
        * (1 - prob_nocontact)
    )
    expected_grad_sq = (
        prob_nocontact * jnp.outer(grad_zero, grad_zero)
        + (1 - prob_nocontact) * (grad_one.T @ grad_one) / hp.n_meas_samples
    )

    print("Expected gradient (should be near-0):", expected_grad)
    print(
        "Expected negative Hessian (know r-x a lot and z a little):\n",
        expected_neg_hess,
    )
    print(
        "Expected squared gradient(know r-x a lot and z a little):\n", expected_grad_sq
    )
    print("Asserting crude equality...", end="")
    assert jnp.allclose(
        expected_grad, jnp.zeros_like(expected_grad), atol=1e-2, rtol=1e-2
    )
    assert jnp.allclose(expected_neg_hess, expected_grad_sq, atol=1e0, rtol=1e0)
    print("Success!")

    print("Calculating expected gradients and Hessians...", flush=True)
    meas_random = jax.tree.map(lambda leaf: leaf[1], meas_samples)
    grad_obs = grad_logpdf(params_packed, meas_random)
    hess_obs = hess_logpdf(params_packed, meas_random)
    print("Observed gradient:", grad_obs)
    print("Observed gradient squared:\n", jnp.outer(grad_obs, grad_obs))
    print("Observed negative Hessian:\n", -hess_obs)

    # NOTE: observed info has high variance, especially for the gradient squared


def test_gradients():
    """Test grad log p(m_t|x_T) against fininte differences"""

    env_ground = _create_env_ground()
    

def test_expected_info_tminus1():
    """Test the expected information from the measurements at time t-1 (top left)"""
    rng_normal, rng_rand_idx, rng_sample_normals = jax.random.split(jax.random.key(1), 3)
    hp = TestSVGDHyperparameters()
    env = _create_env_air()
    params = env["params"]
    x_learned = env["x_learned"]
    omega = (params, x_learned[-1:])
    x_grid = jnp.linspace(-2.0, 2.0, 1000)
    y_grid = jnp.linspace(jnp.min(x_learned[..., 1]) - params["radius"], jnp.max(x_learned[..., 1]) + params["radius"], 1000)
    xx, yy = jnp.meshgrid(x_grid, y_grid)
    grid_points = jnp.stack([xx.flatten(), yy.flatten()], axis=-1)

    # Evaluate logpdfs on the grid
    logpdf_dyn = jax.vmap(lambda x: logpdf_dynamics(params, x_learned[-1], x, hp))(
        grid_points
    )
    logpdf_dyn = logpdf_dyn.reshape(xx.shape)

    # SVGD sample x's at time t-1
    def _logdyn(pair, *, omega, hp):
        x_t, x_tp1 = pair
        return logpdf_dynamics(omega[0], x_tp1, x_t, hp)

    _logdyn_partial = partial(_logdyn, omega=omega, hp=hp)

    def _grad_dyn(pair, *, omega, hp, link_to_terminal=False):
        x_t, x_tp1 = pair
        link = jnp.asarray(link_to_terminal)

        def loss(omega):
            x_next = jax.lax.select(link, omega[1][0], x_tp1)
            return logpdf_dynamics(omega[0], x_next, x_t, hp)

        return jax.grad(loss)(omega)

    _grad_dyn_partial = partial(_grad_dyn, omega=omega, hp=hp)

    def _logmeas(m_t, x_t, *, omega, hp):
        return logpdf_measurement(omega[0], x_t, m_t, hp)

    _logmeas_partial = partial(_logmeas, omega=omega, hp=hp)

    def _grad_logmeas(m_t, x_t, *, omega, hp, link_to_terminal=False):
        link = jnp.asarray(link_to_terminal)

        def log_all(omega: Any) -> dict[str, dict[str, jax.Array]]:
            object_x = jax.lax.select(link, omega[1][0], x_t)
            return logpdf_measurement(omega[0], object_x, m_t, hp)

        return jax.jacfwd(log_all)(omega)

    _grad_logmeas_partial = partial(_grad_logmeas, omega=omega, hp=hp)

    svgd_hp = SVGDHyperparameters(
        n_particles=1000, n_svgd_iters=2000, svgd_step=2e-4, init_sample_std=5e-1
    )

    x_particles = sample_state_particles_svgd(
        env["x_learned"][:-1],
        env["x_learned"][-1:],
        rng_normal,
        svgd_hp,
        _logdyn_partial,
    )

    plt.figure()
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
    plt.scatter(
        x_particles[0, :, 0], x_particles[0, :, 1], color="blue", label="SVGD Particles"
    )
    for idx in range(len(x_learned)):
        circle = patches.Circle(
            (x_learned[idx, 0], x_learned[idx, 1]),
            params["radius"],
            color="red",
            fill=False,
        )
        plt.gca().add_patch(circle)
    plt.title("Dynamics Log PDF (Unnormalized)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    ## TODO: Compute the measurement likelihood gradients for t=T-1
    measurements = env["measurements"]
    zero_meas = jax.tree.map(jnp.zeros_like, measurements)
    for sensor_name, sensor_measurement in measurements.items():
        zero_meas[sensor_name]["position"] = sensor_measurement["position"]
    g_dyn_per_timestep = grad_dyn_wrt_params(
        omega,
        x_particles,
        _logdyn_partial,
        _grad_dyn_partial,
    )
    g_dyn_tminus1_packed = jnp.concatenate(
        jax.tree.leaves(
            jax.tree.map(
                lambda leaf: leaf[-2].reshape(svgd_hp.n_particles, -1),
                g_dyn_per_timestep,
            )
        ),
        axis=-1,
    )

    # Compute the gradient of the measurement likelihood with respect to the final state
    def _grad_meas_wrt_xfinal(
        m_t: Any, x_particles_t: jax.Array, g_dyn_t_packed: jax.Array
    ) -> Any:
        logmeas_t = jax.vmap(_logmeas_partial, in_axes=(None, 0))(m_t, x_particles_t)
        gradmeas_t = jax.vmap(_grad_logmeas_partial, in_axes=(None, 0))(
            m_t, x_particles_t
        )
        gradmeas_t_packed = jax.tree.map(
            lambda leaf: jnp.concatenate(
                jax.tree.leaves(
                    jax.tree.map(
                        lambda subleaf: subleaf.reshape(svgd_hp.n_particles, -1), leaf
                    )
                ),
                axis=-1,
            ),
            gradmeas_t,
            is_leaf=lambda leaf: jax.tree.structure(leaf) == jax.tree.structure(omega),
        )
        return {
            sensor_name: (
                jax.nn.softmax(logmeas_t[sensor_name])[None, ...]
                @ (gradmeas_t_packed[sensor_name] + g_dyn_t_packed)
            ).squeeze(-2)
            - jnp.mean(g_dyn_t_packed, axis=0)
            for sensor_name in logmeas_t.keys()
        }
        

    ## Collect zero-measurement gradient at time T-1
    zero_meas_tminus1 = jax.tree.map(lambda leaf: leaf[0], zero_meas)
    grad_measzero_wrt_xfinal = _grad_meas_wrt_xfinal(
        zero_meas_tminus1, x_particles[0], g_dyn_tminus1_packed
    )

    ## Sample normals from the vmf-mixture distribution
    rand_idx = jax.random.randint(rng_rand_idx, (hp.n_meas_samples,), 0, x_particles[0].shape[0])
    meas_samples = _sample_normals(rng_sample_normals, x_particles[0, rand_idx, :], zero_meas_tminus1, hp)

    ## Collect one-measurement gradients at time T-1
    grad_meassamples_wrt_xfinal = jax.vmap(
        _grad_meas_wrt_xfinal, in_axes=(0, None, None)
    )(meas_samples, x_particles[0], g_dyn_tminus1_packed)
    grad_measone_wrt_xfinal = jax.tree.map(
        lambda leaf: jnp.mean(leaf, axis=0), grad_meassamples_wrt_xfinal
    )

    prob_nocontact = jax.vmap(p_nocontact, in_axes=(None, 0, None, None))(
        params, x_particles[0], zero_meas_tminus1, hp
    )

    expected_grad_wrt_xfinal = jax.tree.map(
        lambda pnc, gzero, gone: jnp.mean(pnc) * gzero + (1.0 - jnp.mean(pnc)) * gone,
        prob_nocontact,
        grad_measzero_wrt_xfinal,
        grad_measone_wrt_xfinal,
    )

    gradsq_measzero_wrt_xfinal = jax.tree.map(
        lambda leaf: jnp.outer(leaf, leaf), grad_measzero_wrt_xfinal
    )
    gradsq_measone_wrt_xfinal = jax.tree.map(
        lambda leaf: (leaf.T @ leaf) / hp.n_meas_samples, grad_meassamples_wrt_xfinal
    )

    expected_gradsq_wrt_xfinal = jax.tree.map(
        lambda pnc, gsqzero, gsqone: jnp.mean(pnc) * gsqzero
        + (1.0 - jnp.mean(pnc)) * gsqone,
        prob_nocontact,
        gradsq_measzero_wrt_xfinal,
        gradsq_measone_wrt_xfinal,
    )
    print("Expected gradient squared wrt x_final (top_left):", expected_gradsq_wrt_xfinal["top_left"])

    ## Let's say we observe only top-contact at T-1.
    obs_gradsq_wrt_xfinal = {
        "top_left": gradsq_measone_wrt_xfinal["top_left"],
        "top_right": gradsq_measone_wrt_xfinal["top_right"],
        "bottom_left": gradsq_measzero_wrt_xfinal["bottom_left"],
        "bottom_right": gradsq_measzero_wrt_xfinal["bottom_right"],
    }

    # Compute log measurement likelihoods with fit
    print("Computing log measurement likelihoods with fit...", flush=True)
    x_final = x_learned[-1:]
    x_vals = jnp.linspace(-0.1, 0.1, 21)
    out_top_lefts = []
    for x_val in x_vals:
        print("x_val:", x_val)
        x_final_val = x_final.at[0, 0].set(x_val)
        x_part_val = sample_state_particles_svgd(
            env["x_learned"][:-1],
            x_final_val,
            rng_normal,
            svgd_hp,
            _logdyn_partial,
        )
        logmeaszero_wrt_final = jax.tree.map(jax.nn.logsumexp, jax.vmap(_logmeas_partial, in_axes=(None, 0))(zero_meas_tminus1, x_part_val[0]))
        logmeasone_wrt_final = jax.tree.map(lambda leaf: jnp.mean(jax.nn.logsumexp(leaf, axis=-1), axis=0), jax.vmap(jax.vmap(_logmeas_partial, in_axes=(None, 0)), in_axes=(0, None))(meas_samples, x_part_val[0]))

        expected_logmeas_wrt_final = jax.tree.map(
            lambda pnc, gzero, gone: jnp.nan_to_num(jnp.mean(pnc) * gzero + (1.0 - jnp.mean(pnc)) * gone),
            prob_nocontact,
            logmeaszero_wrt_final,
            logmeasone_wrt_final,
        )
        out_top_lefts.append(expected_logmeas_wrt_final["top_left"])
        print("expected_logmeas_wrt_final:", expected_logmeas_wrt_final)
    print("out_top_lefts:", out_top_lefts)
    exp_neg_hessian_x = -2.0 * jnp.polyfit(x_vals, jnp.stack(out_top_lefts), deg=2)[0]
    print("Empirical expected negative Hessian wrt x (top_left):", exp_neg_hessian_x)
    breakpoint()


def test_plot_distributions():
    """Test plotting the distributions."""
    hp = TestSVGDHyperparameters()
    env = _create_env_ground()
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
    logpdf_meas_bl = logpdf_meas["bottom_left"].reshape(xx.shape)

    # Plotting code
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(
        xx,
        yy,
        logpdf_dyn,
        levels=np.linspace(-100, 0.0, 50),
        extend="min",
        cmap="viridis",
    )
    plt.colorbar(label="Log PDF Dynamics")
    plt.scatter(
        measurements["bottom_left"]["position"][..., 0],
        measurements["bottom_left"]["position"][..., 1],
        color="blue",
        label="Sensors",
    )
    plt.scatter(x_learned[:, 0], x_learned[:, 1], color="red", label="Learned States")
    for idx in range(len(x_learned)):
        circle = patches.Circle(
            (x_learned[idx, 0], x_learned[idx, 1]),
            params["radius"],
            color="red",
            fill=False,
        )
        plt.gca().add_patch(circle)
    plt.title("Dynamics Log PDF (Unnormalized)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.contourf(
        xx,
        yy,
        logpdf_meas_bl,
        levels=np.linspace(-100, 0.0, 50),
        extend="min",
        cmap="viridis",
    )
    plt.colorbar(label="Log PDF Measurement")
    plt.scatter(
        measurements["bottom_left"]["position"][..., 0],
        measurements["bottom_left"]["position"][..., 1],
        color="blue",
        label="Sensors",
    )
    plt.scatter(x_learned[:, 0], x_learned[:, 1], color="red", label="Learned States")
    for idx in range(len(x_learned)):
        circle = patches.Circle(
            (x_learned[idx, 0], x_learned[idx, 1]),
            params["radius"],
            color="red",
            fill=False,
        )
        plt.gca().add_patch(circle)
    plt.title("Measurement Log PDF")
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
    phi_values = jnp.linspace(-0.01, 0.01, 101)
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
    # test_plot_distributions()
    # test_expected_info_final()
    test_expected_info_tminus1()
