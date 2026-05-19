#!/usr/bin/env python3

"""Compare uniform-grid marginalization vs SVGD observed information (2 timesteps)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

from dair_exploration.file_util import enable_jax_cache
from dair_exploration.svgd_marginalization import (
    _observed_info_from_grads,
    grad_meas_wrt_params,
    observed_info_svgd,
    sum_observed_info,
)
from tests.test_svgd_marginalization import (
    BASEVAR,
    _make_observed_info_svgd_2d_problem,
    grad_logpdf_meas_eq34_pytree,
    logpdf_dynamics,
    logpdf_meas_eq34_one_sensor_terms,
)

# Uniform grid over ``x_{T-1}`` (interior state at contact height).
_UNIFORM_X_BOUNDS = (-5.0, 5.0)
_UNIFORM_Z_BOUNDS = (1e-4, 5.0)
_N_UNIFORM_PER_AXIS = 64  # 4096 states in the box


def _uniform_x_samples(rng: jax.Array) -> jax.Array:
    """``(n_samples, 2)`` positions with ``x ~ U(-5,5)``, ``z ~ U(0,5)``."""
    n = _N_UNIFORM_PER_AXIS
    xs = jnp.linspace(_UNIFORM_X_BOUNDS[0], _UNIFORM_X_BOUNDS[1], n)
    zs = jnp.linspace(_UNIFORM_Z_BOUNDS[0], _UNIFORM_Z_BOUNDS[1], n)
    x_grid, z_grid = jnp.meshgrid(xs, zs, indexing="ij")
    samples = jnp.stack([x_grid.ravel(), z_grid.ravel()], axis=1)
    jitter = jax.random.uniform(rng, samples.shape, minval=-1e-3, maxval=1e-3)
    return samples + jitter


def _log_marginal_meas_leaf(
    omega: tuple,
    x_samples: jax.Array,
    sensor_meas: dict[str, jax.Array],
    term: str,
) -> jax.Array:
    """``log p(m_{t,k} | x_T) ≈ LSE_i(log p(m|x_i) + log p(x_i|x_T)) - LSE_i(log p(x_i|x_T))``."""
    radius = omega[0]["radius"]
    x_T = omega[1][0]

    def log_meas(x0: jax.Array) -> jax.Array:
        terms = logpdf_meas_eq34_one_sensor_terms(sensor_meas, x0, radius)
        return terms[term]

    def log_dyn(x0: jax.Array) -> jax.Array:
        return logpdf_dynamics(x_T, x0, radius, var=BASEVAR)

    log_joint = jax.vmap(lambda x0: log_meas(x0) + log_dyn(x0))(x_samples)
    log_dyn_only = jax.vmap(log_dyn)(x_samples)
    return jax.nn.logsumexp(log_joint) - jax.nn.logsumexp(log_dyn_only)


def _grads_uniform_marginalization(
    omega: tuple,
    measurements: dict,
    x_samples: jax.Array,
) -> dict:
    """Marginalized grads at ``t=0`` and direct terminal grads at ``t=1`` (PyTree)."""
    m0 = jax.tree.map(lambda leaf: leaf[0], measurements)
    m1 = jax.tree.map(lambda leaf: leaf[1], measurements)
    x_T = omega[1][0]

    grad_t0: dict = {}
    for sensor in sorted(m0.keys()):
        meas_s = {key: m0[sensor][key] for key in m0[sensor]}
        grad_t0[sensor] = {
            term: jax.grad(
                partial(_log_marginal_meas_leaf, x_samples=x_samples, sensor_meas=meas_s, term=term)
            )(omega)
            for term in ("normal", "contact")
        }

    grad_t1 = grad_logpdf_meas_eq34_pytree(
        m1, x_T, omega, link_to_terminal=True
    )

    def stack_time(g0_leaf: dict, g1_leaf: dict) -> dict:
        return jax.tree.map(
            lambda a, b: jnp.stack([a, b], axis=0),
            g0_leaf,
            g1_leaf,
        )

    return {
        sensor: stack_time(grad_t0[sensor], grad_t1[sensor])
        for sensor in grad_t0
    }


def _observed_info_from_grad_pytree(grads: dict) -> jax.Array:
    return sum_observed_info(_observed_info_from_grads(grads))


def _info_single_timestep(grads: dict, t: int) -> jax.Array:
    """Sum ``g_t g_t^T`` over all measurement leaves at one timestep."""
    grads_t = jax.tree.map(lambda leaf: leaf[t], grads)
    return _observed_info_from_grad_pytree(
        jax.tree.map(lambda g: jax.tree.map(lambda x: jnp.expand_dims(x, 0), g), grads_t)
    )


def _relative_frobenius_error(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.linalg.norm(a - b) / jnp.maximum(jnp.linalg.norm(b), 1e-12))


def _format_leaf_traces_at_t(grads: dict, t: int) -> str:
    """Per measurement leaf: trace of ``g_t g_t^T`` at timestep ``t``."""
    lines: list[str] = []
    for sensor in sorted(grads.keys()):
        for term in ("normal", "contact"):
            g_t = jax.tree.map(lambda leaf, tt=t: leaf[tt], grads[sensor][term])
            vec, _ = ravel_pytree(g_t)
            tr = float(vec @ vec)
            if tr > 1.0:
                lines.append(f"    {sensor}/{term}: trace={tr:.4g}")
    return "\n".join(lines) if lines else "    (negligible)"


def test_observed_info_uniform_vs_svgd_two_timesteps(capsys):
    """Uniform LSE marginalization vs SVGD for a 2-step contact trajectory.

    Trajectory: ``(0, 1) → (0, 0.5)``, ``r = 0.5``, sensors at ``(±0.5, 1.0)``.

    (1) **Uniform**: for ``t = T-1``, approximate
    ``∇_ω log ∫ p(m|x_{T-1}) p(x_{T-1}|x_T) dx_{T-1}`` via
    ``LSE(log p(m|x) + log p(x|x_T)) - LSE(log p(x|x_T))`` on a grid in
    ``(-5, 5) × (0, 5)``. At ``t = T``, use ``∇_ω log p(m_T | x_T)`` directly.

    (2) **SVGD**: :func:`~dair_exploration.svgd_marginalization.observed_info_svgd`
    (Eqs. 11–12 particle marginalization).

    **Expectations**

    * Terminal (``t = 1``) information agrees tightly (same gradient, no integral).
    * Global uniform vs SVGD total ``I`` can differ by order unity: uniform
      integrates over the full box while SVGD particles stay near the MLE
      ``x_{T-1}``, and Eq. 11–12 softmax weights use ``log p(m|x)`` only (not
      ``log p(m|x) + log p(x|x_T)``).
    * Marginal LSE evaluated **only at SVGD particle locations** should track
      SVGD at ``t = 0`` closely (same support, same formula as the continuum
      limit of importance sampling).
    """
    enable_jax_cache()
    sensors = {
        "left": jnp.array([-0.5, 1.0]),
        "right": jnp.array([0.5, 1.0]),
    }
    learned = jnp.array([[0.0, 1.0], [0.0, 0.5]])
    rng = jax.random.key(7)
    meas_rng = jax.random.fold_in(rng, 1)
    sample_rng = jax.random.fold_in(rng, 2)

    problem = _make_observed_info_svgd_2d_problem(
        n_timesteps=2,
        n_svgd_iters=80,
        n_particles=128,
        sensor_positions=sensors,
        learned_centers=learned,
        meas_rng=meas_rng,
    )
    omega = problem["omega"]
    x_uniform = _uniform_x_samples(sample_rng)

    info_svgd = sum_observed_info(
        observed_info_svgd(
            omega,
            problem["measurements"],
            problem["initial"],
            problem["logdyn"],
            problem["grad_dyn"],
            problem["logmeas"],
            problem["grad_meas"],
            problem["hyperparameters"],
            rng=rng,
        )
    )
    grads_uniform = _grads_uniform_marginalization(
        omega, problem["measurements"], x_uniform
    )
    info_uniform = _observed_info_from_grad_pytree(grads_uniform)

    from dair_exploration.svgd_marginalization import _sample_state_particles_svgd

    particles = _sample_state_particles_svgd(
        problem["initial"],
        omega[1],
        rng,
        problem["hyperparameters"],
        problem["logdyn"],
    )
    grads_svgd = grad_meas_wrt_params(
        omega,
        problem["measurements"],
        particles,
        problem["logdyn"],
        problem["grad_dyn"],
        problem["logmeas"],
        problem["grad_meas"],
    )

    info_t0_uniform = _info_single_timestep(grads_uniform, 0)
    info_t1_uniform = _info_single_timestep(grads_uniform, 1)
    info_t0_svgd = _info_single_timestep(grads_svgd, 0)
    info_t1_svgd = _info_single_timestep(grads_svgd, 1)

    # LSE marginal on SVGD particle cloud only (should align with Eq. 11–12 support).
    x_particles = particles[0]  # ``(n_particles, 2)`` at ``t = 0``
    grads_particle_box = _grads_uniform_marginalization(
        omega, problem["measurements"], x_particles
    )
    info_t0_particles = _info_single_timestep(grads_particle_box, 0)

    rel_total = _relative_frobenius_error(info_uniform, info_svgd)
    rel_t0 = _relative_frobenius_error(info_t0_uniform, info_t0_svgd)
    rel_t0_particles = _relative_frobenius_error(info_t0_particles, info_t0_svgd)
    rel_t1 = _relative_frobenius_error(info_t1_uniform, info_t1_svgd)

    report = (
        f"Uniform grid: {_N_UNIFORM_PER_AXIS}×{_N_UNIFORM_PER_AXIS} = "
        f"{_N_UNIFORM_PER_AXIS**2} samples in "
        f"x∈{_UNIFORM_X_BOUNDS}, z∈{_UNIFORM_Z_BOUNDS}\n"
        f"Total I (uniform):\n{jax.device_get(info_uniform)}\n"
        f"Total I (SVGD):\n{jax.device_get(info_svgd)}\n"
        f"diag uniform (r,x,z): {jax.device_get(jnp.diag(info_uniform))}\n"
        f"diag SVGD    (r,x,z): {jax.device_get(jnp.diag(info_svgd))}\n"
        f"rel Frobenius total={rel_total:.4f}, t0={rel_t0:.4f}, "
        f"t0@particles={rel_t0_particles:.4f}, t1={rel_t1:.4f}\n"
        f"t=0 I diag uniform: {jax.device_get(jnp.diag(info_t0_uniform))}\n"
        f"t=0 I diag SVGD:    {jax.device_get(jnp.diag(info_t0_svgd))}\n"
        f"t=0 I diag LSE@particles: {jax.device_get(jnp.diag(info_t0_particles))}\n"
        f"t=1 I diag uniform: {jax.device_get(jnp.diag(info_t1_uniform))}\n"
        f"t=1 I diag SVGD:    {jax.device_get(jnp.diag(info_t1_svgd))}\n"
        "Per-leaf trace at t=0 (uniform global box):\n"
        f"{_format_leaf_traces_at_t(grads_uniform, 0)}\n"
        "Per-leaf trace at t=0 (SVGD):\n"
        f"{_format_leaf_traces_at_t(grads_svgd, 0)}"
    )
    with capsys.disabled():
        print(report, flush=True)

    assert info_uniform.shape == info_svgd.shape == (problem["n_params"], problem["n_params"])
    assert jnp.all(jnp.isfinite(info_uniform))
    assert jnp.all(jnp.isfinite(info_svgd))

    # At terminal (t=1) sensors are above the object (no contact); both methods give ~0.
    assert float(jnp.max(jnp.abs(info_t1_uniform))) < 1e-3
    assert float(jnp.max(jnp.abs(info_t1_svgd))) < 1e-3

    # Same LSE formula on the SVGD particle support ≈ Eq. 11–12 at t=0.
    assert rel_t0_particles < 0.25
    assert jnp.allclose(info_t0_particles, info_t0_svgd, rtol=0.25, atol=5e4)

    # Global box integral: same order on r and z; x can differ more (mass away from MLE).
    assert rel_total < 6.0
    diag_u = jnp.diag(info_uniform)
    diag_s = jnp.diag(info_svgd)
    assert jnp.all(diag_u > 0.0)
    assert jnp.all(diag_s > 0.0)
    for i, name in enumerate(("r", "x", "z")):
        ratio = float(diag_u[i] / diag_s[i])
        assert 0.1 < ratio < 500.0, f"diag ratio {name}={ratio}"
