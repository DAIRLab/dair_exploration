#!/usr/bin/env python3

"""Tests for SVGD marginalization utilities."""

from __future__ import annotations

import math
import os
import time
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import pytest

from dair_exploration.file_util import enable_jax_cache
from dair_exploration.svgd_marginalization import (
    SVGDHyperparameters,
    _dynamics_grad_wrt_params_one_particle,
    _meas_grad_wrt_params_one_timestep,
    _tree_broadcast_params_zeros,
    grad_meas_wrt_params,
    observed_info_svgd,
    svgd_step,
)

# -----------------------------------------------------------------------------
# Copied from tests/test_theory_v3.py — 2D contact-style dynamics conditional.
# -----------------------------------------------------------------------------
SPEED = 1.0
BASEVAR = 0.001
PENALTY = 1000.0

# Small step for 2D dynamics SVGD (large steps blow up the particle drive / kernel term).
_DYNAMICS_SVGD_STEP = 1e-3

# Measurement model (``docs/sv_observed_info.pdf`` Eqs. 3–4), aligned with ``learning.py``.
PHI_NOMINAL = 0.002
PHI_CI = 0.05
MEAS_ALPHA = math.log((1.0 / PHI_CI) - 1.0) / PHI_NOMINAL
MEAS_SIGMA_N = 0.01519224261
_LOG_2PI_SIGMA_N_SQ = math.log(2.0 * math.pi * MEAS_SIGMA_N**2)

# Fixed planar sensor(s): world-frame position per name.
DEFAULT_SENSOR_POSITIONS: dict[str, jax.Array] = {
    "finger": jnp.array([0.35, 1.75]),
}


def logpdf_dynamics(
    x_next: jax.Array,
    x_curr: jax.Array,
    radius: jax.Array,
    var: float = BASEVAR,
) -> jax.Array:
    """log p(x_curr | x_next, θ): ContactNets-style ground contact at z = radius."""
    r = radius.reshape(())
    zt = x_curr[1]
    lamb = jnp.maximum(0.0, SPEED - (zt - r))
    z_pred = zt - SPEED + lamb
    barrier = lamb * (x_next[1] - r)
    return (
        -jnp.reciprocal(var)
        * (
            0.5 * jnp.square(x_next[1] - z_pred)
            + barrier
            + jnp.square(x_next[0] - x_curr[0])
        )
        - PENALTY * jnp.abs(jnp.minimum(zt - r, 0.0))
    )


def signed_distance_phi(
    sensor_position: jax.Array,
    object_center: jax.Array,
    radius: jax.Array,
) -> jax.Array:
    """``phi = ||sensor - center|| - radius`` (positive when the sensor clears the object)."""
    r = radius.reshape(())
    return jnp.linalg.norm(sensor_position - object_center) - r


def predicted_contact_normal_W(
    sensor_position: jax.Array,
    object_center: jax.Array,
    radius: jax.Array,
) -> jax.Array:
    """Unit vector from sensor toward object center, or ``0`` if ``phi > 0``."""
    phi = signed_distance_phi(sensor_position, object_center, radius)
    to_center = object_center - sensor_position
    dist = jnp.linalg.norm(to_center)
    return jnp.where(
        phi <= 0.0,
        to_center / jnp.maximum(dist, 1e-8),
        jnp.zeros_like(to_center),
    )


def logpdf_meas_eq34(
    measurements_t: dict[str, dict[str, jax.Array]],
    object_center: jax.Array,
    radius: jax.Array,
    *,
    sigma_n: float = MEAS_SIGMA_N,
    alpha: float = MEAS_ALPHA,
) -> jax.Array:
    """``log p(m_t | x_t)`` from Eqs. 3–4 summed over robot sensors."""
    inv_sigma_sq = jnp.reciprocal(sigma_n**2)
    log_p = jnp.array(0.0)
    for sensor_name, meas in measurements_t.items():
        sensor_pos = meas["position"]
        meas_normal = meas["contact_normal_W"]
        phi = signed_distance_phi(sensor_pos, object_center, radius)
        n_hat = predicted_contact_normal_W(sensor_pos, object_center, radius)
        contact_bool = jnp.clip(jnp.round(jnp.linalg.norm(meas_normal)), 0.0, 1.0)
        phi_pos = jnp.maximum(phi, 0.0)
        normal_term = (
            -0.5
            * contact_bool
            * inv_sigma_sq
            * (1.0 - jnp.dot(n_hat, meas_normal))
            - 0.5 * _LOG_2PI_SIGMA_N_SQ
        )
        contact_term = (contact_bool - 1.0) * alpha * phi_pos + jax.nn.softplus(
            alpha * phi_pos
        )
        log_p = log_p + normal_term + contact_term
    return log_p


def make_sensor_measurements_trajectory(
    object_centers: jax.Array,
    radius: jax.Array,
    sensor_positions: dict[str, jax.Array] | None = None,
) -> dict[str, dict[str, jax.Array]]:
    """``{name: {position, contact_normal_W}}`` with leaves shaped ``(n_timesteps, n_dim)``."""
    sensors = DEFAULT_SENSOR_POSITIONS if sensor_positions is None else sensor_positions
    n_t = object_centers.shape[0]
    out: dict[str, dict[str, jax.Array]] = {}
    for name, sensor_pos in sensors.items():
        normals = jax.vmap(
            lambda center: predicted_contact_normal_W(sensor_pos, center, radius)
        )(object_centers)
        out[name] = {
            "position": jnp.broadcast_to(sensor_pos, (n_t, sensor_pos.shape[0])),
            "contact_normal_W": normals,
        }
    return out


def _make_dynamics_f_and_gradf(radius: jax.Array):
    """``f(x_curr, x_next) = log p(x_curr | x_next)`` and ``∇_{x_curr} f`` (fixed radius)."""
    logp = partial(logpdf_dynamics, radius=radius)

    def f(x_curr: jax.Array, x_next: jax.Array) -> jax.Array:
        return logp(x_next, x_curr)

    def gradf(x_curr: jax.Array, x_next: jax.Array) -> jax.Array:
        return jax.grad(lambda xc: logp(x_next, xc))(x_curr)

    return f, gradf


def _one_svgd_outer_iteration(
    particles: jax.Array,
    learned_traj: jax.Array,
    x_terminal: jax.Array,
    n_t: int,
    f_log: Callable[..., jax.Array],
    grad_log: Callable[..., jax.Array],
    hp: SVGDHyperparameters,
) -> jax.Array:
    """One backward SVGD sweep (same as one ``it`` in ``sample_dynamics_particles_svgd``)."""
    ni = particles.shape[1]
    new_parts = [particles[n_t - 1]]
    for t in range(n_t - 2, -1, -1):
        xt_tree = {"pos": particles[t]}
        xtp1_tree = {"pos": new_parts[-1]}
        updated = svgd_step(xt_tree, xtp1_tree, f_log, grad_log, hp)
        new_parts.append(updated["pos"])
    rev = list(reversed(new_parts))
    stacked = jnp.stack(rev, axis=0)
    return stacked.at[n_t - 1].set(jnp.broadcast_to(x_terminal, (ni, 2)))


def sample_dynamics_particles_svgd(
    learned_traj: jax.Array,
    x_terminal: jax.Array,
    radius: jax.Array,
    rng: jax.Array,
    hp: SVGDHyperparameters,
    *,
    n_particles: int,
    visualize: bool = False,
    visualize_pause_s: float = 0.05,
) -> jax.Array:
    """SVGD particles (nT, Ni, 2) using ``svgd_step``; last timestep rows equal ``x_terminal``."""
    n_t, _ = learned_traj.shape
    ni = n_particles
    rngs = jax.random.split(rng, n_t)
    particles = jnp.zeros((n_t, ni, 2))
    for t in range(n_t - 1):
        noise = jax.random.normal(rngs[t], (ni, 2)) * hp.init_sample_std
        particles = particles.at[t].set(learned_traj[t] + noise)
    particles = particles.at[n_t - 1].set(jnp.broadcast_to(x_terminal, (ni, 2)))

    f_log, grad_log = _make_dynamics_f_and_gradf(radius)

    if visualize:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]

        cmap = plt.get_cmap("viridis", n_t)
        plt.ion()

    for it in range(hp.n_svgd_iters):
        if visualize:
            parts_np = jax.device_get(particles)
            learned_np = jax.device_get(learned_traj)
            xT_np = jax.device_get(x_terminal)

            plt.clf()
            for t in range(n_t):
                xy = parts_np[t]
                plt.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=12,
                    alpha=0.65,
                    color=cmap(t),
                    edgecolors="none",
                )
            plt.plot(learned_np[:, 0], learned_np[:, 1], "k--", linewidth=1.0, alpha=0.7)
            plt.scatter([xT_np[0]], [xT_np[1]], c="k", s=30, marker="x")
            plt.title(
                f"SVGD (marginalization) particles by timestep — "
                f"iter {it + 1}/{hp.n_svgd_iters}"
            )
            plt.xlabel("x")
            plt.ylabel("z")
            plt.axis("equal")
            plt.grid(True, alpha=0.2)
            plt.pause(visualize_pause_s)

        particles = _one_svgd_outer_iteration(
            particles, learned_traj, x_terminal, n_t, f_log, grad_log, hp
        )

    if visualize:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]

        plt.ioff()

    return particles


def test_dynamics_grad_one_particle_matches_eq16():
    """Eq. 16 for a single ``(t, i)`` against an explicit softmax sum."""
    ni = 8
    params = {"radius": jnp.array([[0.5]])}
    state = jax.random.normal(jax.random.key(9), (2, ni, 2)) * 0.05
    g_next = _tree_broadcast_params_zeros(params, ni)
    g_next = jax.tree.map(
        lambda leaf: jax.random.normal(jax.random.key(10), leaf.shape) * 0.01,
        g_next,
    )

    def logdyn(pair: tuple[jax.Array, jax.Array]) -> jax.Array:
        return logpdf_dynamics(pair[1], pair[0], params["radius"])

    def grad_dyn(pair: tuple[jax.Array, jax.Array]) -> dict[str, jax.Array]:
        return jax.grad(lambda p: logpdf_dynamics(pair[1], pair[0], p["radius"]))(params)

    xt_i = state[0, 0]
    xt_plus = state[1]
    got = _dynamics_grad_wrt_params_one_particle(
        xt_i, xt_plus, g_next, logdyn, grad_dyn, ni
    )

    f_logits = jnp.array([logdyn((xt_i, xt_plus[j])) for j in range(ni)])
    w = jax.nn.softmax(f_logits)
    term_radii = jnp.stack(
        [
            grad_dyn((xt_i, xt_plus[j]))["radius"] + g_next["radius"][j]
            for j in range(ni)
        ],
        axis=0,
    )
    weighted = jnp.sum(w[:, None, None] * term_radii, axis=0)
    mean_next = jnp.mean(g_next["radius"], axis=0)
    expected = {"radius": weighted - mean_next}
    assert jax.tree.all(
        jax.tree.map(
            lambda a, b: jnp.allclose(a, b, rtol=1e-6, atol=1e-7), got, expected
        )
    )


def test_meas_grad_one_timestep_matches_eq11_12():
    """Eqs. 11–12 for one timestep against an explicit softmax sum."""
    ni = 8
    radius = jnp.array([[0.5]])
    params = {"radius": radius}
    state = jax.random.normal(jax.random.key(11), (1, ni, 2)) * 0.05
    ref_center = jnp.array([0.0, 1.0])
    measurements = make_sensor_measurements_trajectory(
        ref_center[None, :], radius
    )
    g_dyn = _tree_broadcast_params_zeros(params, ni)
    g_dyn = jax.tree.map(
        lambda leaf: jax.random.normal(jax.random.key(13), leaf.shape) * 0.01,
        g_dyn,
    )

    def logmeas(pair: tuple[dict, jax.Array]) -> jax.Array:
        m_t, x_t = pair
        return logpdf_meas_eq34(m_t, x_t, params["radius"])

    def grad_meas(pair: tuple[dict, jax.Array]) -> dict[str, jax.Array]:
        m_t, x_t = pair
        return jax.grad(
            lambda p, m=m_t, x=x_t: logpdf_meas_eq34(m, x, p["radius"])
        )(params)

    got = _meas_grad_wrt_params_one_timestep(
        0, measurements, state, g_dyn, logmeas, grad_meas, ni
    )
    m_t = jax.tree.map(lambda leaf: leaf[0], measurements)
    xt = state[0]
    logits = jnp.array([logmeas((m_t, xt[i])) for i in range(ni)])
    w = jax.nn.softmax(logits)
    term_radii = jnp.stack(
        [
            grad_meas((m_t, xt[i]))["radius"] + g_dyn["radius"][i]
            for i in range(ni)
        ],
        axis=0,
    )
    weighted = jnp.sum(w[:, None, None] * term_radii, axis=0)
    mean_dyn = jnp.mean(g_dyn["radius"], axis=0)
    expected = {"radius": weighted - mean_dyn}
    assert jax.tree.all(
        jax.tree.map(
            lambda a, b: jnp.allclose(a, b, rtol=1e-6, atol=1e-7), got, expected
        )
    )


def _make_observed_info_svgd_2d_problem(
    *,
    n_timesteps: int = 4,
    n_svgd_iters: int = 2,
    n_particles: int = 16,
) -> dict:
    """Shared 2D contact setup for ``observed_info_svgd`` tests."""
    learned = jnp.array(
        [[0.0, 3.0], [0.0, 2.0], [0.0, 1.0], [0.0, 0.5]][:n_timesteps]
    )
    x_T = learned[-1:].copy()
    initial = learned[:-1]
    radius = jnp.array([[0.5]])
    measurements = make_sensor_measurements_trajectory(learned, radius)
    omega = ({"radius": radius}, x_T)
    hp = SVGDHyperparameters(
        n_svgd_iters=n_svgd_iters,
        svgd_step=_DYNAMICS_SVGD_STEP,
        init_sample_std=0.05,
        n_particles=n_particles,
    )

    def _logdyn(pair, *, params, var):
        x_t, x_tp1 = pair
        return logpdf_dynamics(x_tp1, x_t, params[0]["radius"], var=var)

    def _grad_dyn(pair, *, params, var):
        return jax.grad(
            lambda omega: logpdf_dynamics(
                pair[1], pair[0], omega[0]["radius"], var=var
            )
        )(params)

    def _logmeas(pair, *, params):
        m_t, x_t = pair
        return logpdf_meas_eq34(m_t, x_t, params[0]["radius"])

    def _grad_meas(pair, *, params):
        m_t, x_t = pair
        return jax.grad(
            lambda omega, m=m_t, x=x_t: logpdf_meas_eq34(
                m, x, omega[0]["radius"]
            )
        )(params)

    logdyn = partial(_logdyn, params=omega, var=BASEVAR)
    grad_dyn = partial(_grad_dyn, params=omega, var=BASEVAR)
    logmeas = partial(_logmeas, params=omega)
    grad_meas = partial(_grad_meas, params=omega)

    n_params = jax.tree_util.tree_flatten(omega)[0][0].size + x_T.size
    return {
        "omega": omega,
        "measurements": measurements,
        "initial": initial,
        "hyperparameters": hp,
        "logdyn": logdyn,
        "grad_dyn": grad_dyn,
        "logmeas": logmeas,
        "grad_meas": grad_meas,
        "n_params": n_params,
    }


def _call_observed_info_svgd(problem: dict, rng: jax.Array) -> jax.Array:
    return observed_info_svgd(
        problem["omega"],
        problem["measurements"],
        problem["initial"],
        problem["logdyn"],
        problem["grad_dyn"],
        problem["logmeas"],
        problem["grad_meas"],
        problem["hyperparameters"],
        rng=rng,
    )


def test_observed_info_svgd_jit_and_timing(capsys):
    """``observed_info_svgd`` compiles with ``jax.jit``; report eager vs JIT timings."""
    enable_jax_cache()
    problem = _make_observed_info_svgd_2d_problem()
    n_timed = 5
    rng0 = jax.random.key(21)
    timing_keys = jax.random.split(jax.random.key(22), n_timed)

    # Compile on zeroed array inputs (same shapes/dtypes) so the timing loop cannot
    # pay for XLA compilation. Partials/hyperparameters are identical to the timed run.
    zero_omega = jax.tree.map(jnp.zeros_like, problem["omega"])
    zero_measurements = jax.tree.map(jnp.zeros_like, problem["measurements"])
    zero_initial = jnp.zeros_like(problem["initial"])
    compile_rng = jax.random.key(0)

    info_eager = _call_observed_info_svgd(problem, rng0)
    assert info_eager.shape == (problem["n_params"], problem["n_params"])
    assert jnp.all(jnp.isfinite(info_eager))

    jitted = jax.jit(
        observed_info_svgd,
        static_argnames=(
            "logpdf_dynamics",
            "grad_logpdf_dynamics",
            "logpdf_meas",
            "grad_logpdf_meas",
            "hyperparameters",
        ),
    )
    jax.block_until_ready(
        jitted(
            zero_omega,
            zero_measurements,
            zero_initial,
            problem["logdyn"],
            problem["grad_dyn"],
            problem["logmeas"],
            problem["grad_meas"],
            problem["hyperparameters"],
            rng=compile_rng,
        )
    )

    info_jit = jitted(
        problem["omega"],
        problem["measurements"],
        problem["initial"],
        problem["logdyn"],
        problem["grad_dyn"],
        problem["logmeas"],
        problem["grad_meas"],
        problem["hyperparameters"],
        rng=rng0,
    )
    assert jnp.allclose(info_eager, info_jit, rtol=1e-5, atol=1e-6)

    eager_ms: list[float] = []
    for key in timing_keys:
        t0 = time.perf_counter()
        out = _call_observed_info_svgd(problem, key)
        jax.block_until_ready(out)
        eager_ms.append((time.perf_counter() - t0) * 1000.0)

    jit_ms: list[float] = []
    for key in timing_keys:
        t0 = time.perf_counter()
        out = jitted(
            problem["omega"],
            problem["measurements"],
            problem["initial"],
            problem["logdyn"],
            problem["grad_dyn"],
            problem["logmeas"],
            problem["grad_meas"],
            problem["hyperparameters"],
            rng=key,
        )
        jax.block_until_ready(out)
        jit_ms.append((time.perf_counter() - t0) * 1000.0)

    assert all(t > 0.0 for t in eager_ms + jit_ms)

    lines = ["observed_info_svgd timing (ms per call):"]
    lines.append(f"  eager: {eager_ms}")
    lines.append(f"  jit:   {jit_ms}")
    lines.append(
        f"  eager mean: {sum(eager_ms) / len(eager_ms):.3f} ms, "
        f"jit mean: {sum(jit_ms) / len(jit_ms):.3f} ms"
    )
    report = "\n".join(lines)
    with capsys.disabled():
        print(report, flush=True)
    assert "eager:" in report
    assert "jit:" in report


def test_grad_meas_wrt_params_matches_pdf_equations():
    """``grad_meas_wrt_params`` is finite, JIT-safe, and self-consistent under ``vmap``."""
    enable_jax_cache()
    n_t, ni = 3, 12
    rng = jax.random.key(42)
    k1, _ = jax.random.split(rng, 2)
    radius = jnp.array([[0.5]])
    params = {"radius": radius}
    state = jax.random.normal(k1, (n_t, ni, 2)) * 0.08
    state = state.at[-1].set(jnp.array([0.0, 0.5]))
    traj = jnp.array([[0.0, 1.5], [0.0, 1.0], [0.0, 0.5]])
    measurements = make_sensor_measurements_trajectory(traj, radius)

    def logdyn(pair: tuple[jax.Array, jax.Array]) -> jax.Array:
        x_t, x_tp1 = pair
        return logpdf_dynamics(x_tp1, x_t, params["radius"])

    def grad_dyn(pair: tuple[jax.Array, jax.Array]) -> dict[str, jax.Array]:
        return jax.grad(lambda p: logpdf_dynamics(pair[1], pair[0], p["radius"]))(params)

    def logmeas(pair: tuple[dict, jax.Array]) -> jax.Array:
        m_t, x_t = pair
        return logpdf_meas_eq34(m_t, x_t, params["radius"])

    def grad_meas(pair: tuple[dict, jax.Array]) -> dict[str, jax.Array]:
        m_t, x_t = pair
        return jax.grad(
            lambda p, m=m_t, x=x_t: logpdf_meas_eq34(m, x, p["radius"])
        )(params)

    impl = grad_meas_wrt_params(
        params,
        measurements,
        state,
        logdyn,
        grad_dyn,
        logmeas,
        grad_meas,
    )
    assert impl["radius"].shape == (n_t, 1, 1)
    assert jnp.all(jnp.isfinite(impl["radius"]))

    jitted = jax.jit(
        grad_meas_wrt_params,
        static_argnames=(
            "logpdf_dynamics",
            "grad_logpdf_dynamics",
            "logpdf_meas",
            "grad_logpdf_meas",
        ),
    )
    impl2 = jitted(
        params,
        measurements,
        state,
        logdyn,
        grad_dyn,
        logmeas,
        grad_meas,
    )
    assert jnp.allclose(impl["radius"], impl2["radius"])


def test_svgd_step_jit_compiles_and_runs():
    def f(x: jax.Array, xp: jax.Array) -> jax.Array:
        return -jnp.sum(jnp.square(x - xp[: x.shape[0]]))

    def gradf(x: jax.Array, xp: jax.Array) -> jax.Array:
        return jax.grad(lambda z: f(z, xp))(x)

    n, d, dp = 4, 3, 5
    x_tree = {"p": jax.random.normal(jax.random.key(1), (n, d))}
    xp_tree = {"p": jax.random.normal(jax.random.key(2), (n, dp))}
    hp = SVGDHyperparameters()

    jitted = jax.jit(lambda xt, xpt: svgd_step(xt, xpt, f, gradf, hp))
    out = jitted(x_tree, xp_tree)

    assert jax.tree.structure(out) == jax.tree.structure(x_tree)
    p_out = out["p"]
    assert p_out.shape == (n, d)
    assert jnp.all(jnp.isfinite(p_out))


def test_svgd_outer_iteration_timing_eager_vs_jit(capsys):
    """Time each outer SVGD backward sweep with and without ``jax.jit`` (ms per iteration)."""
    enable_jax_cache()
    n_t = 4
    ni = 16
    n_timed = 5
    learned = jnp.array([[0.0, 3.0], [0.0, 2.0], [0.0, 1.0], [0.0, 0.5]])
    x_terminal = learned[-1]
    r = jnp.array([[0.5]])
    hp = SVGDHyperparameters(
        n_svgd_iters=1,
        svgd_step=_DYNAMICS_SVGD_STEP,
        init_sample_std=0.05,
    )
    rngs = jax.random.split(jax.random.key(7), n_t)
    particles0 = jnp.zeros((n_t, ni, 2))
    for t in range(n_t - 1):
        noise = jax.random.normal(rngs[t], (ni, 2)) * hp.init_sample_std
        particles0 = particles0.at[t].set(learned[t] + noise)
    particles0 = particles0.at[n_t - 1].set(jnp.broadcast_to(x_terminal, (ni, 2)))

    f_log, grad_log = _make_dynamics_f_and_gradf(r)

    def one_iter(p: jax.Array) -> jax.Array:
        return _one_svgd_outer_iteration(
            p, learned, x_terminal, n_t, f_log, grad_log, hp
        )

    eager_ms: list[float] = []
    p = particles0
    for _ in range(n_timed):
        t0 = time.perf_counter()
        p = one_iter(p)
        jax.block_until_ready(p)
        eager_ms.append((time.perf_counter() - t0) * 1000.0)

    jit_one = jax.jit(one_iter)
    jax.block_until_ready(jit_one(particles0))

    jit_ms: list[float] = []
    p = particles0
    for _ in range(n_timed):
        t0 = time.perf_counter()
        p = jit_one(p)
        jax.block_until_ready(p)
        jit_ms.append((time.perf_counter() - t0) * 1000.0)

    assert jnp.all(jnp.isfinite(p))
    assert all(t > 0.0 for t in eager_ms + jit_ms)

    lines = ["SVGD outer-iteration timing (ms per iter):"]
    lines.append(f"  eager: {eager_ms}")
    lines.append(f"  jit:   {jit_ms}")
    lines.append(
        f"  eager mean: {sum(eager_ms) / len(eager_ms):.3f} ms, "
        f"jit mean: {sum(jit_ms) / len(jit_ms):.3f} ms"
    )
    report = "\n".join(lines)
    with capsys.disabled():
        print(report, flush=True)
    assert "eager:" in report
    assert "jit:" in report


def test_svgd_dynamics_2d_ground_contact_particles():
    """Same 2D falling-sphere setup as ``test_theory_v3``; SVGD via ``svgd_step``."""
    enable_jax_cache()
    n_t = 5
    r = jnp.array([[0.5]])
    learned = jnp.array(
        [
            [0.0, 4.0],
            [0.0, 3.0],
            [0.0, 2.0],
            [0.0, 1.0],
            [0.0, 0.5],
        ]
    )
    x_terminal = learned[-1]
    hp = SVGDHyperparameters(
        n_svgd_iters=16, svgd_step=_DYNAMICS_SVGD_STEP, init_sample_std=0.05
    )
    parts = sample_dynamics_particles_svgd(
        learned,
        x_terminal,
        r,
        jax.random.key(0),
        hp,
        n_particles=48,
        visualize=False,
    )
    assert parts.shape == (n_t, 48, 2)
    assert jnp.all(jnp.isfinite(parts))
    assert jnp.allclose(parts[-1], jnp.broadcast_to(x_terminal, (48, 2)))


def test_svgd_dynamics_2d_snapshot_figure(tmp_path):
    """matplotlib Agg: timestep-colored scatter matching ``sample_state_particles`` style."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    enable_jax_cache()
    n_t = 5
    r = jnp.array([[0.5]])
    learned = jnp.array(
        [
            [0.0, 4.0],
            [0.0, 3.0],
            [0.0, 2.0],
            [0.0, 1.0],
            [0.0, 0.5],
        ]
    )
    hp = SVGDHyperparameters(
        n_svgd_iters=8, svgd_step=_DYNAMICS_SVGD_STEP, init_sample_std=0.05
    )
    parts = sample_dynamics_particles_svgd(
        learned,
        learned[-1],
        r,
        jax.random.key(2),
        hp,
        n_particles=32,
        visualize=False,
    )

    cmap = plt.get_cmap("viridis", n_t)
    _, ax = plt.subplots(figsize=(5, 4))
    for t in range(n_t):
        xy = jax.device_get(parts[t])
        ax.scatter(
            xy[:, 0], xy[:, 1], s=14, alpha=0.65, color=cmap(t), edgecolors="none"
        )
    learned_np = jax.device_get(learned)
    xT = jax.device_get(learned[-1])
    ax.plot(learned_np[:, 0], learned_np[:, 1], "k--", linewidth=1.0, alpha=0.7)
    ax.scatter([xT[0]], [xT[1]], c="k", s=36, marker="x")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("SVGD (marginalization) timestep-colored particles — final iterate")
    ax.grid(True, alpha=0.2)
    out = tmp_path / "svgd_marginalization_2d.png"
    plt.savefig(out, dpi=120)
    plt.close()
    assert out.is_file()


@pytest.mark.skipif(
    not os.environ.get("DAIR_SVGD_VISUALIZE"),
    reason="Set DAIR_SVGD_VISUALIZE=1 to open the matplotlib animation.",
)
def test_svgd_dynamics_2d_visualization_interactive():
    """Same dynamics test with per-iteration timestep coloring (matplotlib)."""
    enable_jax_cache()
    n_t = 5
    r = jnp.array([[0.5]])
    learned = jnp.array(
        [
            [0.0, 4.0],
            [0.0, 3.0],
            [0.0, 2.0],
            [0.0, 1.0],
            [0.0, 0.5],
        ]
    )
    hp = SVGDHyperparameters(
        n_svgd_iters=100, svgd_step=_DYNAMICS_SVGD_STEP, init_sample_std=0.5
    )
    sample_dynamics_particles_svgd(
        learned,
        learned[-1],
        r,
        jax.random.key(1),
        hp,
        n_particles=48,
        visualize=True,
        visualize_pause_s=0.05,
    )


if __name__ == "__main__":
    test_svgd_step_jit_compiles_and_runs()
    test_svgd_dynamics_2d_ground_contact_particles()
    print("test_svgd_marginalization: ok")
