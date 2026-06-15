#!/usr/bin/env python3

"""Tests for SVGD marginalization utilities."""

from __future__ import annotations

import math
import os
import time
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

from dair_exploration.file_util import enable_jax_cache
from dair_exploration.svgd_marginalization import (
    SVGDHyperparameters,
    _dynamics_grad_wrt_params_one_particle,
    _meas_grad_wrt_params_one_timestep,
    _sample_state_particles_svgd,
    _tree_broadcast_params_zeros,
    _tree_index_leading,
    grad_meas_wrt_params,
    observed_info_svgd,
    sum_observed_info,
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
    "finger": jnp.array([1.0, 3.0]),
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


def sample_noisy_contact_normal_W(
    n_hat: jax.Array,
    key: jax.Array,
    *,
    sigma_n: float = MEAS_SIGMA_N,
) -> jax.Array:
    """Unit measured normal; ``(1 - n_hat·n_meas) ~ N(0, sigma_n^2)`` when in contact."""
    in_contact = jnp.linalg.norm(n_hat) > 0.5
    one_minus_dot = jax.random.normal(key) * sigma_n
    dot_target = jnp.clip(1.0 - one_minus_dot, -1.0, 1.0)
    delta = jnp.arccos(dot_target)
    sign = jnp.where(jax.random.randint(key, (), 0, 2) == 0, -1.0, 1.0)
    theta = jnp.arctan2(n_hat[1], n_hat[0])
    theta_meas = theta + sign * delta
    noisy = jnp.stack([jnp.cos(theta_meas), jnp.sin(theta_meas)])
    return jnp.where(in_contact, noisy, jnp.zeros_like(n_hat))


def logpdf_meas_eq34_one_sensor_terms(
    sensor_meas: dict[str, jax.Array],
    object_center: jax.Array,
    radius: jax.Array,
    *,
    sigma_n: float = MEAS_SIGMA_N,
    alpha: float = MEAS_ALPHA,
) -> dict[str, jax.Array]:
    """Per-component ``log p`` for one sensor: ``normal`` and ``contact``."""
    sensor_pos = sensor_meas["position"]
    meas_normal = sensor_meas["contact_normal_W"]
    phi = signed_distance_phi(sensor_pos, object_center, radius)
    n_hat = predicted_contact_normal_W(sensor_pos, object_center, radius)
    contact_bool = jnp.clip(jnp.round(jnp.linalg.norm(meas_normal)), 0.0, 1.0)
    phi_pos = jnp.maximum(phi, 0.0)
    inv_sigma_sq = jnp.reciprocal(sigma_n**2)
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
    return {"normal": normal_term, "contact": contact_term}


def logpdf_meas_eq34_one_sensor(
    sensor_meas: dict[str, jax.Array],
    object_center: jax.Array,
    radius: jax.Array,
    **kwargs: float,
) -> jax.Array:
    """Scalar log-density for one sensor (sum of all components)."""
    terms = logpdf_meas_eq34_one_sensor_terms(
        sensor_meas, object_center, radius, **kwargs
    )
    return terms["normal"] + terms["contact"]


def logpdf_meas_eq34_pytree(
    measurements_t: dict[str, dict[str, jax.Array]],
    object_center: jax.Array,
    radius: jax.Array,
    **kwargs: float,
) -> dict[str, dict[str, jax.Array]]:
    """``{sensor_name: {normal, contact}}`` log terms in sorted sensor order."""
    return {
        name: logpdf_meas_eq34_one_sensor_terms(
            measurements_t[name], object_center, radius, **kwargs
        )
        for name in sorted(measurements_t.keys())
    }


def grad_logpdf_meas_eq34_pytree(
    measurements_t: dict[str, dict[str, jax.Array]],
    object_center: jax.Array,
    params: Any,
    *,
    link_to_terminal: jax.Array | bool = False,
    **kwargs: float,
) -> Any:
    """Jacobian matching :func:`logpdf_meas_eq34_pytree`; each leaf is a ``params`` PyTree."""
    link = jnp.asarray(link_to_terminal)

    def log_all(omega: Any) -> dict[str, dict[str, jax.Array]]:
        center = jax.lax.select(link, omega[1][0], object_center)
        return logpdf_meas_eq34_pytree(
            measurements_t, center, omega[0]["radius"], **kwargs
        )

    return jax.jacfwd(log_all)(params)


def logpdf_meas_eq34(
    measurements_t: dict[str, dict[str, jax.Array]],
    object_center: jax.Array,
    radius: jax.Array,
    **kwargs: float,
) -> jax.Array:
    """Sum of all per-sensor, per-component log terms."""
    tree = logpdf_meas_eq34_pytree(measurements_t, object_center, radius, **kwargs)
    return sum(jax.tree.leaves(tree))


def make_sensor_measurements_trajectory(
    object_centers: jax.Array,
    radius: jax.Array,
    sensor_positions: dict[str, jax.Array] | None = None,
    *,
    rng: jax.Array | None = None,
    sigma_n: float = MEAS_SIGMA_N,
) -> dict[str, dict[str, jax.Array]]:
    """``{name: {position, contact_normal_W}}`` with leaves shaped ``(n_timesteps, n_dim)``.

    In-contact normals are noisy: ``(1 - n_hat·n_meas) ~ N(0, sigma_n^2)``, matching
    :func:`logpdf_meas_eq34_one_sensor_terms` (``sigma_n`` defaults to :data:`MEAS_SIGMA_N`).
    """
    sensors = DEFAULT_SENSOR_POSITIONS if sensor_positions is None else sensor_positions
    n_t = object_centers.shape[0]
    if rng is None:
        rng = jax.random.key(0)
    keys = jax.random.split(rng, len(sensors) * n_t)
    out: dict[str, dict[str, jax.Array]] = {}
    for i, (name, sensor_pos) in enumerate(sensors.items()):
        n_hat = jax.vmap(
            lambda center: predicted_contact_normal_W(sensor_pos, center, radius)
        )(object_centers)
        sensor_keys = keys[i * n_t : (i + 1) * n_t]
        noisy_normal = partial(sample_noisy_contact_normal_W, sigma_n=sigma_n)
        normals = jax.vmap(noisy_normal)(n_hat, sensor_keys)
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

    def grad_dyn(
        pair: tuple[jax.Array, jax.Array], *, link_to_terminal=False
    ) -> dict[str, jax.Array]:
        del link_to_terminal
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

    def logmeas(pair: tuple[dict, jax.Array]) -> dict[str, dict[str, jax.Array]]:
        m_t, x_t = pair
        return logpdf_meas_eq34_pytree(m_t, x_t, params["radius"])

    def grad_meas(
        pair: tuple[dict, jax.Array], *, link_to_terminal=False
    ) -> dict[str, dict[str, Any]]:
        del link_to_terminal
        m_t, x_t = pair

        def grad_sensor(name: str) -> dict[str, Any]:
            def grad_term(term: str) -> dict[str, jax.Array]:
                def loss(p: dict[str, jax.Array]) -> jax.Array:
                    terms = logpdf_meas_eq34_one_sensor_terms(
                        m_t[name], x_t, p["radius"]
                    )
                    return terms[term]

                return jax.grad(loss)(params)

            return {term: grad_term(term) for term in ("normal", "contact")}

        return {name: grad_sensor(name) for name in sorted(m_t.keys())}

    got = _meas_grad_wrt_params_one_timestep(
        0, measurements, state, g_dyn, logmeas, grad_meas, ni
    )
    m_t = jax.tree.map(lambda leaf: leaf[0], measurements)
    xt = state[0]
    mean_dyn = jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), g_dyn)

    def expected_for_leaf(name: str, term: str) -> dict[str, jax.Array]:
        logits = jnp.array([logmeas((m_t, xt[i]))[name][term] for i in range(ni)])
        w = jax.nn.softmax(logits)
        term_grads = jax.tree.map(
            lambda *rows: jnp.stack(rows, axis=0),
            *[
                jax.tree.map(
                    jnp.add,
                    grad_meas((m_t, xt[i]))[name][term],
                    _tree_index_leading(g_dyn, i),
                )
                for i in range(ni)
            ],
        )
        weighted = jax.tree.map(
            lambda leaf: jnp.sum(
                w.reshape((ni,) + (1,) * (leaf.ndim - 1)) * leaf, axis=0
            ),
            term_grads,
        )
        return jax.tree.map(jnp.subtract, weighted, mean_dyn)

    expected = {
        name: {term: expected_for_leaf(name, term) for term in ("normal", "contact")}
        for name in sorted(m_t.keys())
    }
    assert jax.tree.all(
        jax.tree.map(
            lambda a, b: jnp.allclose(a, b, rtol=1e-6, atol=1e-7), got, expected
        )
    )


def _offdiag_frobenius_ratio(info: jax.Array) -> float:
    """``||I - diag(I)||_F / max_i |I_ii|`` — small when ``I`` is nearly diagonal."""
    diag = jnp.diag(info)
    off = info - jnp.diag(diag)
    return float(jnp.linalg.norm(off) / jnp.max(jnp.abs(diag)))


def _format_info_contrib_by_meas_term(problem: dict, rng: jax.Array) -> str:
    """Per-leaf ``g g^T`` by timestep (Eq. 12 grads); param order ``r, x, z``."""
    particles = _sample_state_particles_svgd(
        problem["initial"],
        problem["omega"][1],
        rng,
        problem["hyperparameters"],
        problem["logdyn"],
    )
    grads = grad_meas_wrt_params(
        problem["omega"],
        problem["measurements"],
        particles,
        problem["logdyn"],
        problem["grad_dyn"],
        problem["logmeas"],
        problem["grad_meas"],
    )
    n_t = jax.tree.leaves(grads)[0].shape[0]
    lines = ["Per-term observed-info contributions (g g^T by t):"]
    for t in range(n_t):
        total_t = jnp.zeros((3, 3))
        leaf_lines: list[str] = []
        for sensor in sorted(grads.keys()):
            for term in ("normal", "contact"):
                g_t = jax.tree.map(lambda leaf, tt=t: leaf[tt], grads[sensor][term])
                vec, _ = ravel_pytree(g_t)
                mat = jnp.outer(vec, vec)
                total_t = total_t + mat
                tr = float(jnp.trace(mat))
                if tr <= 1e-6 and float(jnp.max(jnp.abs(mat))) <= 1e-6:
                    continue
                d = jax.device_get(jnp.diag(mat))
                g = jax.device_get(vec.reshape(-1))
                leaf_lines.append(
                    f"    {sensor}/{term}: trace={tr:.4g} diag(r,x,z)={d} "
                    f"g=({g[0]:.4g}, {g[1]:.4g}, {g[2]:.4g})"
                )
        tr_total = float(jnp.trace(total_t))
        if tr_total <= 1e-6:
            continue
        lines.append(
            f"  t={t}: total trace={tr_total:.4g} "
            f"diag(r,x,z)={jax.device_get(jnp.diag(total_t))}"
        )
        lines.extend(leaf_lines)
    return "\n".join(lines)


def _make_observed_info_svgd_2d_problem(
    *,
    n_timesteps: int = 4,
    n_svgd_iters: int = 50,
    n_particles: int = 32,
    sensor_positions: dict[str, jax.Array] | None = None,
    learned_centers: jax.Array | None = None,
    meas_rng: jax.Array | None = None,
) -> dict:
    """Shared 2D contact setup for ``observed_info_svgd`` tests."""
    default_learned = jnp.array(
        [[0.0, 3.0], [0.0, 2.0], [0.0, 1.0], [0.0, 0.5]][:n_timesteps]
    )
    learned = default_learned if learned_centers is None else learned_centers
    if learned.shape[0] != n_timesteps:
        raise ValueError(
            f"learned_centers has {learned.shape[0]} rows but n_timesteps={n_timesteps}"
        )
    x_final = learned[-1:].copy()
    initial = learned[:-1]
    radius = jnp.array([[0.5]])
    measurements = make_sensor_measurements_trajectory(
        learned, radius, sensor_positions, rng=meas_rng
    )
    omega = ({"radius": radius}, x_final)
    hp = SVGDHyperparameters(
        n_svgd_iters=n_svgd_iters,
        svgd_step=_DYNAMICS_SVGD_STEP,
        init_sample_std=0.2,
        n_particles=n_particles,
    )

    def _logdyn(pair, *, params, var):
        x_t, x_tp1 = pair
        return logpdf_dynamics(x_tp1, x_t, params[0]["radius"], var=var)

    def _grad_dyn(pair, *, params, var, link_to_terminal=False):
        x_t, x_tp1 = pair
        link = jnp.asarray(link_to_terminal)

        def loss(omega):
            x_next = jax.lax.select(link, omega[1][0], x_tp1)
            return logpdf_dynamics(x_next, x_t, omega[0]["radius"], var=var)

        return jax.grad(loss)(params)

    def _logmeas(pair, *, params):
        m_t, x_t = pair
        return logpdf_meas_eq34_pytree(m_t, x_t, params[0]["radius"])

    def _grad_meas(pair, *, params, link_to_terminal=False):
        m_t, x_t = pair
        return grad_logpdf_meas_eq34_pytree(
            m_t, x_t, params, link_to_terminal=link_to_terminal
        )

    logdyn = partial(_logdyn, params=omega, var=BASEVAR)
    grad_dyn = partial(_grad_dyn, params=omega, var=BASEVAR)
    logmeas = partial(_logmeas, params=omega)
    grad_meas = partial(_grad_meas, params=omega)

    n_params = jax.tree_util.tree_flatten(omega)[0][0].size + x_final.size
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
    info_by_meas = observed_info_svgd(
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
    return sum_observed_info(info_by_meas)


def test_observed_info_far_sensor_is_zero(capsys):
    """Distant sensor (no contact): observed information is numerically zero."""
    enable_jax_cache()
    far_sensor = {"finger": jnp.array([1.0, 4.0])}
    problem = _make_observed_info_svgd_2d_problem(
        n_svgd_iters=16,
        n_particles=32,
        sensor_positions=far_sensor,
    )
    info = _call_observed_info_svgd(problem, jax.random.key(31))

    report = (
        "Observed information (sensor at (1.0, 4.0), no contact):\n"
        f"{jax.device_get(info)}\n"
        f"max |entry| = {float(jnp.max(jnp.abs(info))):.6e}"
    )
    with capsys.disabled():
        print(report, flush=True)

    assert info.shape == (problem["n_params"], problem["n_params"])
    assert jnp.allclose(info, 0.0, atol=1e-5)


def test_observed_info_tangential_sensor_constrains_z_and_x_minus_r(capsys):
    """Sensor at (0.5, 0.5): on the radius at terminal; info on z and x−r."""
    enable_jax_cache()
    tangential_sensor = {"finger": jnp.array([0.5, 0.5])}
    problem = _make_observed_info_svgd_2d_problem(
        n_svgd_iters=16,
        n_particles=32,
        sensor_positions=tangential_sensor,
    )
    info = _call_observed_info_svgd(problem, jax.random.key(32))
    x_final = problem["omega"][1][0]
    radius = float(problem["omega"][0]["radius"].reshape(()))
    phi_terminal = float(
        signed_distance_phi(
            tangential_sensor["finger"], x_final, problem["omega"][0]["radius"]
        )
    )

    # Param order from ``ravel_pytree(omega)``: radius, x_T[0], x_T[1].
    v_x_minus_r = jnp.array([-1.0, 1.0, 0.0])
    quad_x_minus_r = float(v_x_minus_r @ info @ v_x_minus_r)

    report = (
        "Observed information (sensor at (0.5, 0.5), tangential at terminal):\n"
        f"{jax.device_get(info)}\n"
        f"terminal x_final = {jax.device_get(x_final)}, radius = {radius}, "
        f"phi_terminal = {phi_terminal:.6f}, x_final[0]-r = {float(x_final[0] - radius):.6f}\n"
        f"I_zz = {float(info[2, 2]):.6e}, quad form on x-r direction = {quad_x_minus_r:.6e}"
    )
    with capsys.disabled():
        print(report, flush=True)

    assert info.shape == (problem["n_params"], problem["n_params"])
    assert jnp.all(jnp.isfinite(info))
    assert jnp.max(jnp.abs(info)) > 1.0
    assert phi_terminal < 1e-4
    assert info[2, 2] > 1.0
    assert quad_x_minus_r > 1.0
    assert info[0, 0] > info[1, 1]


def test_observed_info_two_sensors_at_height_one(capsys):
    """Symmetric sensors at (±0.5, 1.0): per-sensor scores identify r, x, and z at contact."""
    enable_jax_cache()
    sensors = {
        "left": jnp.array([-0.5, 1.0]),
        "right": jnp.array([0.5, 1.0]),
    }
    rng = jax.random.key(33)
    problem = _make_observed_info_svgd_2d_problem(
        n_svgd_iters=50,
        n_particles=32,
        sensor_positions=sensors,
        meas_rng=jax.random.fold_in(rng, 1),
    )
    info = _call_observed_info_svgd(problem, rng)
    x_final = problem["omega"][1][0]
    radius = problem["omega"][0]["radius"]
    center_at_z1 = jnp.array([0.0, 1.0])
    phi_z1 = {
        name: float(signed_distance_phi(pos, center_at_z1, radius))
        for name, pos in sensors.items()
    }
    diag = jnp.diag(info)
    offdiag_ratio = _offdiag_frobenius_ratio(info)

    residual_z1 = {
        name: float(
            1.0
            - jnp.dot(
                predicted_contact_normal_W(pos, center_at_z1, radius),
                problem["measurements"][name]["contact_normal_W"][2],
            )
        )
        for name, pos in sensors.items()
    }
    report = (
        "Observed information (sensors at (-0.5, 1.0) and (0.5, 1.0)):\n"
        f"{jax.device_get(info)}\n"
        f"terminal x_final = {jax.device_get(x_final)}, radius = {float(radius.reshape(()))}\n"
        f"phi at z=1 (t=2 center): {phi_z1}\n"
        f"1 - n_hat·n_meas at t=2: {residual_z1}\n"
        f"diag(r, x_final[0], x_final[1]) = {jax.device_get(diag)}\n"
        f"off-diagonal Frobenius / max|diag| = {offdiag_ratio:.4f}\n"
        f"{_format_info_contrib_by_meas_term(problem, rng)}"
    )
    with capsys.disabled():
        print(report, flush=True)

    assert info.shape == (problem["n_params"], problem["n_params"])
    assert jnp.all(jnp.isfinite(info))
    assert all(phi < 1e-4 for phi in phi_z1.values())
    assert diag[0] > 1.0e3
    assert diag[1] > 1.0
    assert diag[2] > 1.0e3
    info_by_meas = observed_info_svgd(
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
    assert float(info_by_meas["left"]["normal"][2, 2]) > 1.0e3


def test_observed_info_two_sensors_at_terminal_height_near_diagonal(capsys):
    """Symmetric sensors at (±0.5, 0.5): contact at terminal; ``I`` nearly diagonal."""
    enable_jax_cache()
    sensors = {
        "left": jnp.array([-0.5, 0.5]),
        "right": jnp.array([0.5, 0.5]),
    }
    rng = jax.random.key(35)
    problem = _make_observed_info_svgd_2d_problem(
        n_svgd_iters=50,
        n_particles=32,
        sensor_positions=sensors,
        meas_rng=jax.random.fold_in(rng, 1),
    )
    info = _call_observed_info_svgd(problem, rng)
    x_final = problem["omega"][1][0]
    radius = problem["omega"][0]["radius"]
    phi_terminal = {
        name: float(signed_distance_phi(pos, x_final, radius))
        for name, pos in sensors.items()
    }
    diag = jnp.diag(info)
    offdiag_ratio = _offdiag_frobenius_ratio(info)

    residual_terminal = {
        name: float(
            1.0
            - jnp.dot(
                predicted_contact_normal_W(pos, x_final, radius),
                problem["measurements"][name]["contact_normal_W"][3],
            )
        )
        for name, pos in sensors.items()
    }
    report = (
        "Observed information (sensors at (-0.5, 0.5) and (0.5, 0.5)):\n"
        f"{jax.device_get(info)}\n"
        f"terminal x_final = {jax.device_get(x_final)}, radius = {float(radius.reshape(()))}\n"
        f"phi at terminal: {phi_terminal}\n"
        f"1 - n_hat·n_meas at t=3: {residual_terminal}\n"
        f"diag(r, x_final[0], x_final[1]) = {jax.device_get(diag)}\n"
        f"off-diagonal Frobenius / max|diag| = {offdiag_ratio:.4f}\n"
        f"{_format_info_contrib_by_meas_term(problem, rng)}"
    )
    with capsys.disabled():
        print(report, flush=True)

    assert info.shape == (problem["n_params"], problem["n_params"])
    assert jnp.all(jnp.isfinite(info))
    assert all(phi < 1e-4 for phi in phi_terminal.values())
    assert any(abs(r) > 1e-6 for r in residual_terminal.values())
    assert diag[0] > 1.0e4
    assert diag[1] > 1.0e4
    assert diag[2] > 1.0e5
    assert offdiag_ratio < 0.05
    info_by_meas = observed_info_svgd(
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
    assert float(info_by_meas["left"]["normal"][2, 2]) > 1.0e5
    tangential = _make_observed_info_svgd_2d_problem(
        n_svgd_iters=50,
        n_particles=32,
        sensor_positions={"finger": jnp.array([0.5, 0.5])},
    )
    info_tangential = _call_observed_info_svgd(tangential, jax.random.key(36))
    assert offdiag_ratio < 0.2 * _offdiag_frobenius_ratio(info_tangential)


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

    info_jit = sum_observed_info(
        jitted(
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
    )
    # Many SVGD sweeps amplify harmless eager vs XLA rounding drift (not a logic bug).
    assert jnp.allclose(info_eager, info_jit, rtol=1e-4, atol=1e-4)

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
    x_T = jnp.array([[0.0, 0.5]])
    omega = ({"radius": radius}, x_T)
    state = jax.random.normal(k1, (n_t, ni, 2)) * 0.08
    state = state.at[-1].set(jnp.array([0.0, 0.5]))
    traj = jnp.array([[0.0, 1.5], [0.0, 1.0], [0.0, 0.5]])
    measurements = make_sensor_measurements_trajectory(traj, radius)

    def logdyn(pair: tuple[jax.Array, jax.Array]) -> jax.Array:
        x_t, x_tp1 = pair
        return logpdf_dynamics(x_tp1, x_t, omega[0]["radius"])

    def grad_dyn(
        pair: tuple[jax.Array, jax.Array], *, link_to_terminal=False
    ) -> Any:
        x_t, x_tp1 = pair
        link = jnp.asarray(link_to_terminal)

        def loss(params: Any) -> jax.Array:
            x_next = jax.lax.select(link, params[1][0], x_tp1)
            return logpdf_dynamics(x_next, x_t, params[0]["radius"])

        return jax.grad(loss)(omega)

    def logmeas(pair: tuple[dict, jax.Array]) -> dict[str, dict[str, jax.Array]]:
        m_t, x_t = pair
        return logpdf_meas_eq34_pytree(m_t, x_t, omega[0]["radius"])

    def grad_meas(
        pair: tuple[dict, jax.Array], *, link_to_terminal=False
    ) -> dict[str, dict[str, Any]]:
        m_t, x_t = pair
        return grad_logpdf_meas_eq34_pytree(
            m_t, x_t, omega, link_to_terminal=link_to_terminal
        )

    impl = grad_meas_wrt_params(
        omega,
        measurements,
        state,
        logdyn,
        grad_dyn,
        logmeas,
        grad_meas,
    )
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(impl))

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
        omega,
        measurements,
        state,
        logdyn,
        grad_dyn,
        logmeas,
        grad_meas,
    )
    assert jax.tree.all(
        jax.tree.map(
            lambda a, b: jnp.allclose(a, b, rtol=1e-4, atol=1e-4), impl, impl2
        )
    )


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
