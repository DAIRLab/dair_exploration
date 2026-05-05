#!/usr/bin/env python3

"""
Stein variational observed information (docs/sv_observed_info.pdf) in a 2D toy:
object-geom (sphere radius θ, translation x,z), ground at z=0, two point spherebots.

Run directly (no pytest): ``python tests/test_theory_v3.py`` from the repo root, or
``python test_theory_v3.py`` from ``tests/``, with the package installed (e.g.
``pip install -e .`` in your virtualenv).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import jax
import jax.numpy as jnp

from dair_exploration.file_util import enable_jax_cache


# Mirrors dair_exploration.exploration (same field names / defaults); defined here so
# tests do not import exploration.py (which loads mujoco.mjx).
class InfoStyle(Enum):
    IDENTITY = 0
    DIFFSIM = 1
    SAMPLING = 2


@dataclass(frozen=True)
class InfoHyperparameters:
    phi_nominal: float = 0.002
    phi_ci: float = 0.05
    normal_var: float = 0.2
    epsilon: float = 1e-8
    style: InfoStyle = InfoStyle.SAMPLING

SPEED = 1.0
BASEVAR = 0.001
PENALTY = 100.0


@dataclass(frozen=True)
class SvObservedInfoHyperparameters:
    """Extra knobs for SVGD sampling + observed-info Monte Carlo size."""

    n_particles: int = 32
    n_svgd_iters: int = 12
    svgd_step: float = 0.15
    init_sample_std: float = 0.08


def _phi_alpha(hp: InfoHyperparameters) -> jax.Array:
    return jnp.log(jnp.reciprocal(hp.phi_ci) - 1.0) / hp.phi_nominal


def dynamics_object(x: jax.Array, radius: jax.Array) -> jax.Array:
    """Constant downward speed in z until the sphere center rests at z = radius."""
    r = radius.reshape(())
    z_next = jnp.maximum(x[1] - SPEED, r)
    return jnp.stack([x[0], z_next])


def logpdf_dynamics(
    x_next: jax.Array,
    x_curr: jax.Array,
    radius: jax.Array,
    var: float = BASEVAR,
) -> jax.Array:
    """log p(x_curr | x_next, θ): ContactNets-style ground contact at z = radius (v2 analog)."""
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


def model_contact_normal(
    object_pos: jax.Array, robot_pos: jax.Array, radius: jax.Array
) -> jax.Array:
    """Unit vector from robot toward object when ||o-s|| <= r, else zeros."""
    r = radius.reshape(())
    diff = object_pos - robot_pos
    dist = jnp.linalg.norm(diff)
    unit = diff / jnp.maximum(dist, 1e-8)
    return jnp.where(dist <= r, unit, jnp.zeros_like(diff))


def logmeas_timestep(
    measurements_t: dict[str, dict[str, jax.Array]],
    object_pos: jax.Array,
    radius: jax.Array,
    hp: InfoHyperparameters,
) -> jax.Array:
    """Per-timestep log p(m_t | x_t, θ) from sv_observed_info.pdf eq. (3)-(4), two spherebots."""
    alpha = _phi_alpha(hp)
    inv_sigma2 = jnp.reciprocal(hp.normal_var)
    r = radius.reshape(())
    total = jnp.array(0.0)
    for name in ("spherebot1-geom", "spherebot2-geom"):
        pack = measurements_t[name]
        s = pack["position"].reshape((2,))
        n_meas = pack["contact_normal_W"].reshape((2,))
        c = jnp.round(jnp.sum(jnp.square(n_meas)))
        n_hat = model_contact_normal(object_pos, s, r)
        d = jnp.linalg.norm(object_pos - s)
        phi = d - r
        normal_term = (
            -0.5 * inv_sigma2 * c * jnp.sum(jnp.square(n_meas - n_hat))
            - 0.5 * c * jnp.log(2.0 * jnp.pi * hp.normal_var) * 2.0
        )
        contact_term = (c - 1.0) * alpha * phi + jax.nn.softplus(alpha * phi)
        total = total + normal_term + contact_term
    return total


def _rbf_kernel_matrix(x: jax.Array, h_sq: jax.Array) -> jax.Array:
    """k(x_i, x_j) = exp(-||x_i-x_j||^2 / (2 h^2)), x shape (Ni, 2)."""
    diff = x[:, None, :] - x[None, :, :]
    d2 = jnp.sum(jnp.square(diff), axis=-1)
    return jnp.exp(-0.5 * d2 / h_sq)


def _median_pairwise_sq_dist(x: jax.Array) -> jax.Array:
    d2 = jnp.sum(jnp.square(x[:, None, :] - x[None, :, :]), axis=-1)
    upper = jnp.triu(d2, k=1)
    flat = upper[jnp.triu_indices_from(upper, k=1)]
    return jnp.median(flat)


def _svgd_step_on_timestep(
    x_particles: jax.Array,
    x_tp1_particles: jax.Array,
    radius: jax.Array,
    step_size: float,
) -> jax.Array:
    """One SVGD step (pdf eq. 5) on p(x_t | x_{t+1}) using softmax-approx gradient (eq. 10)."""
    ni = x_particles.shape[0]
    d2_med = _median_pairwise_sq_dist(x_particles)
    h_sq = jnp.maximum(d2_med / jnp.log(float(ni)), 1e-8)

    def grad_log_cond(xi: jax.Array) -> jax.Array:
        fs = jax.vmap(lambda xp1: logpdf_dynamics(xp1, xi, radius))(x_tp1_particles)
        w = jax.nn.softmax(fs)
        g = jax.vmap(
            lambda xp1: jax.grad(logpdf_dynamics, argnums=1)(xp1, xi, radius)
        )(x_tp1_particles)
        return w @ g

    grads = jax.vmap(grad_log_cond)(x_particles)
    kmat = _rbf_kernel_matrix(x_particles, h_sq)

    def kernel_grad_row(i: int) -> jax.Array:
        def k_ij(j: int) -> jax.Array:
            d = x_particles[j] - x_particles[i]
            return kmat[j, i] * d / h_sq

        return jax.vmap(k_ij)(jnp.arange(ni)).sum(axis=0)

    kernel_grads = jax.vmap(kernel_grad_row)(jnp.arange(ni))
    drive = (kmat @ grads + jnp.sum(kernel_grads, axis=0)) / float(ni)
    return x_particles + step_size * drive


def sample_state_particles(
    learned_traj: jax.Array,
    x_terminal: jax.Array,
    radius: jax.Array,
    rng: jax.Array,
    hp: SvObservedInfoHyperparameters,
    visualize: bool = True,
    visualize_pause_s: float = 0.05,
) -> jax.Array:
    """SVGD particles (nT, Ni, 2). Last timestep rows equal x_terminal."""
    n_t, _ = learned_traj.shape
    ni = hp.n_particles
    rngs = jax.random.split(rng, n_t)
    particles = jnp.zeros((n_t, ni, 2))
    for t in range(n_t - 1):
        noise = jax.random.normal(rngs[t], (ni, 2)) * hp.init_sample_std
        particles = particles.at[t].set(learned_traj[t] + noise)
    particles = particles.at[n_t - 1].set(jnp.broadcast_to(x_terminal, (ni, 2)))

    if visualize:
        # Local import so tests/headless runs don't require matplotlib.
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
            plt.title(f"SVGD particles by timestep — iter {it + 1}/{hp.n_svgd_iters}")
            plt.xlabel("x")
            plt.ylabel("z")
            plt.axis("equal")
            plt.grid(True, alpha=0.2)
            plt.pause(visualize_pause_s)
        
        new_parts = [particles[n_t - 1]]
        for t in range(n_t - 2, -1, -1):
            new_parts.append(
                _svgd_step_on_timestep(
                    particles[t],
                    new_parts[-1],
                    radius,
                    hp.svgd_step,
                )
            )
        rev = list(reversed(new_parts))
        stacked = jnp.stack(rev, axis=0)
        particles = stacked.at[n_t - 1].set(jnp.broadcast_to(x_terminal, (ni, 2)))

    if visualize:
        plt.ioff()
    return particles


def _log_p_xt_cond_x_tp1_particles(
    xt: jax.Array, x_tp1_particles: jax.Array, radius: jax.Array
) -> jax.Array:
    fs = jax.vmap(lambda xp1: logpdf_dynamics(xp1, xt, radius))(x_tp1_particles)
    return jax.scipy.special.logsumexp(fs)


def _terminal_log_factor(xt: jax.Array, x_terminal: jax.Array) -> jax.Array:
    return -1.0e4 * jnp.sum(jnp.square(xt - x_terminal))


def _x_tp1_rows(
    t: int,
    n_t: int,
    particles: jax.Array,
    x_terminal: jax.Array,
    grad_wrt_terminal: bool,
) -> jax.Array:
    """Rows of x_{t+1} particles; last timestep uses x_terminal (optionally AD-traced)."""
    ni = particles.shape[1]
    if t + 1 == n_t - 1:
        rows = jnp.broadcast_to(x_terminal, (ni, 2))
        return rows if grad_wrt_terminal else jax.lax.stop_gradient(rows)
    return jax.lax.stop_gradient(particles[t + 1])


def log_p_xt_given_xT(
    xt: jax.Array,
    t: int,
    n_t: int,
    particles: jax.Array,
    x_terminal: jax.Array,
    radius: jax.Array,
    grad_wrt_terminal: bool = False,
) -> jax.Array:
    """Discrete log p(x_t | x_T) using frozen interior particles (pdf eq. 9)."""
    if t == n_t - 1:
        return _terminal_log_factor(xt, x_terminal)
    x_tp1 = _x_tp1_rows(t, n_t, particles, x_terminal, grad_wrt_terminal)
    return _log_p_xt_cond_x_tp1_particles(xt, x_tp1, radius)


def grad_omega_log_p_xt_given_xT(
    omega_vec: jax.Array,
    xt: jax.Array,
    t: int,
    n_t: int,
    particles: jax.Array,
) -> jax.Array:
    """Eq. (13)-(14): gradient of log p(x_t|x_T) w.r.t. omega = [r, x_Tx, x_Tz]."""
    xt_f = jax.lax.stop_gradient(xt)
    parts_f = jax.lax.stop_gradient(particles)

    def scalar_log(v: jax.Array) -> jax.Array:
        r, xT = v[0], v[1:3]
        return log_p_xt_given_xT(xt_f, t, n_t, parts_f, xT, r, grad_wrt_terminal=True)

    return jax.grad(scalar_log)(omega_vec)


def grad_omega_log_p_mt_given_xT(
    omega_vec: jax.Array,
    measurements_t: dict[str, dict[str, jax.Array]],
    xt_samples: jax.Array,
    t: int,
    n_t: int,
    particles: jax.Array,
    info_hp: InfoHyperparameters,
) -> jax.Array:
    """Eq. (11): softmax over i of (logmeas + logstate) minus softmax of logstate only."""
    xt_sg = jax.lax.stop_gradient(xt_samples)

    def unpack(v: jax.Array) -> tuple[jax.Array, jax.Array]:
        return v[0], v[1:3]

    def log_num_i(v: jax.Array, xi: jax.Array) -> jax.Array:
        r, xT = unpack(v)
        lm = logmeas_timestep(measurements_t, xi, r, info_hp)
        ls = log_p_xt_given_xT(
            xi, t, n_t, particles, xT, r, grad_wrt_terminal=True
        )
        return lm + ls

    def log_den_i(v: jax.Array, xi: jax.Array) -> jax.Array:
        r, xT = unpack(v)
        return log_p_xt_given_xT(xi, t, n_t, particles, xT, r, grad_wrt_terminal=True)

    log_nums = jax.vmap(lambda xi: log_num_i(omega_vec, xi))(xt_sg)
    log_dens = jax.vmap(lambda xi: log_den_i(omega_vec, xi))(xt_sg)
    w_num = jax.nn.softmax(log_nums)
    w_den = jax.nn.softmax(log_dens)

    jac_num = jax.vmap(
        lambda xi: jax.jacrev(lambda v: log_num_i(v, xi))(omega_vec)
    )(xt_sg)
    jac_den = jax.vmap(
        lambda xi: jax.jacrev(lambda v: log_den_i(v, xi))(omega_vec)
    )(xt_sg)
    return w_num @ jac_num - w_den @ jac_den


def pack_omega(
    geom: dict[str, dict[str, jax.Array]], traj: dict[str, dict[str, jax.Array]]
) -> jax.Array:
    r = geom["object-geom"]["radius"].reshape(())
    xT = traj["object-geom"]["position"][-1]
    return jnp.concatenate([jnp.array([r]), xT])


def observed_info(
    params: tuple[dict[str, dict[str, jax.Array]], dict[str, dict[str, jax.Array]]],
    measurements: dict[str, dict[str, jax.Array]],
    base_model: Any,
    hyperparams: InfoHyperparameters,
    sv_hp: Optional[SvObservedInfoHyperparameters] = None,
) -> tuple[jax.Array, Optional[jax.Array]]:
    """SV observed information (pdf eq. 2) with SVGD samples (sec. 2.1).

    Args:
        params: (geometry with object-geom.radius, learned object-geom.position (nT x 2)).
        measurements: per-geom stacked over time (nT, ...) as in exploration.py.
        base_model: unused in this toy (kept for schema parity with exploration.observed_info).
        hyperparams: phi / normal variances; style must be SAMPLING for this estimator.
        sv_hp: SVGD / particle counts.

    Returns:
        (3 x 3) observed information in omega = [r, x_Tx, x_Tz], and None for final-pose jac.
    """
    del base_model
    if hyperparams.style != InfoStyle.SAMPLING:
        raise NotImplementedError(
            "Toy observed_info (sv_observed_info.pdf) requires InfoStyle.SAMPLING."
        )
    sv_hp = sv_hp or SvObservedInfoHyperparameters()

    geom, traj = params
    learned = traj["object-geom"]["position"]
    n_t, _ = learned.shape
    x_terminal = learned[-1]
    radius = geom["object-geom"]["radius"]

    rng = jax.random.key(0)
    particles = sample_state_particles(learned, x_terminal, radius, rng, sv_hp)

    omega0 = pack_omega(geom, traj)

    outers = []
    for t in range(n_t):
        xt_s = particles[t]
        meas_t = {
            "spherebot1-geom": {
                "position": measurements["spherebot1-geom"]["position"][t],
                "contact_normal_W": measurements["spherebot1-geom"]["contact_normal_W"][t],
            },
            "spherebot2-geom": {
                "position": measurements["spherebot2-geom"]["position"][t],
                "contact_normal_W": measurements["spherebot2-geom"]["contact_normal_W"][t],
            },
        }
        g = grad_omega_log_p_mt_given_xT(
            omega0, meas_t, xt_s, t, n_t, particles, hyperparams
        )
        outers.append(jnp.outer(g, g))
    return jnp.sum(jnp.stack(outers, axis=0), axis=0) / float(n_t), None


def test_observed_info_ground_contact_no_spherebot_contact():
    """nT=3: object lands on ground; spherebots stay outside the sphere."""
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
    geom = {"object-geom": {"radius": r}}
    traj = {"object-geom": {"position": learned}}

    far = 5.0
    z_bot = 0.25
    measurements = {
        "spherebot1-geom": {
            "position": jnp.broadcast_to(jnp.array([far, z_bot]), (n_t, 2)),
            "contact_normal_W": jnp.zeros((n_t, 2)),
        },
        "spherebot2-geom": {
            "position": jnp.broadcast_to(jnp.array([-far, z_bot]), (n_t, 2)),
            "contact_normal_W": jnp.zeros((n_t, 2)),
        },
    }

    info_hp = InfoHyperparameters(style=InfoStyle.SAMPLING)
    sv_hp = SvObservedInfoHyperparameters(
        n_particles=200, n_svgd_iters=30, svgd_step=2e-3, init_sample_std=0.2
    )
    i_mat, jac_final = observed_info(
        (geom, traj), measurements, None, info_hp, sv_hp=sv_hp
    )
    assert jac_final is None
    assert i_mat.shape == (3, 3)
    assert jnp.all(jnp.isfinite(i_mat))
    eigvals = jnp.linalg.eigvalsh(0.5 * (i_mat + i_mat.T))
    assert jnp.all(eigvals >= -1e-5)

    # Smoke: dynamics one-step matches resting height at last time.
    x_last = learned[-1]
    assert jnp.allclose(dynamics_object(x_last, r), x_last)


if __name__ == "__main__":
    test_observed_info_ground_contact_no_spherebot_contact()
    print("test_theory_v3: ok")
