"""Tests for SVGD marginalization utilities."""

from __future__ import annotations

import os
import time
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import pytest

from dair_exploration.file_util import enable_jax_cache
from dair_exploration.svgd_marginalization import SVGDHyperparameters, svgd_step

# -----------------------------------------------------------------------------
# Copied from tests/test_theory_v3.py — 2D contact-style dynamics conditional.
# -----------------------------------------------------------------------------
SPEED = 1.0
BASEVAR = 0.001
PENALTY = 100.0

# Small step for 2D dynamics SVGD (large steps blow up the particle drive / kernel term).
_DYNAMICS_SVGD_STEP = 1e-3


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
        n_svgd_iters=100, svgd_step=_DYNAMICS_SVGD_STEP, init_sample_std=0.2
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
