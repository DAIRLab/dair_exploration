#!/usr/bin/env python3

"""Uniform vs SVGD at ``x_{T-1}``: Eq. 7–8 / Eq. 16–20 gradient comparisons."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt  # noqa: E402

from dair_exploration.file_util import enable_jax_cache
from dair_exploration.svgd_marginalization import (
    SVGDHyperparameters,
    _sample_state_particles_svgd,
)
from test_svgd_marginalization import (
    BASEVAR,
    _DYNAMICS_SVGD_STEP,
    logpdf_dynamics,
    logpdf_meas_eq34,
    logpdf_meas_eq34_pytree,
    make_sensor_measurements_trajectory,
)

# Same 2D contact setup as earlier uniform-vs-SVGD experiments.
SENSOR_POSITIONS: dict[str, jax.Array] = {
    "left": jnp.array([-0.5, 1.0]),
    "right": jnp.array([0.5, 1.0]),
}
LEARNED_CENTERS = jnp.array([[0.0, 1.0], [0.0, 0.5]])
RADIUS = jnp.array([[0.5]])
X_T = LEARNED_CENTERS[-1:]
X_T_MINUS_1_MLE = LEARNED_CENTERS[0]
OMEGA = ({"radius": RADIUS}, X_T)

UNIFORM_X_BOUNDS = (-5.0, 5.0)
UNIFORM_Z_BOUNDS = (1e-4, 5.0)
N_UNIFORM_SAMPLES = 80_000

SVGD_HYPERPARAMETERS = SVGDHyperparameters(
    n_svgd_iters=160,
    svgd_step=_DYNAMICS_SVGD_STEP,
    init_sample_std=0.5,
    n_particles=128,
)


def _make_logdyn(omega: tuple):
    def logdyn(pair) -> jax.Array:
        x_t, x_tp1 = pair
        return logpdf_dynamics(x_tp1, x_t, omega[0]["radius"], var=BASEVAR)

    return logdyn


_logdyn = _make_logdyn(OMEGA)


def _build_jitted_dynamics_fns(omega: tuple):
    """JIT ``log p(x_{T-1} | x_T)`` and ``∇_ω`` for fixed ``ω`` (terminal linked in ``ω[1]``)."""

    @jax.jit
    def logpdf_at_x_curr(x_curr: jax.Array) -> jax.Array:
        return logpdf_dynamics(omega[1][0], x_curr, omega[0]["radius"], var=BASEVAR)

    @jax.jit
    def grad_logpdf_at_x_curr(x_curr: jax.Array):
        return jax.grad(
            lambda om: logpdf_dynamics(om[1][0], x_curr, om[0]["radius"], var=BASEVAR)
        )(omega)

    return logpdf_at_x_curr, grad_logpdf_at_x_curr


def _grad_to_vec(grad_omega) -> jax.Array:
    return ravel_pytree(grad_omega)[0]


def eq8_softmax_weighted_grad(
    x_samples: jax.Array,
    logpdf_at: callable,
    grad_at: callable,
) -> tuple[jax.Array, jax.Array]:
    """Eq. 8 with two passes: collect logits, then weighted sum of JIT grads."""
    n = int(x_samples.shape[0])
    log_vals = []
    for i in range(n):
        log_vals.append(logpdf_at(x_samples[i]))
    log_vals = jnp.stack(log_vals)
    weights = jax.nn.softmax(log_vals)
    grad_sum = jnp.zeros_like(_grad_to_vec(grad_at(x_samples[0])))
    for i in range(n):
        grad_sum = grad_sum + weights[i] * _grad_to_vec(grad_at(x_samples[i]))
    return grad_sum, weights


def _measurements_at_t0(measurements: dict) -> dict:
    return jax.tree.map(lambda leaf: leaf[0], measurements)


def _iter_meas_leaf_paths(log_meas: dict, prefix: tuple[str, ...] = ()):
    if isinstance(log_meas, dict):
        for key in sorted(log_meas.keys()):
            yield from _iter_meas_leaf_paths(log_meas[key], prefix + (key,))
    else:
        yield prefix


def _node_at_path(tree: dict, path: tuple[str, ...]) -> jax.Array:
    node = tree
    for key in path:
        node = node[key]
    return node


def _build_per_meas_leaf_fns(
    omega: tuple, measurements_t: dict
) -> dict[tuple[str, ...], tuple[callable, callable, callable]]:
    """Per measurement leaf: ``log p(m_k|x)``, ``log p(m_k|x)+log p(x|x_T)``, ``∇_ω`` of the sum."""

    @jax.jit
    def log_dyn_at_x_curr(x_curr: jax.Array) -> jax.Array:
        return logpdf_dynamics(
            omega[1][0], x_curr, omega[0]["radius"], var=BASEVAR
        )

    ref = logpdf_meas_eq34_pytree(
        measurements_t, X_T_MINUS_1_MLE, omega[0]["radius"]
    )
    fns: dict[tuple[str, ...], tuple[callable, callable, callable]] = {}
    for leaf_path in _iter_meas_leaf_paths(ref):

        def _make(leaf_path: tuple[str, ...]):
            @jax.jit
            def log_meas_leaf(x_curr: jax.Array) -> jax.Array:
                log_meas = logpdf_meas_eq34_pytree(
                    measurements_t, x_curr, omega[0]["radius"]
                )
                return _node_at_path(log_meas, leaf_path)

            @jax.jit
            def log_joint_leaf(x_curr: jax.Array) -> jax.Array:
                return log_meas_leaf(x_curr) + log_dyn_at_x_curr(x_curr)

            @jax.jit
            def grad_joint_leaf(x_curr: jax.Array):
                def loss(om: tuple) -> jax.Array:
                    log_meas = logpdf_meas_eq34_pytree(
                        measurements_t, x_curr, om[0]["radius"]
                    )
                    log_dyn = logpdf_dynamics(
                        om[1][0], x_curr, om[0]["radius"], var=BASEVAR
                    )
                    return _node_at_path(log_meas, leaf_path) + log_dyn

                return jax.grad(loss)(omega)

            return log_meas_leaf, log_joint_leaf, grad_joint_leaf

        fns[leaf_path] = _make(leaf_path)
    return fns


def _format_meas_leaf(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _rel_frobenius(a: jax.Array, b: jax.Array) -> float:
    den = max(float(jnp.linalg.norm(b)), 1e-12)
    return float(jnp.linalg.norm(a - b) / den)


def _per_leaf_softmax_weighted_grad(
    x_samples: jax.Array,
    leaf_fns: dict[tuple[str, ...], tuple[callable, callable, callable]],
    *,
    weight_on_meas_only: bool,
) -> dict[tuple[str, ...], jax.Array]:
    """One ``∇_ω`` vector per measurement leaf (no sum across leaves)."""
    n = int(x_samples.shape[0])
    by_leaf: dict[tuple[str, ...], jax.Array] = {}

    for leaf_path, (log_meas_leaf, log_joint_leaf, grad_joint_leaf) in sorted(
        leaf_fns.items()
    ):
        log_vals = []
        for i in range(n):
            log_w = log_meas_leaf if weight_on_meas_only else log_joint_leaf
            log_vals.append(log_w(x_samples[i]))
        weights = jax.nn.softmax(jnp.stack(log_vals))
        grad_leaf = jnp.zeros_like(_grad_to_vec(grad_joint_leaf(x_samples[0])))
        for i in range(n):
            grad_leaf = grad_leaf + weights[i] * _grad_to_vec(
                grad_joint_leaf(x_samples[i])
            )
        by_leaf[leaf_path] = grad_leaf
    return by_leaf


def eq7_per_meas_softmax_grad_by_leaf(
    x_samples: jax.Array,
    leaf_fns: dict[tuple[str, ...], tuple[callable, callable, callable]],
) -> dict[tuple[str, ...], jax.Array]:
    """Eq. 7: per ``m_k``, ``softmax_i(log p(m_k|x_i)+log p(x_i|x_T)) · ∇_ω(...)``."""
    return _per_leaf_softmax_weighted_grad(
        x_samples, leaf_fns, weight_on_meas_only=False
    )


def eq16_per_meas_softmax_grad_by_leaf(
    x_samples: jax.Array,
    leaf_fns: dict[tuple[str, ...], tuple[callable, callable, callable]],
) -> dict[tuple[str, ...], jax.Array]:
    """Eq. 16: per ``m_k``, ``softmax_i(log p(m_k|x_i)) · ∇_ω(log p(m_k|x_i)+log p(x|x_T))``."""
    return _per_leaf_softmax_weighted_grad(
        x_samples, leaf_fns, weight_on_meas_only=True
    )


# Single measurement term used for FD vs Eq. (7−8) / Eq. (16−20) checks.
FD_MEAS_LEAF: tuple[str, ...] = ("left", "contact")


def marginal_log_p_meas_leaf(
    omega: tuple,
    measurements_t: dict,
    rng: jax.Array,
    leaf_path: tuple[str, ...],
    *,
    hyperparameters: SVGDHyperparameters = SVGD_HYPERPARAMETERS,
) -> jax.Array:
    """``log p(m_k|ω) = LSE_i log p(m_k|x_i) - log N`` for one measurement component ``m_k``."""
    x_svgd = svgd_samples_x_T_minus_1(rng, omega=omega, hyperparameters=hyperparameters)
    n = int(x_svgd.shape[0])
    log_vals = []
    for i in range(n):
        log_meas = logpdf_meas_eq34_pytree(
            measurements_t, x_svgd[i], omega[0]["radius"]
        )
        log_vals.append(_node_at_path(log_meas, leaf_path))
    return jax.nn.logsumexp(jnp.stack(log_vals)) - jnp.log(float(n))


def _finite_difference_grad_log_meas_leaf(
    omega: tuple,
    measurements_t: dict,
    rng: jax.Array,
    leaf_path: tuple[str, ...],
    *,
    eps_radius: float = 1e-4,
    eps_x_T: float = 1e-3,
    hyperparameters: SVGDHyperparameters = SVGD_HYPERPARAMETERS,
) -> jax.Array:
    """Central FD on ``(r, x_T_x, x_T_z)`` for one ``m_k``; re-runs SVGD at each ``ω``."""
    vec0, unravel = ravel_pytree(omega)
    vec0 = jax.device_get(vec0)
    dim = int(vec0.size)
    eps_vec = jnp.array([eps_radius, eps_x_T, eps_x_T])[:dim]
    fd = jnp.zeros(dim)

    for k in range(dim):
        step = float(eps_vec[k])
        e = jnp.zeros(dim).at[k].set(1.0)
        omega_plus = unravel(vec0 + step * e)
        omega_minus = unravel(vec0 - step * e)
        log_plus = float(
            marginal_log_p_meas_leaf(
                omega_plus,
                measurements_t,
                jax.random.fold_in(rng, 1 + k),
                leaf_path,
                hyperparameters=hyperparameters,
            )
        )
        log_minus = float(
            marginal_log_p_meas_leaf(
                omega_minus,
                measurements_t,
                jax.random.fold_in(rng, 101 + k),
                leaf_path,
                hyperparameters=hyperparameters,
            )
        )
        fd = fd.at[k].set((log_plus - log_minus) / (2.0 * step))

    return fd


def eq20_mean_grad(
    x_samples: jax.Array,
    grad_at: callable,
) -> jax.Array:
    """Eq. 20: ``(1/N) Σ_i ∇_ω f(x_i, x_T)`` (Python loop, JIT per point)."""
    n = int(x_samples.shape[0])
    grad_sum = jnp.zeros_like(_grad_to_vec(grad_at(x_samples[0])))
    for i in range(n):
        grad_sum = grad_sum + _grad_to_vec(grad_at(x_samples[i]))
    return grad_sum / float(n)


def uniform_samples_x_T_minus_1(rng: jax.Array, n: int = N_UNIFORM_SAMPLES) -> jax.Array:
    """i.i.d. uniform on ``(x, z) in [-5, 5] x (0, 5)``."""
    kx, kz = jax.random.split(rng)
    return jnp.stack(
        [
            jax.random.uniform(kx, (n,), minval=UNIFORM_X_BOUNDS[0], maxval=UNIFORM_X_BOUNDS[1]),
            jax.random.uniform(kz, (n,), minval=UNIFORM_Z_BOUNDS[0], maxval=UNIFORM_Z_BOUNDS[1]),
        ],
        axis=1,
    )


def svgd_samples_x_T_minus_1(
    rng: jax.Array,
    *,
    omega: tuple = OMEGA,
    hyperparameters: SVGDHyperparameters = SVGD_HYPERPARAMETERS,
) -> jax.Array:
    """Particles at ``t = 0`` after backward SVGD toward fixed terminal ``x_T``."""
    particles = _sample_state_particles_svgd(
        LEARNED_CENTERS[:-1],
        omega[1],
        rng,
        hyperparameters,
        _make_logdyn(omega),
    )
    return particles[0]


def plot_x_T_minus_1_samples(
    x_uniform: jax.Array,
    x_svgd: jax.Array,
    *,
    measurements: dict | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Scatter uniform box samples and SVGD ``x_{T-1}`` particles in the plane."""
    x_u = jax.device_get(x_uniform)
    x_s = jax.device_get(x_svgd)
    mle = jax.device_get(X_T_MINUS_1_MLE)
    x_t = jax.device_get(X_T[0])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        x_u[:, 0],
        x_u[:, 1],
        s=4,
        alpha=0.15,
        c="tab:blue",
        label=f"uniform i.i.d. (n={x_u.shape[0]})",
        rasterized=True,
    )
    ax.scatter(
        x_s[:, 0],
        x_s[:, 1],
        s=28,
        alpha=0.85,
        c="tab:orange",
        edgecolors="k",
        linewidths=0.3,
        label=f"SVGD / MC (n={x_s.shape[0]})",
    )
    ax.scatter([mle[0]], [mle[1]], c="lime", s=120, marker="*", zorder=5, label="MLE $x_{T-1}$")
    ax.scatter([x_t[0]], [x_t[1]], c="k", s=80, marker="x", zorder=5, label="$x_T$")

    for name, pos in SENSOR_POSITIONS.items():
        p = jax.device_get(pos)
        ax.scatter([p[0]], [p[1]], c="red", s=60, marker="^", zorder=4)
        ax.annotate(name, (p[0], p[1]), textcoords="offset points", xytext=(4, 4), fontsize=9)

    title = (
        "Measurements built from learned trajectory (contact at $z=1$)"
        if measurements is not None
        else r"Samples of $x_{T-1}$: uniform box vs SVGD"
    )
    ax.set_title(title)
    ax.plot(
        [
            UNIFORM_X_BOUNDS[0],
            UNIFORM_X_BOUNDS[1],
            UNIFORM_X_BOUNDS[1],
            UNIFORM_X_BOUNDS[0],
            UNIFORM_X_BOUNDS[0],
        ],
        [
            UNIFORM_Z_BOUNDS[0],
            UNIFORM_Z_BOUNDS[0],
            UNIFORM_Z_BOUNDS[1],
            UNIFORM_Z_BOUNDS[1],
            UNIFORM_Z_BOUNDS[0],
        ],
        "b--",
        linewidth=1,
        alpha=0.5,
        label="uniform box",
    )
    ax.set_xlim(UNIFORM_X_BOUNDS[0] - 0.5, UNIFORM_X_BOUNDS[1] + 0.5)
    ax.set_ylim(UNIFORM_Z_BOUNDS[0], UNIFORM_Z_BOUNDS[1] + 0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def test_plot_uniform_and_svgd_x_T_minus_1_samples(tmp_path, capsys):
    """Generate both sample sets and save an overlay plot."""
    enable_jax_cache()
    rng = jax.random.key(7)
    meas_rng = jax.random.fold_in(rng, 1)
    sample_rng = jax.random.fold_in(rng, 2)
    svgd_rng = jax.random.fold_in(rng, 3)

    measurements = make_sensor_measurements_trajectory(
        LEARNED_CENTERS, RADIUS, SENSOR_POSITIONS, rng=meas_rng
    )
    x_uniform = uniform_samples_x_T_minus_1(sample_rng)
    x_svgd = svgd_samples_x_T_minus_1(svgd_rng)

    out = tmp_path / "x_T_minus_1_uniform_vs_svgd.png"
    plot_x_T_minus_1_samples(
        x_uniform,
        x_svgd,
        measurements=measurements,
        save_path=str(out),
    )

    assert out.is_file()
    with capsys.disabled():
        print(
            f"Wrote {out}\n"
            f"  uniform: {x_uniform.shape[0]} points in "
            f"x∈{UNIFORM_X_BOUNDS}, z∈{UNIFORM_Z_BOUNDS}\n"
            f"  SVGD:    {x_svgd.shape[0]} particles "
            f"(init_std={SVGD_HYPERPARAMETERS.init_sample_std}, "
            f"iters={SVGD_HYPERPARAMETERS.n_svgd_iters})"
        )


def test_eq8_vs_eq20_dynamics_grad(capsys):
    """Compare Eq. 8 (uniform, softmax-weighted) vs Eq. 20 (SVGD mean) dynamics grads."""
    enable_jax_cache()
    rng = jax.random.key(7)
    logpdf_at, grad_at = _build_jitted_dynamics_fns(OMEGA)

    # Warm up JIT
    _ = logpdf_at(X_T_MINUS_1_MLE)
    _ = grad_at(X_T_MINUS_1_MLE)

    x_uniform = uniform_samples_x_T_minus_1(jax.random.fold_in(rng, 2))
    x_svgd = svgd_samples_x_T_minus_1(jax.random.fold_in(rng, 3))

    g_eq8, w_uni = eq8_softmax_weighted_grad(x_uniform, logpdf_at, grad_at)
    g_eq20 = eq20_mean_grad(x_svgd, grad_at)

    g_eq8 = jax.device_get(g_eq8)
    g_eq20 = jax.device_get(g_eq20)
    diff = g_eq8 - g_eq20
    den = max(float(jnp.linalg.norm(g_eq20)), 1e-12)

    with capsys.disabled():
        print(
            "Dynamics grad comparison at x_{T-1} "
            "(f = log p(x_{T-1}|x_T), ω = (r, x_T)):\n"
            f"  Eq.8  softmax_U[f]·∇f  (n_uniform={x_uniform.shape[0]}): {g_eq8}\n"
            f"  Eq.20 (1/N)Σ ∇f       (n_svgd={x_svgd.shape[0]}):     {g_eq20}\n"
            f"  difference: {diff}\n"
            f"  rel Frobenius vs Eq.20: {float(jnp.linalg.norm(diff) / den):.4f}\n"
            f"  max softmax weight (uniform): {float(jnp.max(w_uni)):.6f}\n"
            f"  sum w in [-1,1]×[0.5,1.5]: "
            f"{float(jnp.sum(w_uni[(x_uniform[:,0]>-1)&(x_uniform[:,0]<1)&(x_uniform[:,1]>0.5)&(x_uniform[:,1]<1.5)])):.6f}"
        )

    # Document that they differ on mismatched supports; same support should be close.
    assert jnp.all(jnp.isfinite(g_eq8))
    assert jnp.all(jnp.isfinite(g_eq20))


def test_eq7_vs_eq16_joint_numerator_grad(capsys):
    """Compare Eq. 7 vs Eq. 16 per measurement leaf (no sum across leaves)."""
    enable_jax_cache()
    rng = jax.random.key(7)
    meas_rng = jax.random.fold_in(rng, 1)

    measurements = make_sensor_measurements_trajectory(
        LEARNED_CENTERS, RADIUS, SENSOR_POSITIONS, rng=meas_rng
    )
    m_t0 = _measurements_at_t0(measurements)
    leaf_fns = _build_per_meas_leaf_fns(OMEGA, m_t0)
    for fns in leaf_fns.values():
        _ = fns[2](X_T_MINUS_1_MLE)

    x_uniform = uniform_samples_x_T_minus_1(jax.random.fold_in(rng, 2))
    x_svgd = svgd_samples_x_T_minus_1(jax.random.fold_in(rng, 3))

    g_eq7_uniform = eq7_per_meas_softmax_grad_by_leaf(x_uniform, leaf_fns)
    g_eq16_svgd = eq16_per_meas_softmax_grad_by_leaf(x_svgd, leaf_fns)
    g_eq7_svgd = eq7_per_meas_softmax_grad_by_leaf(x_svgd, leaf_fns)

    lines = [
        "Numerator grad at x_{T-1} per measurement leaf (ω = (r, x_T)):",
        f"  n_uniform={x_uniform.shape[0]}, n_svgd={x_svgd.shape[0]}, "
        f"svgd_iters={SVGD_HYPERPARAMETERS.n_svgd_iters}",
        "",
        "  Mixed samples (Eq.7 on uniform, Eq.16 on SVGD):",
    ]
    rel_mixed: dict[tuple[str, ...], float] = {}
    rel_svgd_both: dict[tuple[str, ...], float] = {}
    for leaf_path in sorted(g_eq7_uniform.keys()):
        label = _format_meas_leaf(leaf_path)
        g7_u = jax.device_get(g_eq7_uniform[leaf_path])
        g16 = jax.device_get(g_eq16_svgd[leaf_path])
        g7_s = jax.device_get(g_eq7_svgd[leaf_path])
        rel_mixed[leaf_path] = _rel_frobenius(g7_u, g16)
        rel_svgd_both[leaf_path] = _rel_frobenius(g7_s, g16)
        lines.extend(
            [
                f"  [{label}]",
                f"    Eq.7 (uniform): {g7_u}",
                f"    Eq.16 (SVGD):   {g16}",
                f"    rel Frobenius (mixed): {rel_mixed[leaf_path]:.4f}",
                f"    Eq.7 (SVGD):    {g7_s}",
                f"    rel Frobenius (SVGD for both): {rel_svgd_both[leaf_path]:.4f}",
                "",
            ]
        )

    with capsys.disabled():
        print("\n".join(lines))

    for leaf_path in g_eq7_uniform:
        assert jnp.all(jnp.isfinite(g_eq7_uniform[leaf_path]))
        assert jnp.all(jnp.isfinite(g_eq16_svgd[leaf_path]))
        assert jnp.all(jnp.isfinite(g_eq7_svgd[leaf_path]))
        assert rel_svgd_both[leaf_path] < rel_mixed[leaf_path], (
            f"{_format_meas_leaf(leaf_path)}: expected lower rel error when SVGD "
            f"particles used for both (got {rel_svgd_both[leaf_path]:.4f} vs "
            f"mixed {rel_mixed[leaf_path]:.4f})"
        )


def test_finite_difference_grad_eq7_vs_eq16(capsys):
    """FD for one ``m_k`` vs Eq. (7−8) on uniform and Eq. (16−20) on SVGD (no sum over leaves)."""
    enable_jax_cache()
    rng = jax.random.key(7)
    meas_rng = jax.random.fold_in(rng, 1)
    fd_rng = jax.random.fold_in(rng, 9)
    leaf_path = FD_MEAS_LEAF
    leaf_label = _format_meas_leaf(leaf_path)

    measurements = make_sensor_measurements_trajectory(
        LEARNED_CENTERS, RADIUS, SENSOR_POSITIONS, rng=meas_rng
    )
    m_t0 = _measurements_at_t0(measurements)
    leaf_fns = _build_per_meas_leaf_fns(OMEGA, m_t0)
    logpdf_dyn, grad_dyn = _build_jitted_dynamics_fns(OMEGA)
    _ = logpdf_dyn(X_T_MINUS_1_MLE)
    _ = grad_dyn(X_T_MINUS_1_MLE)
    for fns in leaf_fns.values():
        _ = fns[2](X_T_MINUS_1_MLE)

    x_uniform = uniform_samples_x_T_minus_1(jax.random.fold_in(rng, 2))
    x_svgd = svgd_samples_x_T_minus_1(jax.random.fold_in(rng, 3))

    log_p0 = float(
        marginal_log_p_meas_leaf(
            OMEGA,
            m_t0,
            jax.random.fold_in(fd_rng, 0),
            leaf_path,
            hyperparameters=SVGD_HYPERPARAMETERS,
        )
    )
    g_fd = _finite_difference_grad_log_meas_leaf(
        OMEGA, m_t0, fd_rng, leaf_path, hyperparameters=SVGD_HYPERPARAMETERS
    )

    g_eq7_u = eq7_per_meas_softmax_grad_by_leaf(x_uniform, leaf_fns)[leaf_path]
    g_eq8_u, _ = eq8_softmax_weighted_grad(x_uniform, logpdf_dyn, grad_dyn)
    g_7_minus_8_u = g_eq7_u - g_eq8_u

    g_eq16 = eq16_per_meas_softmax_grad_by_leaf(x_svgd, leaf_fns)[leaf_path]
    g_eq20 = eq20_mean_grad(x_svgd, grad_dyn)
    g_16_minus_20 = g_eq16 - g_eq20

    g_eq7_s = eq7_per_meas_softmax_grad_by_leaf(x_svgd, leaf_fns)[leaf_path]
    g_eq8_s, _ = eq8_softmax_weighted_grad(x_svgd, logpdf_dyn, grad_dyn)
    g_7_minus_8_s = g_eq7_s - g_eq8_s

    g_fd = jax.device_get(g_fd)
    g_7_minus_8_u = jax.device_get(g_7_minus_8_u)
    g_16_minus_20 = jax.device_get(g_16_minus_20)
    g_7_minus_8_s = jax.device_get(g_7_minus_8_s)

    rel_7m8_u = _rel_frobenius(g_7_minus_8_u, g_fd)
    rel_16m20 = _rel_frobenius(g_16_minus_20, g_fd)
    rel_7m8_s = _rel_frobenius(g_7_minus_8_s, g_fd)
    winner = min(
        [
            ("Eq.7−Eq.8 uniform", rel_7m8_u),
            ("Eq.16−Eq.20 SVGD", rel_16m20),
            ("Eq.7−Eq.8 SVGD", rel_7m8_s),
        ],
        key=lambda item: item[1],
    )

    lines = [
        f"Finite-difference ∇_ω log p({leaf_label}|ω)  "
        "(LSE over SVGD x_{{T-1}}, re-sample per ω perturbation):",
        f"  log p({leaf_label}|ω) @ nominal ω: {log_p0:.6f}",
        f"  FD grad (r, x_T_x, x_T_z): {g_fd}",
        "",
        f"  Eq.7−Eq.8 [{leaf_label}] (uniform, n={x_uniform.shape[0]}): "
        f"{g_7_minus_8_u}  rel vs FD: {rel_7m8_u:.4f}",
        f"  Eq.16−Eq.20 [{leaf_label}] (SVGD, n={x_svgd.shape[0]}): "
        f"{g_16_minus_20}  rel vs FD: {rel_16m20:.4f}",
        f"  Eq.7−Eq.8 [{leaf_label}] (SVGD, n={x_svgd.shape[0]}): "
        f"{g_7_minus_8_s}  rel vs FD: {rel_7m8_s:.4f}",
        "",
        f"  Closest to FD: {winner[0]} (rel={winner[1]:.4f})",
    ]
    with capsys.disabled():
        print("\n".join(lines))

    assert jnp.all(jnp.isfinite(g_fd))
    assert jnp.all(jnp.isfinite(g_7_minus_8_u))
    assert jnp.all(jnp.isfinite(g_16_minus_20))


@pytest.mark.skipif(
    not os.environ.get("DAIR_UNIFORM_VISUALIZE"),
    reason="Set DAIR_UNIFORM_VISUALIZE=1 to open the matplotlib window.",
)
def test_plot_uniform_and_svgd_x_T_minus_1_interactive():
    """Same plot, displayed interactively when ``DAIR_UNIFORM_VISUALIZE=1``."""
    enable_jax_cache()
    rng = jax.random.key(7)
    x_uniform = uniform_samples_x_T_minus_1(jax.random.fold_in(rng, 2))
    x_svgd = svgd_samples_x_T_minus_1(jax.random.fold_in(rng, 3))
    plot_x_T_minus_1_samples(x_uniform, x_svgd, show=True)
