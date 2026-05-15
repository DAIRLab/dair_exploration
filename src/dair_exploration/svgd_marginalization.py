#!/usr/bin/env python3

"""
Exploration w/ Sampling and Marginalization
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np


@dataclass(frozen=True)
class SVGDHyperparameters:
    """Extra knobs for SVGD sampling + observed-info Monte Carlo size."""

    n_svgd_iters: int = 12
    svgd_step: float = 0.15
    init_sample_std: float = 0.08


def pack_particles(particles: Any) -> jax.Array:
    """Concatenate all leaves of a particle pytree into one feature matrix.

    Each leaf must have shape ``(n_particles, *feat_dims)``. Every leaf is reshaped to
    ``(n_particles, prod(feat_dims))`` and leaves are concatenated along axis 1 via
    ``jax.tree.reduce`` (``jax.tree.leaves`` / flatten order).

    Returns:
        Array of shape ``(n_particles, sum_i prod(feat_dims_i))``.

    The pytree structure must be static (fixed keys/nesting) for ``jax.jit``; array values
    may vary. Use :func:`unpack_particles` with an identical-structure ``template``.
    """
    reshaped = jax.tree.map(
        lambda leaf: leaf.reshape(leaf.shape[0], -1),
        particles,
    )
    return jax.tree.reduce(
        lambda a, b: jnp.concatenate([a, b], axis=1),
        reshaped,
    )


def unpack_particles(packed: jax.Array, template: Any) -> Any:
    """Inverse of :func:`pack_particles` for a fixed pytree structure.

    Per-leaf widths use ``jax.tree.map`` → ``jax.tree.leaves``; concatenated columns are split
    with ``jnp.split`` along axis 1; ``jax.tree.map`` restores shapes (no Python loop).

    Args:
        packed: Matrix from :func:`pack_particles`, shape ``(n_particles, feat_total)``.
        template: Same structure/shapes/types as the tree that was packed; values unused.

    Returns:
        Pytree with the same structure as ``template`` and leaf shapes restored.

    For ``jax.jit``, template leaf shapes must be concrete so ``sizes_arr`` / split indices are
    static at compile time.
    """
    treedef = jax.tree.structure(template)
    sizes_tree = jax.tree.map(lambda leaf: math.prod(leaf.shape[1:]), template)
    sizes_py = jax.tree.leaves(sizes_tree)
    # NumPy so split indices are compile-time constants inside ``jit`` (avoid traced cumsum).
    split_points = np.cumsum(np.array(sizes_py[:-1], dtype=np.int32))
    flat_segments = jnp.split(packed, split_points, axis=1)
    segments_tree = jax.tree.unflatten(treedef, flat_segments)
    return jax.tree.map(
        lambda leaf, flat: flat.reshape(leaf.shape).astype(leaf.dtype),
        template,
        segments_tree,
    )


def _pairwise_squared_and_bandwidth(x_packed: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Pairwise squared L2 distances and bandwidth ``h^2`` (median heuristic)."""
    assert x_packed.shape[0] >= 2, "_rbf_kernel requires at least two particles"

    pairwise_l2_squared = jax.vmap(
        lambda xi: jnp.sum(jnp.square(x_packed - jax.lax.stop_gradient(xi)), axis=-1)
    )(x_packed)

    pairwise_dist = jax.lax.stop_gradient(jnp.sqrt(pairwise_l2_squared))
    n = x_packed.shape[0]
    iu, ju = jnp.triu_indices(n, k=1)
    d_med = jnp.median(pairwise_dist[iu, ju])
    log_n = jnp.log(jnp.asarray(n, dtype=x_packed.dtype))
    h_sq = jnp.maximum(
        jnp.square(d_med) / log_n,
        jnp.asarray(1e-8, dtype=x_packed.dtype),
    )
    return pairwise_l2_squared, h_sq


def _rbf_kernel_position_grads(kmat: jax.Array, x_packed: jax.Array, h_sq: jax.Array) -> jax.Array:
    """Row ``i`` is ``sum_j \\nabla_{x_i} K(x_j, x_i)``; shape ``(n_particles, feat_dim)``."""
    ni = x_packed.shape[0]

    def kernel_grad_row(i: int) -> jax.Array:
        def k_ij(j: int) -> jax.Array:
            diff = x_packed[j] - x_packed[i]
            return kmat[j, i] * diff / h_sq

        return jax.vmap(k_ij)(jnp.arange(ni)).sum(axis=0)

    return jax.vmap(kernel_grad_row)(jnp.arange(ni))


def _svgd_step(
    x_particles: Any,
    x_plus_particles: Any,
    f: Callable[[jax.Array, jax.Array], jax.Array],
    gradf: Callable[[jax.Array, jax.Array], jax.Array],
    hyperparameters: SVGDHyperparameters,
) -> Any:
    """One SVGD step: softmax-weighted target scores plus RBF Stein terms, then unpack.

    For each packed row ``x`` (one particle),

    ``w_i \\propto \\mathrm{softmax}_i(f(x, x^+_i))``,
    ``\\nabla_x \\log p(x) \\approx \\sum_i w_i \\, \\nabla_x f(x, x^+_i)``.

    Params:
        x_particles: Pytree of leaves shaped ``(n_particles, *...)`` before packing.
        x_plus_particles: Pytree for the next timestep, same particle count convention;
            packed rows are ``x^+_i``.
        f: Scalar log-density term ``f(x, x^+)`` with ``x.shape == (n_features,)``,
            and second argument ``(n_features_plus,)``.
        gradf: ``(\\nabla_x f)(x, x^+) -> (n_features,)``.
        hyperparameters: SVGD step size ``svgd_step``.

    Returns:
        Updated particle pytree with the same structure as ``x_particles``.
    """
    x_packed = pack_particles(x_particles)
    x_plus_packed = pack_particles(x_plus_particles)
    n_particles = x_packed.shape[0]
    assert n_particles >= 2, "_svgd_step requires at least two particles"

    def grad_log_p_row(x_row: jax.Array) -> jax.Array:
        logits = jax.vmap(lambda xp: f(x_row, xp))(x_plus_packed)
        w = jax.nn.softmax(logits)
        g_rows = jax.vmap(lambda xp: gradf(x_row, xp))(x_plus_packed)
        return w @ g_rows

    grad_log_p = jax.vmap(grad_log_p_row)(x_packed)
    pairwise_l2_squared, h_sq = _pairwise_squared_and_bandwidth(x_packed)
    kernels = jnp.exp(-pairwise_l2_squared / (2.0 * jax.lax.stop_gradient(h_sq)))
    kernel_summands = _rbf_kernel_position_grads(kernels, x_packed, jax.lax.stop_gradient(h_sq))
    drive = (kernels @ grad_log_p + jnp.sum(kernel_summands, axis=0)) / float(n_particles)
    updated_packed = x_packed + hyperparameters.svgd_step * drive
    return unpack_particles(updated_packed, x_particles)


def _particles_time_slice(particles: Any, t: jax.Array | int) -> Any:
    """Slice ``particles`` at timestep ``t`` (leaves become ``(n_particles, *...)``)."""
    return jax.tree.map(lambda a: a[t], particles)


def _particles_sample_at_time(particles_t: Any, i: jax.Array | int) -> Any:
    """Particle ``i`` at a fixed timestep (leaves lose the particle axis)."""
    return jax.tree.map(lambda a: a[i], particles_t)


def _measurements_time_slice(measurements: Any, t: jax.Array | int) -> Any:
    """Per-timestep measurement subtree (``a[t]`` along leading time axis)."""
    return jax.tree.map(lambda a: a[t], measurements)


def grad_meas_wrt_params(
    omega: Any,
    particles: Any,
    measurements: Any,
    g: Callable[[Any, Any, Any], jax.Array],
    gradg: Callable[[Any, Any, Any], Any],
    f: Callable[[Any, Any, Any], jax.Array],
    gradf: Callable[[Any, Any, Any], Any],
    *,
    terminal_log: Optional[Callable[[Any, Any], jax.Array]] = None,
) -> jax.Array:
    """Per-timestep :math:`\\nabla_\\omega \\log p(m_t \\mid x_T)` (sv_observed_info.pdf eq. (11)).

    For each timestep ``t``, with particles :math:`x_{t,i}` frozen (no gradient through SVGD),

    .. math::

        \\nabla_\\omega \\log p(m_t \\mid x_T)
        \\approx
        \\sum_i \\Big(\\mathrm{softmax}_i(\\log p(m_t \\mid x_{t,i})) - \\tfrac{1}{N_i}\\Big)
        \\,\\nabla_\\omega\\big(\\log p(m_t \\mid x_{t,i}) + \\log p(x_{t,i} \\mid x_T)\\big),

    where :math:`\\nabla_\\omega \\log p(x_{t,i} \\mid x_T)` is built backward in time using
    eq. (12)–(14) with dynamics log-density ``f`` / ``gradf`` and frozen next-timestep particles.

    Args:
        omega: Parameter PyTree (e.g. geometry + terminal pose).
        particles: PyTree whose leaves have shape ``(n_timestep, n_particles, *state)``.
        measurements: PyTree whose leaves have shape ``(n_timestep, *meas)`` aligned in time.
        ``g``: scalar ``log p(m_t \\mid x_{t,i}, \\omega)`` from one timestep's measurements and
            one particle state subtree.
        ``gradg``: :math:`\\nabla_\\omega g` (PyTree matching ``omega``).
        ``f``: scalar dynamics term ``\\log p(x_{\\mathrm{curr}} \\mid x_{\\mathrm{next}}, \\omega)``
            (same role as :math:`f_\\theta` in the PDF).
        ``gradf``: :math:`\\nabla_\\omega f`.
        terminal_log: Optional ``(omega, x_{T,i}) -> scalar`` for :math:`\\log p(x_{T,i}\\mid x_T)`.
            If ``None``, the terminal state-gradient rows are treated as zero.

    Returns:
        ``(n_timestep, d_omega)`` stacked flat gradients (same flattening as ``omega``).
    """
    parts_sg = jax.tree.map(jax.lax.stop_gradient, particles)
    meas_sg = jax.tree.map(jax.lax.stop_gradient, measurements)

    lead = jax.tree.leaves(parts_sg)[0]
    n_t, n_p = int(lead.shape[0]), int(lead.shape[1])
    if n_t < 2:
        raise ValueError("grad_meas_wrt_params requires n_timestep >= 2")
    if n_p < 2:
        raise ValueError("grad_meas_wrt_params requires n_particles >= 2 (eq. (11) softmax).")

    omega_flat, _ = ravel_pytree(omega)
    d_omega = omega_flat.shape[0]

    def to_flat(g_tree: Any) -> jax.Array:
        g_flat, _ = ravel_pytree(g_tree)
        return g_flat.astype(omega_flat.dtype)

    part_terminal = _particles_time_slice(parts_sg, n_t - 1)
    if terminal_log is not None:

        def terminal_row(i: jax.Array) -> jax.Array:
            xi = _particles_sample_at_time(part_terminal, i)
            return to_flat(jax.grad(lambda om: terminal_log(om, xi))(omega))

        h_init = jax.vmap(terminal_row)(jnp.arange(n_p))
    else:
        h_init = jnp.zeros((n_p, d_omega), dtype=omega_flat.dtype)

    def eq11_at_t(h_rows: jax.Array, t: jax.Array | int) -> jax.Array:
        part_t = _particles_time_slice(parts_sg, t)
        meas_t = _measurements_time_slice(meas_sg, t)

        def log_meas_i(i: jax.Array) -> jax.Array:
            xi = _particles_sample_at_time(part_t, i)
            return g(meas_t, xi, omega)

        logits = jax.vmap(log_meas_i)(jnp.arange(n_p))
        w = jax.nn.softmax(logits) - 1.0 / float(n_p)

        def jac_row(i: jax.Array) -> jax.Array:
            xi = _particles_sample_at_time(part_t, i)
            return to_flat(gradg(meas_t, xi, omega)) + h_rows[i]

        jac = jax.vmap(jac_row)(jnp.arange(n_p))
        return jnp.einsum("i,id->d", w, jac)

    y_terminal = eq11_at_t(h_init, n_t - 1)

    def scan_body(carry: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
        h_next = carry
        part_t = _particles_time_slice(parts_sg, t)
        part_tp1 = _particles_time_slice(parts_sg, t + 1)

        def h_at_i(i: jax.Array) -> jax.Array:
            x_i = _particles_sample_at_time(part_t, i)

            def eq12_branch(_: None) -> jax.Array:
                x_tp1_0 = _particles_sample_at_time(part_tp1, 0)
                return to_flat(gradf(omega, x_i, x_tp1_0))

            def rec_branch(_: None) -> jax.Array:
                def logit_j(j: jax.Array) -> jax.Array:
                    x_next_j = _particles_sample_at_time(part_tp1, j)
                    return f(omega, x_i, x_next_j)

                logits_j = jax.vmap(logit_j)(jnp.arange(n_p))
                wj = jax.nn.softmax(logits_j) - 1.0 / float(n_p)

                def inner_flat(j: jax.Array) -> jax.Array:
                    x_next_j = _particles_sample_at_time(part_tp1, j)
                    return to_flat(gradf(omega, x_i, x_next_j)) + h_next[j]

                inner = jax.vmap(inner_flat)(jnp.arange(n_p))
                return jnp.einsum("j,jd->d", wj, inner)

            return jax.lax.cond(jnp.equal(t, n_t - 2), eq12_branch, rec_branch, None)

        h_t = jax.vmap(h_at_i)(jnp.arange(n_p))
        y_t = eq11_at_t(h_t, t)
        return h_t, y_t

    ts = jnp.arange(n_t - 2, -1, -1)
    _h_final, ys_rev = jax.lax.scan(scan_body, h_init, ts)
    ys_interior = jnp.flip(ys_rev, axis=0)
    return jnp.concatenate([ys_interior, y_terminal[None, :]], axis=0)


# Public alias
svgd_step = _svgd_step


