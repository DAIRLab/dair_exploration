#!/usr/bin/env python3

"""
Exploration w/ Sampling and Marginalization
"""

import math
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
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


# Public alias
svgd_step = _svgd_step
