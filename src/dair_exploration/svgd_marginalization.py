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

def _tree_broadcast_params_zeros(params: Any, n_particles: int) -> Any:
    """Zero gradients with leading particle axis ``(n_particles, *param_shape)``."""
    return jax.tree.map(
        lambda p: jnp.zeros((n_particles,) + jnp.shape(p), dtype=p.dtype),
        params,
    )


def _tree_index_leading(pytree: Any, index: int) -> Any:
    return jax.tree.map(lambda leaf: leaf[index], pytree)


def _tree_mean_leading(pytree: Any) -> Any:
    return jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), pytree)


def _softmax_weighted_sum(weights: jax.Array, tree: Any) -> Any:
    """``sum_i w_i * tree_i`` for a pytree whose leaves have leading axis ``n``."""
    return jax.tree.map(
        lambda leaf: jnp.sum(
            weights.reshape((weights.shape[0],) + (1,) * (leaf.ndim - 1)) * leaf,
            axis=0,
        ),
        tree,
    )


def _dynamics_grad_wrt_params_one_particle(
    xt_i: Any,
    xt_plus: Any,
    g_next: Any,
    logpdf_dynamics: Callable[[Any], jax.Array],
    grad_logpdf_dynamics: Callable[[Any], jax.Array],
    n_particles: int,
) -> Any:
    """Eq. 14–16: ``∇_ω log p(x_{t,i} | x_T)`` for one particle index ``i``."""

    def logp_j(j: int) -> jax.Array:
        xt_plus_j = _tree_index_leading(xt_plus, j)
        return logpdf_dynamics((xt_i, xt_plus_j))

    def grad_j(j: int) -> Any:
        xt_plus_j = _tree_index_leading(xt_plus, j)
        g_f = grad_logpdf_dynamics((xt_i, xt_plus_j))
        g_dyn = _tree_index_leading(g_next, j)
        return jax.tree.map(jnp.add, g_f, g_dyn)

    logits = jax.vmap(logp_j)(jnp.arange(n_particles))
    weights = jax.nn.softmax(logits)
    grad_terms = jax.vmap(grad_j)(jnp.arange(n_particles))
    weighted = _softmax_weighted_sum(weights, grad_terms)
    mean_next = _tree_mean_leading(g_next)
    return jax.tree.map(jnp.subtract, weighted, mean_next)


def _meas_grad_wrt_params_one_timestep(
    t: int,
    measurements: Any,
    state_particles: Any,
    g_dyn_at_t: Any,
    logpdf_meas: Callable[[Any], jax.Array],
    grad_logpdf_meas: Callable[[Any], jax.Array],
    n_particles: int,
) -> Any:
    """Eq. 11–12: ``∇_ω log p(m_t | x_T)`` at timestep ``t``."""

    m_t = jax.tree.map(lambda leaf: leaf[t], measurements)
    xt = jax.tree.map(lambda leaf: leaf[t], state_particles)

    def log_meas_i(i: int) -> jax.Array:
        xt_i = _tree_index_leading(xt, i)
        return logpdf_meas((m_t, xt_i))

    def grad_meas_i(i: int) -> Any:
        xt_i = _tree_index_leading(xt, i)
        g_m = grad_logpdf_meas((m_t, xt_i))
        g_d = _tree_index_leading(g_dyn_at_t, i)
        return jax.tree.map(jnp.add, g_m, g_d)

    logits = jax.vmap(log_meas_i)(jnp.arange(n_particles))
    weights = jax.nn.softmax(logits)
    grad_terms = jax.vmap(grad_meas_i)(jnp.arange(n_particles))
    weighted = _softmax_weighted_sum(weights, grad_terms)
    mean_dyn = _tree_mean_leading(g_dyn_at_t)
    return jax.tree.map(jnp.subtract, weighted, mean_dyn)


def grad_meas_wrt_params(
    params: Any,
    measurements: Any,
    state_particles: Any,
    logpdf_dynamics: Callable[[Any], jax.Array],
    grad_logpdf_dynamics: Callable[[Any], jax.Array],
    logpdf_meas: Callable[[Any], jax.Array],
    grad_logpdf_meas: Callable[[Any], jax.Array],
) -> Any:
    """
    Per-timestep ``∇_ω log p_θ(m_t | x_T)`` via Eqs. 11–12, with ``∇_ω log p_θ(x_{t,i} | x_T)``
    from the backward recursion in Eqs. 13–16.

    Dynamics callables take ``(x_t, x_{t+1})``; measurement callables take ``(m_t, x_t)``.
    All ``grad_*`` callables return a PyTree with the same structure as ``params``.

    Args:
        params: Model parameters (PyTree).
        measurements: Each leaf has shape ``(n_timesteps, n_measurements)``.
        state_particles: Each leaf has shape ``(n_timesteps, n_particles, *)``.
        logpdf_dynamics: ``log p_θ(x_t | x_{t+1})`` (proportional to ``f_θ``).
        grad_logpdf_dynamics: ``∇_ω`` of the dynamics term w.r.t. ``params``.
        logpdf_meas: ``log p_θ(m_t | x_t)``.
        grad_logpdf_meas: ``∇_ω`` of the measurement term w.r.t. ``params``.

    Returns:
        PyTree matching ``params``; each leaf has shape ``(n_timesteps, *leaf.shape)``.
    """
    n_particles = jax.tree.leaves(state_particles)[0].shape[1]
    n_timesteps = jax.tree.leaves(state_particles)[0].shape[0]

    g_terminal = _tree_broadcast_params_zeros(params, n_particles)

    def backward_step(g_next: Any, t: int) -> tuple[Any, Any]:
        xt = jax.tree.map(lambda leaf: leaf[t], state_particles)
        xt_plus = jax.tree.map(lambda leaf: leaf[t + 1], state_particles)
        g_t = jax.vmap(
            lambda i: _dynamics_grad_wrt_params_one_particle(
                _tree_index_leading(xt, i),
                xt_plus,
                g_next,
                logpdf_dynamics,
                grad_logpdf_dynamics,
                n_particles,
            )
        )(jnp.arange(n_particles))
        return g_t, g_t

    if n_timesteps == 1:
        g_by_timestep = jax.tree.map(lambda leaf: leaf[None, ...], g_terminal)
    else:
        _, g_interior = jax.lax.scan(
            backward_step,
            g_terminal,
            jnp.arange(n_timesteps - 2, -1, -1),
        )
        g_interior = jax.tree.map(lambda leaf: jnp.flip(leaf, axis=0), g_interior)
        g_terminal_row = jax.tree.map(lambda leaf: leaf[None, ...], g_terminal)
        g_by_timestep = jax.tree.map(
            lambda interior, terminal: jnp.concatenate([interior, terminal], axis=0),
            g_interior,
            g_terminal_row,
        )

    def meas_grad_at_t(t: int) -> Any:
        g_dyn_t = jax.tree.map(lambda leaf: leaf[t], g_by_timestep)
        return _meas_grad_wrt_params_one_timestep(
            t,
            measurements,
            state_particles,
            g_dyn_t,
            logpdf_meas,
            grad_logpdf_meas,
            n_particles,
        )

    per_timestep = jax.vmap(meas_grad_at_t)(jnp.arange(n_timesteps))
    return per_timestep


# Public alias
svgd_step = _svgd_step


