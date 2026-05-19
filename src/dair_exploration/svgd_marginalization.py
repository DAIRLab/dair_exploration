#!/usr/bin/env python3

"""
Exploration w/ Sampling and Marginalization
"""

import math
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np


@dataclass(frozen=True)
class SVGDHyperparameters:
    """Extra knobs for SVGD sampling + observed-info Monte Carlo size."""

    n_svgd_iters: int = 12
    svgd_step: float = 0.15
    init_sample_std: float = 0.5
    n_particles: int = 48


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
    grad_logpdf_dynamics: Callable[[Any], Any],
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
    grad_logpdf_meas: Callable[[Any], Any],
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
    grad_logpdf_dynamics: Callable[[Any], Any],
    logpdf_meas: Callable[[Any], jax.Array],
    grad_logpdf_meas: Callable[[Any], Any],
) -> Any:
    """
    Per-timestep ``∇_ω log p_θ(m_t | x_T)`` via Eqs. 11–12, with ``∇_ω log p_θ(x_{t,i} | x_T)``
    from the backward recursion in Eqs. 13–16.

    Dynamics callables take ``(x_t, x_{t+1})``; measurement callables take ``(m_t, x_t)``.
    All ``grad_*`` callables return a PyTree with the same structure as ``params``.
    Use :func:`functools.partial` at the call site to bind hyperparameters or ``params``.

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


def _make_state_dynamics_callables(
    logpdf_dynamics: Callable[[Any], jax.Array],
) -> tuple[
    Callable[[jax.Array, jax.Array], jax.Array],
    Callable[[jax.Array, jax.Array], jax.Array],
]:
    """``f`` and ``∇_{x_t} f`` for :func:`svgd_step` from the dynamics log-density."""

    def f(x_curr: jax.Array, x_next: jax.Array) -> jax.Array:
        return logpdf_dynamics((x_curr, x_next))

    def gradf(x_curr: jax.Array, x_next: jax.Array) -> jax.Array:
        return jax.grad(lambda xc: logpdf_dynamics((xc, x_next)))(x_curr)

    return f, gradf


def _initial_state_particles(
    initial_state_trajectory: Any,
    x_T: Any,
    rng: jax.Array,
    hyperparameters: SVGDHyperparameters,
) -> Any:
    """Gaussian draws around ``initial_state_trajectory``; terminal rows fixed to ``x_T``."""
    ni = hyperparameters.n_particles
    std = hyperparameters.init_sample_std
    treedef = jax.tree.structure(initial_state_trajectory)
    flat_traj = jax.tree.leaves(initial_state_trajectory)
    flat_xT = jax.tree.leaves(x_T)
    if len(flat_traj) != len(flat_xT):
        raise ValueError("initial_state_trajectory and x_T must have the same PyTree structure")
    n_interior = flat_traj[0].shape[0]
    flat_keys = jax.random.split(rng, max(len(flat_traj), 1))

    def one_leaf(traj_leaf: jax.Array, terminal_leaf: jax.Array, key: jax.Array) -> jax.Array:
        feat_shape = traj_leaf.shape[1:]
        noise = jax.random.normal(key, (n_interior, ni) + feat_shape) * std
        interior = traj_leaf[:, None, ...] + noise
        terminal_rows = jnp.broadcast_to(terminal_leaf[0], (ni,) + feat_shape)
        return jnp.concatenate([interior, terminal_rows[None, ...]], axis=0)

    flat_particles = [
        one_leaf(traj_leaf, xT_leaf, key)
        for traj_leaf, xT_leaf, key in zip(flat_traj, flat_xT, flat_keys)
    ]
    return jax.tree.unflatten(treedef, flat_particles)


def _svgd_backward_sweep(
    state_particles: Any,
    hyperparameters: SVGDHyperparameters,
    f: Callable[[jax.Array, jax.Array], jax.Array],
    gradf: Callable[[jax.Array, jax.Array], jax.Array],
) -> Any:
    """One backward pass of :func:`svgd_step` from ``T-2`` down to ``0``."""
    n_timesteps = jax.tree.leaves(state_particles)[0].shape[0]
    if n_timesteps <= 1:
        return state_particles
    terminal = jax.tree.map(lambda leaf: leaf[-1], state_particles)
    updated_tail = terminal
    interior: list[Any] = []
    for t in range(n_timesteps - 2, -1, -1):
        xt = jax.tree.map(lambda leaf: leaf[t], state_particles)
        updated_tail = svgd_step(xt, updated_tail, f, gradf, hyperparameters)
        interior.append(updated_tail)
    interior.reverse()
    stacked = jax.tree.map(
        lambda *rows: jnp.stack(rows, axis=0),
        *interior,
        terminal,
    )
    return stacked


def _sample_state_particles_svgd(
    initial_state_trajectory: Any,
    x_T: Any,
    rng: jax.Array,
    hyperparameters: SVGDHyperparameters,
    logpdf_dynamics: Callable[[Any], jax.Array],
) -> Any:
    """Initialize around the MLE trajectory, then run ``n_svgd_iters`` backward SVGD sweeps."""
    particles = _initial_state_particles(
        initial_state_trajectory, x_T, rng, hyperparameters
    )
    f, gradf = _make_state_dynamics_callables(logpdf_dynamics)
    for _ in range(hyperparameters.n_svgd_iters):
        particles = _svgd_backward_sweep(particles, hyperparameters, f, gradf)
    return particles


def _observed_info_from_grads(grads_per_timestep: Any) -> jax.Array:
    """Sum of outer products of flattened per-timestep parameter gradients."""
    n_timesteps = jax.tree.leaves(grads_per_timestep)[0].shape[0]

    def grad_vector_at_t(t: int) -> jax.Array:
        g_t = jax.tree.map(lambda leaf: leaf[t], grads_per_timestep)
        return ravel_pytree(g_t)[0]

    grad_vectors = jax.vmap(grad_vector_at_t)(jnp.arange(n_timesteps))
    return jnp.sum(jax.vmap(jnp.outer)(grad_vectors, grad_vectors), axis=0)


def observed_info_svgd(
    params: tuple[Any, Any],
    measurements: Any,
    initial_state_trajectory: Any,
    logpdf_dynamics: Callable[[Any], jax.Array],
    grad_logpdf_dynamics: Callable[[Any], Any],
    logpdf_meas: Callable[[Any], jax.Array],
    grad_logpdf_meas: Callable[[Any], Any],
    hyperparameters: SVGDHyperparameters,
    *,
    rng: jax.Array,
) -> jax.Array:
    """
    Observed information via SVGD marginalization (``docs/sv_observed_info.pdf``).

    Parameters are ``ω = (other_params, x_T)``. Terminal state leaves ``x_T`` must have
    shape ``(1, *state_dims)``; ``initial_state_trajectory`` has shape
    ``(n_timesteps - 1, *state_dims)`` (MLE interior trajectory, no terminal row).

    Steps: Gaussian particle init → backward :func:`svgd_step` sweeps →
    :func:`grad_meas_wrt_params` → sum of flattened gradient outer products.

    Args:
        params: ``(other_params, x_T)`` PyTree tuple.
        measurements: Per-timestep measurements, leaves ``(n_timesteps, ...)``.
        initial_state_trajectory: Interior MLE states before terminal time.
        logpdf_dynamics: ``log p(x_t | x_{t+1})``; called as ``f((x_t, x_{t+1}))``.
        grad_logpdf_dynamics: ``∇_ω f``; called as ``g((x_t, x_{t+1}))``.
        logpdf_meas: ``log p(m_t | x_t)``; called as ``f((m_t, x_t))``.
        grad_logpdf_meas: ``∇_ω`` of measurement term.
        hyperparameters: SVGD iteration count, step, init std, and particle count.
        rng: PRNG key for initial particle noise.

    Returns:
        ``(n_params, n_params)`` observed information matrix.

    Bind extra arguments (e.g. ``params=ω``) with :func:`functools.partial` before calling.
    """
    state_particles = _sample_state_particles_svgd(
        initial_state_trajectory,
        params[1],
        rng,
        hyperparameters,
        logpdf_dynamics,
    )
    grads = grad_meas_wrt_params(
        params,
        measurements,
        state_particles,
        logpdf_dynamics,
        grad_logpdf_dynamics,
        logpdf_meas,
        grad_logpdf_meas,
    )
    return _observed_info_from_grads(grads)


# Public alias
svgd_step = _svgd_step


