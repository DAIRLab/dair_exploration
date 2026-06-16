#!/usr/bin/env python3

"""
Exploration w/ Sampling and Marginalization
"""

from functools import partial
import math
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike
import numpy as np


@dataclass(frozen=True)
class SVGDHyperparameters:
    """Extra knobs for SVGD sampling + observed-info Monte Carlo size."""

    n_svgd_iters: int = 12
    svgd_step: float = 1e-2
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


def _rbf_kernel_position_grads(
    kmat: jax.Array, x_packed: jax.Array, h_sq: jax.Array
) -> jax.Array:
    """Row ``i`` is ``sum_j \\nabla_{x_i} K(x_j, x_i)``; shape ``(n_particles, feat_dim)``."""
    ni = x_packed.shape[0]

    def kernel_grad_row(i: ArrayLike) -> jax.Array:
        def k_ij(j: ArrayLike) -> jax.Array:
            diff = x_packed[j] - x_packed[i]
            return 2.0 * kmat[j, i] * diff / h_sq

        return jax.vmap(k_ij)(jnp.arange(ni)).sum(axis=0)

    return jax.vmap(kernel_grad_row)(jnp.arange(ni))


@partial(jax.jit, static_argnums=(2, 3, 4))
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
    kernels = jnp.exp(-pairwise_l2_squared / (jax.lax.stop_gradient(h_sq)))
    kernel_summands = _rbf_kernel_position_grads(
        kernels, x_packed, jax.lax.stop_gradient(h_sq)
    )
    drive = (kernels @ grad_log_p + jnp.sum(kernel_summands, axis=0)) / float(
        n_particles
    )
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
    grad_logpdf_dynamics: Callable[..., Any],
    n_particles: int,
    *,
    link_to_terminal: jax.Array | bool = False,
) -> Any:
    """Eq. 14–16: ``∇_ω log p(x_{t,i} | x_T)`` for one particle index ``i``."""

    link = jnp.asarray(link_to_terminal)

    def logp_j(j: ArrayLike) -> jax.Array:
        xt_plus_j = _tree_index_leading(xt_plus, jnp.int32(j))
        return logpdf_dynamics((xt_i, xt_plus_j))

    def grad_j(j: ArrayLike) -> Any:
        xt_plus_j = _tree_index_leading(xt_plus, jnp.int32(j))
        g_f = grad_logpdf_dynamics((xt_i, xt_plus_j), link_to_terminal=link)
        g_dyn = _tree_index_leading(g_next, jnp.int32(j))
        return jax.tree.map(jnp.add, g_f, g_dyn)

    logits = jax.vmap(logp_j)(jnp.arange(n_particles))
    weights = jax.nn.softmax(logits)
    grad_terms = jax.vmap(grad_j)(jnp.arange(n_particles))
    weighted = _softmax_weighted_sum(weights, grad_terms)
    mean_next = _tree_mean_leading(g_next)
    return jax.tree.map(jnp.subtract, weighted, mean_next)


def _is_params_pytree_node(node: Any) -> bool:
    """True at ``ω`` (parameter) nodes; false for measurement-indexing dicts."""
    if isinstance(node, jax.Array):
        return False
    if isinstance(node, tuple):
        return True
    if isinstance(node, dict):
        return "radius" in node
    return False


def _add_params_pytrees(a: Any, b: Any) -> Any:
    return jax.tree.map(jnp.add, a, b)


def _eq12_marginalize_one_meas_leaf(
    log_by_particle: jax.Array,
    grad_by_particle: Any,
    mean_dyn: Any,
) -> Any:
    """Eq. 11–12 for one scalar measurement component (independent softmax over particles)."""
    weights = jax.nn.softmax(log_by_particle)
    weighted = _softmax_weighted_sum(weights, grad_by_particle)
    return jax.tree.map(jnp.subtract, weighted, mean_dyn)


def _meas_grad_wrt_params_one_timestep(
    t: int,
    measurements: Any,
    state_particles: Any,
    g_dyn_at_t: Any,
    logpdf_meas: Callable[[Any], Any],
    grad_logpdf_meas: Callable[..., Any],
    n_particles: int,
    *,
    link_to_terminal: jax.Array | bool = False,
) -> Any:
    """Eq. 11–12: ``∇_ω log p(m_{t,k} | x_T)`` for each measurement leaf ``k``.

      ``logpdf_meas`` returns an arbitrary PyTree of scalar log terms. ``grad_logpdf_meas``
      returns the same PyTree structure; each leaf is a parameter PyTree whose array leaves
    have leading shape ``(n_particles,)`` (gradient of that measurement term for each particle).
    """

    link = jnp.asarray(link_to_terminal)
    m_t = jax.tree.map(lambda leaf: leaf[t], measurements)
    xt = jax.tree.map(lambda leaf: leaf[t], state_particles)
    mean_dyn = _tree_mean_leading(g_dyn_at_t)

    def log_meas_i(i: ArrayLike) -> Any:
        xt_i = _tree_index_leading(xt, jnp.int32(i))
        return logpdf_meas((m_t, xt_i))

    def grad_meas_i(i: ArrayLike) -> Any:
        xt_i = _tree_index_leading(xt, jnp.int32(i))
        g_m = grad_logpdf_meas((m_t, xt_i), link_to_terminal=link)
        g_d = _tree_index_leading(g_dyn_at_t, jnp.int32(i))
        return jax.tree.map(
            lambda g_term: _add_params_pytrees(g_term, g_d),
            g_m,
            is_leaf=_is_params_pytree_node,
        )

    log_by_particle = jax.vmap(log_meas_i)(jnp.arange(n_particles))
    grad_by_particle = jax.vmap(grad_meas_i)(jnp.arange(n_particles))

    def marginalize_leaf(log_leaf: jax.Array, grad_leaf: Any) -> Any:
        return _eq12_marginalize_one_meas_leaf(log_leaf, grad_leaf, mean_dyn)

    return jax.tree.map(marginalize_leaf, log_by_particle, grad_by_particle)


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
        logpdf_meas: PyTree of scalar ``log p(m_{t,k} | x_t)`` terms (any structure).
        grad_logpdf_meas: Same PyTree; each leaf is a ``params`` PyTree with per-particle
            leading axis ``(n_particles,)``.

    Returns:
        PyTree matching ``logpdf_meas``; each leaf is a ``params`` PyTree whose array
        leaves have shape ``(n_timesteps, *param_shape)``.
    """
    n_particles = jax.tree.leaves(state_particles)[0].shape[1]
    n_timesteps = jax.tree.leaves(state_particles)[0].shape[0]

    g_terminal = _tree_broadcast_params_zeros(params, n_particles)

    def backward_step(g_next: Any, t: ArrayLike) -> tuple[Any, Any]:
        t = jnp.int32(t)
        xt = jax.tree.map(lambda leaf: leaf[t], state_particles)
        xt_plus = jax.tree.map(lambda leaf: leaf[t + 1], state_particles)
        link_to_terminal = jnp.equal(t, n_timesteps - 2)
        g_t = jax.vmap(
            lambda i: _dynamics_grad_wrt_params_one_particle(
                _tree_index_leading(xt, i),
                xt_plus,
                g_next,
                logpdf_dynamics,
                grad_logpdf_dynamics,
                n_particles,
                link_to_terminal=link_to_terminal,
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

    return g_by_timestep

    # def meas_grad_at_t(t: ArrayLike) -> Any:
    #     g_dyn_t = jax.tree.map(lambda leaf: leaf[jnp.int32(t)], g_by_timestep)
    #     return _meas_grad_wrt_params_one_timestep(
    #         jnp.int32(t),
    #         measurements,
    #         state_particles,
    #         g_dyn_t,
    #         logpdf_meas,
    #         grad_logpdf_meas,
    #         n_particles,
    #         link_to_terminal=jnp.equal(jnp.int32(t), n_timesteps - 1),
    #     )

    # per_timestep = jax.vmap(meas_grad_at_t)(jnp.arange(n_timesteps))
    # return per_timestep


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
    x_final: Any,
    rng: jax.Array,
    hyperparameters: SVGDHyperparameters,
) -> Any:
    """Gaussian draws around ``initial_state_trajectory``; terminal rows fixed to ``x_T``."""
    ni = hyperparameters.n_particles
    std = hyperparameters.init_sample_std
    treedef = jax.tree.structure(initial_state_trajectory)
    flat_traj = jax.tree.leaves(initial_state_trajectory)
    flat_x_final = jax.tree.leaves(x_final)
    if len(flat_traj) != len(flat_x_final):
        raise ValueError(
            "initial_state_trajectory and x_final must have the same PyTree structure"
        )
    n_interior = flat_traj[0].shape[0]
    flat_keys = jax.random.split(rng, max(len(flat_traj), 1))

    def one_leaf(
        traj_leaf: jax.Array, terminal_leaf: jax.Array, key: jax.Array
    ) -> jax.Array:
        feat_shape = traj_leaf.shape[1:]
        noise = jax.random.normal(key, (n_interior, ni) + feat_shape) * std
        interior = traj_leaf[:, None, ...] + noise
        terminal_rows = jnp.broadcast_to(terminal_leaf[0], (ni,) + feat_shape)
        return jnp.concatenate([interior, terminal_rows[None, ...]], axis=0)

    flat_particles = [
        one_leaf(traj_leaf, x_final_leaf, key)
        for traj_leaf, x_final_leaf, key in zip(flat_traj, flat_x_final, flat_keys)
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
        # pylint: disable=cell-var-from-loop
        xt = jax.tree.map(lambda leaf: leaf[jnp.int32(t)], state_particles)
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
    x_final: Any,
    rng: jax.Array,
    hyperparameters: SVGDHyperparameters,
    logpdf_dynamics: Callable[[Any], jax.Array],
) -> Any:
    """Initialize around the MLE trajectory, then run ``n_svgd_iters`` backward SVGD sweeps."""
    particles = _initial_state_particles(
        initial_state_trajectory, x_final, rng, hyperparameters
    )
    f, gradf = _make_state_dynamics_callables(logpdf_dynamics)
    for _ in range(hyperparameters.n_svgd_iters):
        particles = _svgd_backward_sweep(particles, hyperparameters, f, gradf)
    return particles

def _outer_info_sum_over_time(grad_timesteps: Any) -> jax.Array:
    """``sum_t g_t g_t^T`` for one measurement leaf (params PyTree with time axis)."""
    n_timesteps = jax.tree.leaves(grad_timesteps)[0].shape[0]

    def vec_at(t: int) -> jax.Array:
        g_t = jax.tree.map(lambda leaf: leaf[t], grad_timesteps)
        return ravel_pytree(g_t)[0]

    return jnp.sum(
        jax.vmap(lambda t: jnp.outer(vec_at(t), vec_at(t)))(jnp.arange(n_timesteps)),
        axis=0,
    )


def _observed_info_from_grads(grads: Any) -> Any:
    """PyTree of ``(n_params, n_params)`` matrices matching the measurement log PyTree."""
    return jax.tree.map(
        _outer_info_sum_over_time,
        grads,
        is_leaf=_is_params_pytree_node,
    )


def sum_observed_info(info_by_meas: Any) -> jax.Array:
    """Add all ``(n_params, n_params)`` leaves from :func:`_observed_info_from_grads`."""
    leaves = jax.tree.leaves(info_by_meas)
    if not leaves:
        raise ValueError("observed information PyTree has no leaves")
    total = jnp.zeros_like(leaves[0])
    for leaf in leaves:
        total = total + leaf
    return total


def observed_info_svgd(
    params: tuple[Any, Any],
    measurements: Any,
    initial_state_trajectory: Any,
    logpdf_dynamics: Callable[[Any], jax.Array],
    grad_logpdf_dynamics: Callable[[Any], Any],
    logpdf_meas: Callable[[Any], Any],
    grad_logpdf_meas: Callable[[Any], Any],
    hyperparameters: SVGDHyperparameters,
    *,
    rng: jax.Array,
) -> Any:
    """
    Observed information via SVGD marginalization (``docs/sv_observed_info.pdf``).

    Parameters are ``ω = (other_params, x_T)``. Terminal state leaves ``x_T`` must have
    shape ``(1, *state_dims)``; ``initial_state_trajectory`` has shape
    ``(n_timesteps - 1, *state_dims)`` (MLE interior trajectory, no terminal row).

    Steps: Gaussian particle init → backward :func:`svgd_step` sweeps →
    :func:`grad_meas_wrt_params` → per-measurement-leaf outer products (summed over time).

    Args:
        params: ``(other_params, x_T)`` PyTree tuple.
        measurements: Per-timestep measurements, leaves ``(n_timesteps, ...)``.
        initial_state_trajectory: Interior MLE states before terminal time.
        logpdf_dynamics: ``log p(x_t | x_{t+1})``; called as ``f((x_t, x_{t+1}))``.
        grad_logpdf_dynamics: ``∇_ω f``; called as ``g((x_t, x_{t+1}))``.
        logpdf_meas: PyTree of scalar log-measurement terms; ``f((m_t, x_t))``.
        grad_logpdf_meas: Same PyTree; each leaf is ``∇_ω`` of that term (per particle).
        hyperparameters: SVGD iteration count, step, init std, and particle count.
        rng: PRNG key for initial particle noise.

    Returns:
        PyTree matching ``logpdf_meas``; each leaf is ``(n_params, n_params)``. Use
        :func:`sum_observed_info` for the total matrix.

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
sample_state_particles_svgd = _sample_state_particles_svgd
