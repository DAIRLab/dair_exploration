#!/usr/bin/env python3

"""QP Solvers

The main contents of this file are as follows:

    * mpax solver
    * (Optional) jaxopt solver
    * (Optional) Moreau solver (via cvxpylayers)
"""
from functools import partial
from typing import Optional

import gin
import jax
import jax.numpy as jnp
from mpax import create_qp, raPDHG


## Custom Matrix sqrt
## See: https://github.com/jax-ml/jax/discussions/30120
def db_iter_sqrt(I, X, *args):  # pylint: disable=unused-argument, invalid-name
    """Denman-Beavers iteration"""
    X1i = jnp.linalg.inv(X[1])  # pylint: disable=invalid-name
    return ((0.5 * X[0] @ (I + X1i), 0.5 * (I + 0.5 * (X[1] + X1i))), None)


def sqrtm(A, s=10):
    """Sqrt of PD matrix, NOTE: not ideal for near-singular matricies"""
    return jax.lax.scan(partial(db_iter_sqrt, jnp.eye(A.shape[0])), (A, A), length=s)[
        0
    ][0]


sqrtm_jit = jax.jit(sqrtm)


def static_vars(**kwargs):
    """Static variables decorator, see
    https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
    """

    def decorate(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func

    return decorate


try:
    from jaxopt import BoxOSQP  # type: ignore

    @static_vars(solve=None, implemented=True)
    @jax.jit
    @jax.vmap
    def jit_vmap_solver_jaxopt(qp_solve: jax.Array, q_solve: jax.Array) -> jax.Array:
        """vmap-ed OSQP solver with >0 constraints"""
        return jit_vmap_solver_jaxopt.solve(
            params_obj=(qp_solve, q_solve),
            params_eq=jnp.eye(qp_solve.shape[-1]),
            params_ineq=(jnp.zeros_like(q_solve), jnp.full(q_solve.shape, jnp.inf)),
        ).params.primal[
            0
        ]  # (x, z), where Ax = z (but A == I so x == z)

    # Register optimizer argument
    gin.register(BoxOSQP, module="jaxopt")

except ImportError:

    @static_vars(solve=None, implemented=False)
    @jax.jit
    @jax.vmap
    def jit_vmap_solver_jaxopt(qp_solve: jax.Array, q_solve: jax.Array) -> jax.Array:
        """vmap-ed OSQP solver with >0 constraints"""
        raise NotImplementedError(
            "Jaxopt is not installed. Install with `pip install jaxopt`"
        )


try:
    import cvxpy as cp  # type: ignore
    from cvxpylayers.jax import CvxpyLayer  # type: ignore

    gin.register(CvxpyLayer, module="cvxpylayers")

    @static_vars(layer=None, implemented=True)
    @jax.jit
    @jax.vmap
    def jit_vmap_solver_moreau(qp_solve: jax.Array, q_solve: jax.Array) -> jax.Array:
        """vmap-ed Moreau solver with >0 constraints"""
        return jit_vmap_solver_moreau.layer(sqrtm_jit(qp_solve), q_solve)[0]

except ImportError:

    @static_vars(layer=None, implemented=False)
    @jax.jit
    @jax.vmap
    def jit_vmap_solver_moreau(qp_solve: jax.Array, q_solve: jax.Array) -> jax.Array:
        """vmap-ed Moreau solver with >0 constraints"""
        raise NotImplementedError(
            "Moreau or cvxpylayers is not installed."
            "Install with `pip install moreau[cuda13] cvxpylayers`"
        )


try:
    import moreau
    import numpy as np

    class _MoreauDirectBoxQPSolver:
        """Direct moreau active-set solver for box QPs. No cvxpylayers."""

        def __init__(self, nvar: int):
            self.nvar = nvar
            self._solvers = {}

            self._P_ro = np.arange(nvar + 1, dtype=np.int64) * nvar
            self._P_ci = np.tile(np.arange(nvar, dtype=np.int64), nvar)
            self._A_ro = np.arange(nvar + 1, dtype=np.int64)
            self._A_ci = np.arange(nvar, dtype=np.int64)
            self._cones = moreau.Cones(num_nonneg_cones=nvar)

        def _get_solver(self, batch_size):
            if batch_size not in self._solvers:
                settings = moreau.Settings(
                    solver="active_set",
                    device="cpu",
                    batch_size=batch_size,
                )
                self._solvers[batch_size] = moreau.CompiledSolver(
                    n=self.nvar, m=self.nvar,
                    P_row_offsets=self._P_ro, P_col_indices=self._P_ci,
                    A_row_offsets=self._A_ro, A_col_indices=self._A_ci,
                    cones=self._cones, settings=settings,
                )
            return self._solvers[batch_size]

        def solve(self, P_batch, q_batch):
            P_np = np.asarray(P_batch, dtype=np.float64)
            q_np = np.asarray(q_batch, dtype=np.float64)
            batch_size = P_np.shape[0]
            nvar = self.nvar
            solver = self._get_solver(batch_size)

            P_flat = np.ascontiguousarray(P_np.reshape(batch_size, -1))
            A_vals = np.broadcast_to(-np.ones(nvar), (batch_size, nvar)).copy()
            bs = np.zeros((batch_size, nvar), dtype=np.float64)

            solver.setup(P_flat, A_vals)
            return np.asarray(solver.solve(q_np, bs).x, dtype=np.float64)

    @static_vars(box_solver=None, implemented=True)
    def jit_vmap_solver_moreau_direct(
        qp_solve: jax.Array, q_solve: jax.Array
    ) -> jax.Array:
        """Direct moreau active-set solver (no cvxpylayers)."""
        P = jax.lax.stop_gradient(qp_solve)
        q = jax.lax.stop_gradient(q_solve)
        return jax.pure_callback(
            jit_vmap_solver_moreau_direct.box_solver.solve,
            jax.ShapeDtypeStruct(q.shape, jnp.float64),
            P, q,
        )

except ImportError:

    @static_vars(box_solver=None, implemented=False)
    def jit_vmap_solver_moreau_direct(
        qp_solve: jax.Array, q_solve: jax.Array
    ) -> jax.Array:
        """Direct moreau active-set solver (not installed)."""
        raise NotImplementedError("moreau is not installed.")


@static_vars(solve=None)
@jax.jit
@jax.vmap
def jit_vmap_solver_mpax(qp_solve: jax.Array, q_solve: jax.Array) -> jax.Array:
    """vmap-ed mpax solver with >0 constraints"""
    return jit_vmap_solver_mpax.solve(
        create_qp(
            Q=qp_solve,
            c=q_solve.T,
            A=jnp.zeros((0, q_solve.shape[-1])),
            b=jnp.zeros(0),
            G=jnp.eye(q_solve.shape[-1]),
            h=jnp.zeros(q_solve.shape[-1]),
            l=jnp.zeros(q_solve.shape[-1]),
            u=jnp.full((q_solve.shape[-1],), jnp.inf),
        ),
    ).primal_solution


# Register optimizer argument
gin.register(raPDHG, module="mpax")


@static_vars(first_execute=True)
def configure_solvers(nvar: Optional[int] = None) -> None:
    """Ensures all solvers are gin-configured"""

    if configure_solvers.first_execute:
        jit_vmap_solver_mpax.solve = jax.jit(raPDHG(iteration_limit=100).optimize)

        if jit_vmap_solver_jaxopt.implemented:
            jit_vmap_solver_jaxopt.solve = jax.jit(BoxOSQP().run)

        if jit_vmap_solver_moreau_direct.implemented and nvar is not None:
            jit_vmap_solver_moreau_direct.box_solver = _MoreauDirectBoxQPSolver(nvar)

        if jit_vmap_solver_moreau.implemented and nvar is not None:
            variables = cp.Variable(nvar)
            objective_matrix = cp.Parameter((nvar, nvar))
            objective_vector = cp.Parameter(nvar)
            objective = cp.Minimize(
                0.5 * cp.sum_squares(objective_matrix @ variables)
                + objective_vector.T @ variables
            )
            constraints = [variables >= 0]
            problem = cp.Problem(objective, constraints)
            jit_vmap_solver_moreau.layer = CvxpyLayer(
                problem,
                parameters=[objective_matrix, objective_vector],
                variables=[variables],
                solver="MOREAU",
                solver_args={
                    "device": "cuda",
                    "ipm_settings": {
                        "tol_gap_abs": 1e-4,
                        "tol_gap_rel": 1e-4,
                        "tol_feas": 1e-6,
                        "tol_infeas_abs": 1e-6,
                        "tol_infeas_rel": 1e-6,
                        "tol_ktratio": 1e-4,
                    },
                },
            )
