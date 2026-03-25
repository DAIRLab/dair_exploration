#!/usr/bin/env python3

"""QP Solvers

The main contents of this file are as follows:

    * mpax solver
    * (Optional) jaxopt solver
    * (Optional) Moreau solver (via cvxpylayers)
"""
import gin
import jax
import jax.numpy as jnp
from mpax import create_qp, raPDHG


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
def configure_solvers() -> None:
    """Ensures all solvers are gin-configured"""

    if configure_solvers.first_execute:
        jit_vmap_solver_mpax.solve = jax.jit(raPDHG().optimize)

        if jit_vmap_solver_jaxopt.implemented:
            jit_vmap_solver_jaxopt.solve = jax.jit(BoxOSQP().run)
