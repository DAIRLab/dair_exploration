#!/usr/bin/env python3

"""
Test theory from journal Sec 5
"""

from functools import partial
import time

import jax
import jax.numpy as jnp

import blackjax

BASEVAR = 0.01
N_SAMPLE = 100000
#N_SAMPLE_MCMC = 500

def dynamics(x):
    # Just fall to the ground
    return x.at[-1].set(0.)

def logpdf(xT, xt, var = BASEVAR):
    # Gaussian
    return -0.5 * jnp.reciprocal(var) * jnp.sum(jnp.square(xT - dynamics(xt)))

def gradloss(xT, xt, varsoft=BASEVAR):
    logpdf_grad = jax.vmap(jax.grad(logpdf), in_axes=(None, 0))(xT, xt) # (N_SAMPLE x 2)
    softmax = jax.nn.softmax(jax.vmap(logpdf, in_axes=(None, 0, None))(xT, xt, varsoft)).reshape((1, logpdf_grad.shape[0]))
    return softmax @ logpdf_grad


def test_theory():
    """Theory Test"""
    rng_key = jax.random.key(int(time.time()))
    xT = jnp.array([0.5, 0.])

    # Uniform Sampling Grad
    print("Uniform sampling")
    xt_uniform = jax.random.uniform(rng_key, shape=(N_SAMPLE, 2), minval=jnp.array([-10., 0.]), maxval=jnp.array([10., 1.]))
    gradloss_uniform = gradloss(xT, xt_uniform)
    info_uniform = -1.0 * jax.jacrev(gradloss)(xT, xt_uniform)

    # Sample MCMC
    """
    print("MCMC Sampling")
    # Build the kernel
    step_size = 1e-3
    inverse_mass_matrix = jnp.array([1., 1.])
    def logdensity(xt):
        return logpdf(xT, xt, var=2.0*BASEVAR)
    nuts = blackjax.nuts(logdensity, step_size, inverse_mass_matrix)

    # Run MCMC inference
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    rng_key, sample_key = jax.random.split(rng_key)
    states = inference_loop(sample_key, jax.jit(nuts.step), nuts.init(xT), N_SAMPLE_MCMC)
    xt_mcmc = states.position
    gradloss_mcmc = gradloss(xT, xt_mcmc, varsoft=2.0*BASEVAR)
    info_mcmc = -1.0 * jax.jacrev(gradloss)(xT, xt_mcmc, varsoft=2.0*BASEVAR)
    print(f"Grad diff: {jnp.linalg.norm(gradloss_uniform - gradloss_mcmc)}")
    """

    # Sample Normal
    print("Gaussian sampling")
    xt_normal_z = jax.random.uniform(rng_key, shape=(N_SAMPLE,), minval=0., maxval=1.)
    xt_normal_x = xT[0] + jnp.sqrt(2.0*BASEVAR) * jax.random.normal(rng_key, (N_SAMPLE,))
    xt_normal = jnp.stack([xt_normal_x, xt_normal_z]).T
    gradloss_normal = gradloss(xT, xt_normal, varsoft=2.0*BASEVAR)
    info_normal = -1.0 * jax.jacrev(partial(gradloss, varsoft=0.001*BASEVAR))(xT, xt_normal)

    breakpoint()


if __name__ == "__main__":
    test_theory()
