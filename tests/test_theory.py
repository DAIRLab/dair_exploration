#!/usr/bin/env python3

"""
Test theory from journal Sec 5
"""

from functools import partial
import time

import jax
import jax.numpy as jnp

import blackjax

# plotting
import numpy as np
import matplotlib.pyplot as plt

SPEED = 1.0
BASEVAR = 0.001
MEASVAR = 1.
N_SAMPLE_MCMC = 5000

def dynamics(x, speed=SPEED):
    # Just fall to the ground at constant speed
    return x.at[-1].subtract(speed).at[-1].max(0.)

def logpdf(xT, xt, var = BASEVAR):
    # Gaussian, penalize negative start
    return -0.5 * jnp.reciprocal(var) * jnp.sum(jnp.square(xT - dynamics(xt))) - 100.0 * jnp.abs(xt.at[-1].min(0.))[-1] #- 100.0 * jnp.abs(xT.at[-1].min(0.))[-1]

def logmeas(mt, xt, var = MEASVAR):
    # Gaussian measurement
    return -0.5 * jnp.reciprocal(var)* jnp.sum(jnp.square(mt - xt))

def gradloss(xT, xt, mt, mult=1.0):
    logpdf_grad = jax.vmap(jax.grad(logpdf), in_axes=(None, 0))(xT, xt) # (N_SAMPLE x 2)
    softmax1 = jax.nn.softmax(jax.vmap(logmeas, in_axes=(None, 0))(mt, xt) + mult * jax.vmap(logpdf, in_axes=(None, 0))(xT, xt)).reshape((1, logpdf_grad.shape[0]))
    softmax2 = jax.nn.softmax(mult * jax.vmap(logpdf, in_axes=(None, 0))(xT, xt)).reshape((1, logpdf_grad.shape[0]))
    return (softmax1 - softmax2) @ logpdf_grad

def gradloss_normterm(xT, xt, mult=1.0):
    logpdf_grad = jax.vmap(jax.grad(logpdf), in_axes=(None, 0))(xT, xt) # (N_SAMPLE x 2)
    softmax = jax.nn.softmax(mult * jax.vmap(logpdf, in_axes=(None, 0))(xT, xt)).reshape((1, logpdf_grad.shape[0]))
    return softmax @ logpdf_grad

def gradloss_numerator(xT, xt, mt, mult=1.0):
    logpdf_grad = jax.vmap(jax.grad(logpdf), in_axes=(None, 0))(xT, xt) # (N_SAMPLE x 2)
    softmax = jax.nn.softmax(jax.vmap(logmeas, in_axes=(None, 0))(mt, xt) + mult * jax.vmap(logpdf, in_axes=(None, 0))(xT, xt)).reshape((1, logpdf_grad.shape[0]))
    return softmax @ logpdf_grad

# MCMC Inference Loop
def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

def test_theory():
    """Theory Test"""
    rng_key = jax.random.key(int(time.time()))
    xT = jnp.array([0.5, 0.])

    # See https://blackjax-devs.github.io/blackjax/examples/quickstart.html
    print("MCMC Warmup for xt...", end="", flush=True)
    start = time.time()
    def logdensityxt(xt):
        return logpdf(xT, xt)
    def logdensityxt2(xt):
        return 0.5 * logpdf(xT, xt)
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensityxt)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    initial_position = xT
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=N_SAMPLE_MCMC)
    print(f"Done in {time.time() - start}s")

    print("MCMC for xt...", end="", flush=True)
    start = time.time()
    kernel = blackjax.nuts(logdensityxt, **parameters).step
    states = inference_loop(sample_key, kernel, state, N_SAMPLE_MCMC)
    xt_mcmc = states.position
    print(f"Done in {time.time() - start}s")

    print("Plotting xt Z...")
    counts, bins = np.histogram(xt_mcmc[:, -1], bins=N_SAMPLE_MCMC//100)
    plt.stairs(counts, bins)
    plt.title("xt, Z")
    plt.show()

    print("MCMC for xt, p2...", end="", flush=True)
    start = time.time()
    kernel = blackjax.nuts(logdensityxt2, **parameters).step
    states = inference_loop(sample_key, kernel, state, N_SAMPLE_MCMC)
    xt2_mcmc = states.position
    print(f"Done in {time.time() - start}s")

    print("Plotting xt, p2 Z...")
    plt.clf()
    counts, bins = np.histogram(xt2_mcmc[:, -1], bins=N_SAMPLE_MCMC//100)
    plt.stairs(counts, bins)
    plt.title("xt, p2 Z")
    plt.show()

    print("MCMC Warmup for mt...", end="", flush=True)
    start = time.time()
    def logdensitymt(mt):
        return jax.scipy.special.logsumexp(jax.vmap(logmeas, in_axes=(None, 0))(mt, xt_mcmc))
    warmupmt = blackjax.window_adaptation(blackjax.nuts, logdensitymt)
    (statemt, parametersmt), _ = warmupmt.run(warmup_key, initial_position, num_steps=N_SAMPLE_MCMC)
    print(f"Done in {time.time() - start}s")

    print("MCMC for mt...", end="", flush=True)
    start = time.time()
    kernelmt = blackjax.nuts(logdensitymt, **parametersmt).step
    statesmt = inference_loop(sample_key, kernelmt, statemt, N_SAMPLE_MCMC)
    mt_mcmc = statesmt.position
    print(f"Done in {time.time() - start}s")

    print("Plotting mt Z...")
    plt.clf()
    counts, bins = np.histogram(mt_mcmc[:, -1], bins=N_SAMPLE_MCMC//100)
    plt.stairs(counts, bins)
    plt.show(block=True)

    grads = jax.vmap(partial(gradloss, mult=0.5), in_axes=(None, None, 0))(xT, xt_mcmc, mt_mcmc)
    print(f"Expected grad loss: {jnp.sum(grads, axis=0) / N_SAMPLE_MCMC}")
    print(f"Expected info: {jnp.sum(jax.vmap(jnp.outer)(grads, grads), axis=0) / N_SAMPLE_MCMC}")

    grads_norm = gradloss_normterm(xT, xt_mcmc)
    print(f"Expected grad normalization: {jnp.sum(grads_norm, axis=0) / N_SAMPLE_MCMC}")
    print(f"Expected grad norm sq: {jnp.sum(jax.vmap(jnp.outer)(grads_norm, grads_norm), axis=0) / N_SAMPLE_MCMC}")

    grads_num = jax.vmap(partial(gradloss_numerator, mult=0.5), in_axes=(None, None, 0))(xT, xt_mcmc, mt_mcmc)
    print(f"Expected grad numerator: {jnp.sum(grads_norm, axis=0) / N_SAMPLE_MCMC}")
    print(f"Expected grad norm sq numerator: {jnp.sum(jax.vmap(jnp.outer)(grads_norm, grads_norm), axis=0) / N_SAMPLE_MCMC}")

    breakpoint()

    # Uniform Sampling Grad
    #print("Uniform sampling")
    #xt_uniform = jax.random.uniform(rng_key, shape=(N_SAMPLE, 2), minval=jnp.array([-5., 0.]), maxval=jnp.array([5., 5.]))
    #gradloss_uniform = gradloss(xT, xt_uniform)
    #breakpoint()

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
    

    rng_key, sample_key = jax.random.split(rng_key)
    states = inference_loop(sample_key, jax.jit(nuts.step), nuts.init(xT), N_SAMPLE_MCMC)
    xt_mcmc = states.position
    gradloss_mcmc = gradloss(xT, xt_mcmc, varsoft=2.0*BASEVAR)
    info_mcmc = -1.0 * jax.jacrev(gradloss)(xT, xt_mcmc, varsoft=2.0*BASEVAR)
    print(f"Grad diff: {jnp.linalg.norm(gradloss_uniform - gradloss_mcmc)}")
    """

    # Sample Normal
    #print("Gaussian sampling")
    #xt_normal_z = jax.random.uniform(rng_key, shape=(N_SAMPLE_NORMAL,), minval=0., maxval=1.)
    #xt_normal_x = xT[0] + jnp.sqrt(2.0*BASEVAR) * jax.random.normal(rng_key, (N_SAMPLE_NORMAL,))
    #xt_normal = jnp.stack([xt_normal_x, xt_normal_z]).T

    #gradloss_normal = gradloss(xT, xt_normal, mult=0.5)
    #info_normal = -1.0 * jax.jacrev(gradloss)(xT, xt_normal, mult=0.5)

    #breakpoint()


if __name__ == "__main__":
    test_theory()
