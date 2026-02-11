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

from dair_exploration.file_util import enable_jax_cache

SPEED = 1.0
BASEVAR = 0.001
MEASVAR = 0.01
N_SAMPLE_MCMC = 2000
BATCH_SIZE = 500
N_SAMPLE_UNIFORM = 2000000


@jax.jit
def dynamics(x, speed=SPEED):
    """Just fall to the ground at constant speed"""
    return x.at[-1].subtract(speed).at[-1].max(0.0)


@jax.jit
def logpdf_old(x_final, xt, var=BASEVAR):
    """Gaussian, penalize negative start"""
    return (
        -0.5 * jnp.reciprocal(var) * jnp.sum(jnp.square(x_final - dynamics(xt)))
        - 100.0 * jnp.abs(xt.at[-1].min(0.0))[-1]
    )


@jax.jit
def logpdf(x_final, xt, var=BASEVAR):
    """ContactNets, min_{lamb>0} (x_final - (xt - SPEED + lamb))^2 + lamb*xt"""
    lamb = (SPEED - xt[-1:]).at[0].max(0.0)[0]
    return (
        -jnp.reciprocal(var)
        * (
            0.5 * jnp.square(x_final[-1] - (xt[-1] - SPEED + lamb))
            + lamb * x_final[-1]
            + jnp.square(x_final[0] - xt[0])
        )
        - 100.0 * jnp.abs(xt.at[-1].min(0.0))[-1]
    )


@jax.jit
def logmeas(mt, xt, var=MEASVAR):
    """Gaussian measurement"""
    return -0.5 * jnp.reciprocal(var) * jnp.sum(jnp.square(mt - xt))


def inference_loop(rng_key, kernel, initial_state, num_samples):
    """MCMC Inference Loop"""

    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


@jax.jit
def loglikelihood(x_final, xts, mt):
    """Analytic log likelihood"""
    return jax.scipy.special.logsumexp(
        jax.vmap(logmeas, in_axes=(None, 0))(mt, xts)
        + jax.vmap(logpdf, in_axes=(None, 0))(x_final, xts)
    ) - jax.scipy.special.logsumexp(jax.vmap(logpdf, in_axes=(None, 0))(x_final, xts))


@jax.jit
def loglikelihood_numerator(x_final, xts, mt):
    """Non-normalization term of the log likelihood"""
    return jax.scipy.special.logsumexp(
        jax.vmap(logmeas, in_axes=(None, 0))(mt, xts)
        + jax.vmap(logpdf, in_axes=(None, 0))(x_final, xts)
    )


@jax.jit
def loglikelihood_norm(x_final, xts):
    """Normalization term of the loglikelihood only"""
    return jax.scipy.special.logsumexp(
        jax.vmap(logpdf, in_axes=(None, 0))(x_final, xts)
    )


@jax.jit
def loglikelihood_grad(x_final, xts, mt, multiplier=1.0):
    """Analytic grad of the entire log likelihood"""
    logpdf_grad = jax.vmap(jax.grad(logpdf), in_axes=(None, 0))(
        x_final, xts
    )  # (N_SAMPLE x 2)
    softmax1 = jax.nn.softmax(
        jax.vmap(logmeas, in_axes=(None, 0))(mt, xts)
        + multiplier * jax.vmap(logpdf, in_axes=(None, 0))(x_final, xts)
    ).reshape((1, logpdf_grad.shape[0]))
    softmax2 = jax.nn.softmax(
        multiplier * jax.vmap(logpdf, in_axes=(None, 0))(x_final, xts)
    ).reshape((1, logpdf_grad.shape[0]))
    return (softmax1 - softmax2) @ logpdf_grad


@jax.jit
def loglikelihood_grad_num(x_final, xts, mt):
    """grad without normalization constant"""
    logpdf_grad = jax.vmap(jax.grad(logpdf), in_axes=(None, 0))(
        x_final, xts
    )  # (N_SAMPLE x 2)
    softmax = jax.nn.softmax(
        jax.vmap(logmeas, in_axes=(None, 0))(mt, xts)
        + jax.vmap(logpdf, in_axes=(None, 0))(x_final, xts)
    ).reshape((1, logpdf_grad.shape[0]))
    return (softmax) @ logpdf_grad


@jax.jit
def loglikelihood_grad_norm(x_final, xts):
    """grad of normalization constant only"""
    logpdf_grad = jax.vmap(jax.grad(logpdf), in_axes=(None, 0))(
        x_final, xts
    )  # (N_SAMPLE x 2)
    softmax = jax.nn.softmax(jax.vmap(logpdf, in_axes=(None, 0))(x_final, xts)).reshape(
        (1, logpdf_grad.shape[0])
    )
    return (softmax) @ logpdf_grad


@jax.jit
def loglikelihood_uniform_norm(x_final):
    """Sample xt, uniform in Z, Gaussian in X"""
    rng_key = jax.random.key(int(time.time()))
    xt_uniform = jax.random.uniform(
        rng_key,
        shape=(N_SAMPLE_UNIFORM, 2),
        minval=jnp.array([-10.0, -10.0]),
        maxval=jnp.array([10.0, 10.0]),
    )
    ret = jax.scipy.special.logsumexp(
        jax.vmap(logpdf, in_axes=(None, 0))(x_final, xt_uniform)
    )
    return ret


@jax.jit
def loglikelihood_uniform_num(x_final, mt):
    """Sample xt, uniform in Z, Gaussian in X"""
    rng_key = jax.random.key(int(time.time()))
    xt_uniform = jax.random.uniform(
        rng_key,
        shape=(N_SAMPLE_UNIFORM, 2),
        minval=jnp.array([-10.0, -10.0]),
        maxval=jnp.array([10.0, 10.0]),
    )
    ret = jax.scipy.special.logsumexp(
        jax.vmap(logmeas, in_axes=(None, 0))(mt, xt_uniform)
        + jax.vmap(logpdf, in_axes=(None, 0))(x_final, xt_uniform)
    )
    return ret


def test_theory():
    """Theory Test"""
    # Tests can be arbitrarily long
    # pylint: disable=too-many-locals,too-many-statements
    # Test names aren't limited
    # pylint: disable=invalid-name
    enable_jax_cache()
    rng_key = jax.random.key(int(time.time()))
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    x_final = jnp.array([0.0, 0.01])
    mt = jnp.array([0.0, 0.01 + SPEED])
    xt_uniform = jax.random.uniform(
        rng_key,
        shape=(N_SAMPLE_UNIFORM, 2),
        minval=jnp.array([-5.0, -5.0]),
        maxval=jnp.array([5.0, 5.0]),
    )

    ### TESTING p(xt | x_final) logpdf_old vs logpdf
    # def logdensityxt(x_finalval, xt_sample):
    #         return logpdf(x_finalval, xt_sample)
    # x_final_zs = 0.01*jnp.arange(6)
    # for x_final_z in x_final_zs:
    #     x_finalval = jnp.array([0., x_final_z])
    #     print(f"MCMC Warmup for {x_final_z:.2f}...", end="", flush=True)
    #     start = time.time()
    #     warmupxt = blackjax.window_adaptation(blackjax.nuts, partial(logdensityxt, x_finalval))
    #     (statext, parametersxt), _ = warmupxt.run(warmup_key, x_final, num_steps=N_SAMPLE_MCMC)
    #     print(f"Done in {time.time() - start}s")
    #     print(f"MCMC for {x_final_z:.2f}...", end="", flush=True)
    #     start = time.time()
    #     kernelxt = blackjax.nuts(partial(logdensityxt, x_finalval), **parametersxt).step
    #     statesxt = inference_loop(sample_key, kernelxt, statext, N_SAMPLE_MCMC)
    #     xt_mcmc = statesxt.position
    #     counts, bins = np.histogram(xt_mcmc[:, -1], bins=20)
    #     plt.stairs(counts, bins, label=f"x_final_z={x_final_z:.2f}")
    #     print(f"Done in {time.time() - start}s")
    # plt.legend()
    # plt.show()

    print("MCMC Warmup for xt (p2)...", end="", flush=True)
    start = time.time()

    def logdensityxt(xt_sample):
        return 0.5 * logpdf(x_final, xt_sample)

    warmupxt = blackjax.window_adaptation(blackjax.nuts, logdensityxt)
    (statext, parametersxt), _ = warmupxt.run(warmup_key, mt, num_steps=N_SAMPLE_MCMC)
    print(f"Done in {time.time() - start}s")

    print("MCMC for xt (p2)...", end="", flush=True)
    start = time.time()
    kernelxt = blackjax.nuts(logdensityxt, **parametersxt).step
    statesxt = inference_loop(sample_key, kernelxt, statext, N_SAMPLE_MCMC)
    xt_mcmc = statesxt.position
    print(f"Done in {time.time() - start}s")

    print("Plotting xt Z...")
    plt.clf()
    counts, bins = np.histogram(xt_mcmc[:, -1], bins=N_SAMPLE_MCMC // 100)
    plt.stairs(counts, bins)
    plt.show()

    # print("Plotting xt...")
    # plt.scatter(xt_uniform[:, 0], xt_uniform[:, 1])
    # plt.show()

    # print("Plotting xt X...")
    # plt.clf()
    # counts, bins = np.histogram(xt_uniform[:, 0], bins=N_SAMPLE_UNIFORM//100)
    # plt.stairs(counts, bins)
    # plt.show()

    print("MCMC Warmup for mt...", end="", flush=True)
    start = time.time()

    @jax.jit
    def logdensity(mt_sample):
        return loglikelihood(x_final, xt_uniform, mt_sample)

    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, parameters), _ = warmup.run(warmup_key, mt, num_steps=N_SAMPLE_MCMC)
    print(f"Done in {time.time() - start}s")

    print("MCMC for mt...", end="", flush=True)
    start = time.time()
    kernel = blackjax.nuts(logdensity, **parameters).step
    states = inference_loop(sample_key, kernel, state, N_SAMPLE_MCMC)
    mt_mcmc = states.position
    print(f"Done in {time.time() - start}s")

    print("MCMC for mt w/ xt_mcmc...", end="", flush=True)

    @jax.jit
    def logdensitywxt(mt_sample):
        return loglikelihood(x_final, xt_mcmc, mt_sample)

    start = time.time()
    kernel = blackjax.nuts(logdensitywxt, **parameters).step
    states = inference_loop(sample_key, kernel, state, N_SAMPLE_MCMC)
    mt_mcmc_xt_mcmc = states.position
    print(f"Done in {time.time() - start}s")

    # print("Plotting mt Z...")
    # plt.clf()
    # counts, bins = np.histogram(mt_mcmc[:, -1], bins=N_SAMPLE_MCMC//100)
    # plt.stairs(counts, bins)
    # plt.show()

    # print("Plotting mt X...")
    # plt.clf()
    # counts, bins = np.histogram(mt_mcmc[:, 0], bins=N_SAMPLE_MCMC//100)
    # plt.stairs(counts, bins)
    # plt.show()

    print("Plotting mt...")
    plt.scatter(mt_mcmc[:, 0], mt_mcmc[:, 1], label="mt_mcmc")
    plt.scatter(mt_mcmc_xt_mcmc[:, 0], mt_mcmc_xt_mcmc[:, 1], label="mt_mcmc_xt_mcmc")
    plt.legend()
    plt.show()

    print("Computing Grads...")
    grads_arr = []
    grads_arr_mcmc = []
    batch_size = BATCH_SIZE
    for idx in range(mt_mcmc.shape[0] // batch_size):
        print(f"{idx+1}/{mt_mcmc.shape[0] // batch_size}...")
        grads_arr.append(
            jax.jit(jax.vmap(jax.grad(loglikelihood), in_axes=(None, None, 0)))(
                x_final, xt_uniform, mt_mcmc[idx * batch_size : (idx + 1) * batch_size]
            )
        )
        grads_arr_mcmc.append(
            jax.jit(jax.vmap(jax.grad(loglikelihood), in_axes=(None, None, 0)))(
                x_final, xt_mcmc, mt_mcmc[idx * batch_size : (idx + 1) * batch_size]
            )
        )
    grads = jnp.concatenate(grads_arr, axis=0)
    grads_mcmc = jnp.concatenate(grads_arr_mcmc, axis=0)
    avg_grad = jnp.sum(grads / mt_mcmc.shape[0], axis=0)
    avg_grad_mcmc = jnp.sum(grads_mcmc / mt_mcmc.shape[0], axis=0)
    print(f"Average grad: {avg_grad}")
    print(f"Average grad w/ xt MCMC: {avg_grad_mcmc}")
    avg_grad_outer = jnp.sum(
        jax.vmap(jnp.outer)(grads, grads) / mt_mcmc.shape[0], axis=0
    )
    avg_grad_outer_mcmc = jnp.sum(
        jax.vmap(jnp.outer)(grads_mcmc, grads_mcmc) / mt_mcmc.shape[0], axis=0
    )
    print(f"Average grad outer product: {avg_grad_outer}")
    print(f"Average grad outer product w/ xt MCMC: {avg_grad_outer_mcmc}")

    print("Computing Grads Analytically...")
    gradsv2_arr = []
    gradsv2_arr_mcmc = []
    gradsv2_arr_mcmc_mcmc = []
    for idx in range(mt_mcmc.shape[0] // batch_size):
        print(f"{idx+1}/{mt_mcmc.shape[0] // batch_size}...")
        gradsv2_arr.append(
            jax.jit(jax.vmap(loglikelihood_grad, in_axes=(None, None, 0)))(
                x_final, xt_uniform, mt_mcmc[idx * batch_size : (idx + 1) * batch_size]
            )
        )
        gradsv2_arr_mcmc.append(
            jax.jit(
                jax.vmap(
                    partial(loglikelihood_grad, multiplier=0.5), in_axes=(None, None, 0)
                )
            )(x_final, xt_mcmc, mt_mcmc[idx * batch_size : (idx + 1) * batch_size])
        )
        gradsv2_arr_mcmc_mcmc.append(
            jax.jit(
                jax.vmap(
                    partial(loglikelihood_grad, multiplier=0.5), in_axes=(None, None, 0)
                )
            )(
                x_final,
                xt_mcmc,
                mt_mcmc_xt_mcmc[idx * batch_size : (idx + 1) * batch_size],
            )
        )
    gradsv2 = jnp.concatenate(gradsv2_arr, axis=0)
    gradsv2_mcmc = jnp.concatenate(gradsv2_arr_mcmc, axis=0)
    gradsv2_mcmc_mcmc = jnp.concatenate(gradsv2_arr_mcmc_mcmc, axis=0)
    avg_gradv2 = jnp.sum(gradsv2, axis=0) / mt_mcmc.shape[0]
    avg_gradv2_mcmc = jnp.sum(gradsv2_mcmc, axis=0) / mt_mcmc.shape[0]
    avg_gradv2_mcmc_mcmc = jnp.sum(gradsv2_mcmc_mcmc, axis=0) / mt_mcmc_xt_mcmc.shape[0]
    print(f"Average analytic grad: {avg_gradv2}")
    print(f"Average analytic grad w/ xt MCMC AND CORRECTION: {avg_gradv2_mcmc}")
    print(
        f"Average analytic grad w/ xt MCMC, \
        correction, and mt MCMC w/ xt MCMC: {avg_gradv2_mcmc_mcmc}"
    )
    avg_gradv2_outer = (
        jnp.sum(jax.vmap(jnp.outer)(gradsv2, gradsv2), axis=0) / mt_mcmc.shape[0]
    )
    avg_gradv2_outer_mcmc = (
        jnp.sum(jax.vmap(jnp.outer)(gradsv2_mcmc, gradsv2_mcmc), axis=0)
        / mt_mcmc.shape[0]
    )
    avg_gradv2_outer_mcmc_mcmc = (
        jnp.sum(jax.vmap(jnp.outer)(gradsv2_mcmc_mcmc, gradsv2_mcmc_mcmc), axis=0)
        / mt_mcmc_xt_mcmc.shape[0]
    )
    print(f"Average analytic grad outer product: {avg_gradv2_outer}")
    print(
        f"Average analytic grad outer product w/ xt MCMC AND CORRECTION: {avg_gradv2_outer_mcmc}"
    )
    print(
        f"Average analytic grad outer product w/ xt MCMC, \
        correction, and mt MCMC w/ xt MCMC: {avg_gradv2_outer_mcmc_mcmc}"
    )

    print("Computing Hessians...")
    hess_arr = []
    for idx in range(mt_mcmc.shape[0] // batch_size):
        print(f"{idx+1}/{mt_mcmc.shape[0] // batch_size}...")
        hess_arr.append(
            jax.jit(jax.vmap(jax.hessian(loglikelihood), in_axes=(None, None, 0)))(
                x_final, xt_uniform, mt_mcmc[idx * batch_size : (idx + 1) * batch_size]
            )
        )
    hess = jnp.concatenate(hess_arr, axis=0)
    avg_hess = jnp.sum(hess, axis=0) / mt_mcmc.shape[0]
    print(f"Average negative Hessian: {-avg_hess}")

    print("Computing Hess from Analytic Grad...")
    hessv2_arr = []
    for idx in range(mt_mcmc.shape[0] // batch_size):
        print(f"{idx+1}/{mt_mcmc.shape[0] // batch_size}...")
        hessv2_arr.append(
            jax.jit(jax.vmap(jax.jacrev(loglikelihood_grad), in_axes=(None, None, 0)))(
                x_final, xt_uniform, mt_mcmc[idx * batch_size : (idx + 1) * batch_size]
            )
        )
    hessv2 = jnp.concatenate(hessv2_arr, axis=0)
    avg_hessv2 = jnp.sum(hessv2, axis=0) / mt_mcmc.shape[0]
    print(f"Average negative Hessian from analytic grad: {-avg_hessv2}")

    print("Computing Hessian of numerator only...")
    hess_num_arr = []
    for idx in range(mt_mcmc.shape[0] // batch_size):
        print(f"{idx+1}/{mt_mcmc.shape[0] // batch_size}...")
        hess_num_arr.append(
            jax.jit(
                jax.vmap(jax.hessian(loglikelihood_numerator), in_axes=(None, None, 0))
            )(x_final, xt_uniform, mt_mcmc[idx * batch_size : (idx + 1) * batch_size])
        )
    hess_num = jnp.concatenate(hess_num_arr, axis=0)
    avg_hess_num = jnp.sum(hess_num, axis=0) / mt_mcmc.shape[0]
    print(f"Average negative Hessian of numerator: {-avg_hess_num}")
    print(
        f"Norm negative Hessian: {-jax.hessian(loglikelihood_norm)(x_final, xt_uniform)}"
    )

    print("Computing Grads Numerator Only...")
    grads_num_arr = []
    for idx in range(mt_mcmc.shape[0] // batch_size):
        print(f"{idx+1}/{mt_mcmc.shape[0] // batch_size}...")
        grads_num_arr.append(
            jax.jit(jax.vmap(loglikelihood_grad_num, in_axes=(None, None, 0)))(
                x_final, xt_uniform, mt_mcmc[idx * batch_size : (idx + 1) * batch_size]
            )
        )
    grads_num = jnp.concatenate(grads_num_arr, axis=0)
    avg_grad_num = jnp.sum(grads_num, axis=0) / mt_mcmc.shape[0]
    print(f"Average grad numerator: {avg_grad_num}")
    avg_grad_num_outer = (
        jnp.sum(jax.vmap(jnp.outer)(grads_num, grads_num), axis=0) / mt_mcmc.shape[0]
    )
    print(f"Average grad numerator outer product: {avg_grad_num_outer}")
    grad_norm = loglikelihood_grad_norm(x_final, xt_uniform)
    print(f"Grad Norm-only: {grad_norm}")
    print(f"Grad Norm-only outer: {jnp.outer(grad_norm, grad_norm)}")


if __name__ == "__main__":
    test_theory()
