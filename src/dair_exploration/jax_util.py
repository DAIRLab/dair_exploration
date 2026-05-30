#!/usr/bin/env python3

"""Utility functions for JAX"""

import time
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from jax.experimental import checkify


def overwrite_keep_gradient(old_value: Any, new_value: Any):
    """Return new_value but keep gradient flow from old_value"""

    assert jax.tree.structure(old_value) == jax.tree.structure(new_value)

    def overwrite(old_val: Union[jax.Array, float], new_val: Union[jax.Array, float]):
        return old_val + jax.lax.stop_gradient(new_val - old_val)

    return jax.tree.map(overwrite, old_value, new_value)


def _von_mises_fisher_sample_cos(kappa: float, num_samples: int, key: jax.Array):
    """Sample from von Mises distribution with mean direction 0 and concentration kappa"""

    # Sample from uniform distribution
    uniform = jax.random.uniform(key, shape=(num_samples,), minval=0.0, maxval=1.0)

    # Inverse CDF of Z-component
    # See sec. 3.1: https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    return 1 + jnp.log(uniform + (1 - uniform) * jnp.exp(-2 * kappa)) / kappa


def von_mises_sample(key: jax.Array, kappa: jax.typing.ArrayLike):
    """Sample from von Mises distribution with mean angle 0 and concentration kappa

    Jax implementation of Numpy
    See: https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c

    checkify.check(kappa >= 0, "Kappa must be non-negative")
    checkify.check(jax.numpy.shape(kappa) == (), "Kappa must be a scalar")
    """

    def small_kappa_uniform(key: jax.Array, kappa: jax.typing.ArrayLike):
        return jax.random.uniform(
            key, shape=jax.numpy.shape(kappa), minval=-jnp.pi, maxval=jnp.pi
        )

    def large_kappa_normal(key: jax.Array, kappa: jax.typing.ArrayLike):
        return jnp.clip(
            jax.random.normal(key, shape=jax.numpy.shape(kappa)) / jnp.sqrt(kappa),
            -jnp.pi,
            jnp.pi,
        )

    def mid_kappa_sample(key: jax.Array, kappa: jax.typing.ArrayLike):

        def s_val_from_kappa(kappa: jax.typing.ArrayLike):
            r_val = 1 + jnp.sqrt(1 + 4 * kappa * kappa)
            rho_val = (r_val - jnp.sqrt(2 * r_val)) / (2 * kappa)
            return (1 + rho_val * rho_val) / (2 * rho_val)

        s_val = jax.lax.cond(
            kappa < jnp.array(1e-5), lambda kappa: 1.0 / kappa + kappa, s_val_from_kappa, kappa
        )

        def get_yw_vals(state):
            key, kappa, s_val, _, _, _ = state
            new_key, zkey, vkey = jax.random.split(key, 3)
            z_val = jnp.cos(
                jnp.pi * jax.random.uniform(zkey, shape=jax.numpy.shape(kappa))
            )
            w_val = (1 + s_val * z_val) / (s_val + z_val)
            y_val = kappa * (s_val - w_val)
            v_val = jax.random.uniform(vkey, shape=jax.numpy.shape(kappa))
            return (new_key, kappa, s_val, y_val, v_val, w_val)

        def yw_cond(state):
            _, _, _, y_val, v_val, _ = state
            cond1 = y_val * (2.0 - y_val) - v_val >= 0
            cond2 = jnp.log(y_val / v_val) + 1 - y_val >= 0
            return ~(cond1 | cond2)

        _, _, _, _, _, w_final = jax.lax.while_loop(
            yw_cond,
            get_yw_vals,
            get_yw_vals(  # Set so yw_cond returns True for the first iteration
                (key, kappa, s_val, jnp.array(0.0), jnp.array(100.0), jnp.array(0.0))
            ),
        )

        uniform_sign = 2.0 * jax.random.binomial(key, jnp.ones_like(kappa), 0.5) - 1.0
        return uniform_sign * jnp.arccos(w_final)

    # if kappa < 1e-8:
    #     return small_kappa_uniform(key, kappa)
    # elif kappa > 1e8:
    #     return large_kappa_normal(key, kappa)
    # else:
    #     return mid_kappa_sample(key, kappa)
    return jax.lax.cond(
        kappa < jnp.array(1e-8),
        small_kappa_uniform,
        lambda key, kappa: jax.lax.cond(
            kappa > 1e8, large_kappa_normal, mid_kappa_sample, key, kappa
        ),
        key,
        kappa,
    )


def von_mises_fisher_sample(
    mu: jax.Array, kappa: float, num_samples: Optional[int], key: jax.Array
):
    """Sample from von Mises-Fisher distribution with mean direction mu and concentration kappa"""

    dim = mu.shape[-1]

    if dim == 2:
        # Sample from von Mises distribution in 2D
        mean_angle = jnp.arctan2(mu[..., 1], mu[..., 0])
        cosw = _von_mises_sample_cos(kappa, num_samples, key)
        uniform_sign = 2.0 * jax.random.binomial(key, jnp.ones(num_samples), 0.5) - 1.0
        samples_angle = mean_angle + uniform_sign * jnp.arccos(cosw)
        samples = jnp.stack([jnp.cos(samples_angle), jnp.sin(samples_angle)], axis=-1)
    elif dim == 3:
        # Sample from von Mises-Fisher distribution in 3D
        # Compute X coordinate as 2D von Mises distribution
        cos_w = _von_mises_sample_cos(kappa, num_samples, key)
        sin_w = jnp.sqrt(1 - cos_w**2)
        uniform_circle = jax.random.normal(key, shape=(num_samples, 2))
        uniform_circle = uniform_circle / jnp.linalg.norm(
            uniform_circle, axis=-1, keepdims=True
        )
        samples = jnp.stack(
            [cos_w, sin_w * uniform_circle[..., 0], sin_w * uniform_circle[..., 1]],
            axis=-1,
        )
    else:
        raise NotImplementedError(
            "Only 2D and 3D von Mises-Fisher sampling is implemented"
        )
    return samples


import matplotlib.pyplot as plt
import scipy
import numpy as np


def test_vonmises_fisher_3d_sample():
    """Test von Mises-Fisher sampling in 3D"""
    key = jax.random.PRNGKey(10)
    mu = jnp.array([1.0, 0.0, 0.0])  # Mean direction in 3D
    kappa = 1.0  # Concentration parameter
    num_samples = 1000000
    samples = von_mises_fisher_sample(mu, kappa, num_samples, key)
    mean_direction = jnp.mean(samples, axis=0)
    mean_direction = mean_direction / jnp.linalg.norm(mean_direction)
    scipy_samples = scipy.stats.vonmises_fisher.rvs(
        mu, kappa, size=num_samples, random_state=10
    )
    numpy_angle_samples = np.random.vonmises(0.0, kappa, size=num_samples)
    numpy_cosangle_samples = np.cos(np.abs(numpy_angle_samples))

    ## Plot distribution of cosw vs histogram of samples
    nx_bins = 1000
    xw = jnp.linspace(-1.0, 1.0, nx_bins)
    plt.subplot(1, 2, 1)
    cosw_samples = jnp.dot(samples, mu)
    scipy_cosw_samples = jnp.dot(scipy_samples, mu)
    plt.hist(cosw_samples, bins=100, density=True, alpha=0.5, label="Samples")
    plt.hist(
        scipy_cosw_samples, bins=100, density=True, alpha=0.5, label="Scipy Samples"
    )
    plt.hist(
        numpy_cosangle_samples, bins=100, density=True, alpha=0.5, label="Numpy Samples"
    )
    wpdf = kappa / (2 * jnp.sinh(kappa)) * jnp.exp(kappa * xw)
    plt.plot(xw, wpdf, label="Von Mises-Fisher PDF")
    plt.legend()
    plt.title("Von Mises-Fisher Sampling in 3D (cosw)")
    # Plot distribution of angle vs histogram of samples
    plt.subplot(1, 2, 2)
    xangle = jnp.linspace(-jnp.pi, jnp.pi, nx_bins)
    w_samples = jnp.arccos(cosw_samples)
    scipy_w_samples = jnp.arccos(scipy_cosw_samples)
    plt.hist(w_samples, bins=100, density=True, alpha=0.5, label="Samples")
    plt.hist(scipy_w_samples, bins=100, density=True, alpha=0.5, label="Scipy Samples")
    plt.hist(
        numpy_angle_samples, bins=100, density=True, alpha=0.5, label="Numpy Samples"
    )
    plt.hist(
        np.abs(numpy_angle_samples),
        bins=100,
        density=True,
        alpha=0.5,
        label="Numpy half-pdf Samples",
    )
    wpdf_angle = 1.0 / (2 * jnp.pi * jnp.i0(kappa)) * jnp.exp(kappa * jnp.cos(xangle))
    plt.plot(jnp.arccos(xw), wpdf, label="Von Mises-Fisher PDF (arccos)")
    plt.plot(xangle, wpdf_angle, label="Von Mises-Fisher PDF (angle)")
    halfpdf = 2.0 * wpdf_angle * (xangle >= 0)  # Only positive angles are valid
    plt.plot(xangle, halfpdf, label="Von Mises-Fisher Half-PDF (angle)")
    plt.legend()
    plt.title("Von Mises-Fisher Sampling in 3D (angle)")
    plt.show()
    breakpoint()

    print("Mu:", mu)
    print("Sample mean:", mean_direction)

    angle_diffs = jnp.arccos(jnp.clip(jnp.dot(samples, mu), -1.0, 1.0))
    plt.hist(angle_diffs, bins=100, density=True, alpha=0.5, label="Samples")

    pdf = jnp.exp(kappa * jnp.cos(x)) / (2 * jnp.pi * jax.scipy.special.i0(kappa))
    halfpdf = 2.0 * pdf * (x >= 0)  # Only positive angles are valid
    plt.plot(xangle, pdf, label="Von Mises PDF")
    plt.plot(xangle, halfpdf, label="Von Mises Half-PDF")
    plt.legend()
    plt.title("Von Mises-Fisher Sampling in 3D")
    plt.show()


def test_vonmises_sample():
    """Test von Mises sampling in 2D"""
    kappa = 0.5  # Concentration parameter
    num_samples = 100000
    von_mises_sample(jax.random.key(0), kappa)
    start_time = time.time()
    print("Compilation start...", end="")
    jax.vmap(von_mises_sample)(
        jax.random.split(jax.random.key(2), num_samples), jnp.ones(num_samples) * kappa
    )
    print(f"...Done! Time taken: {time.time() - start_time:.4f}")
    start_time = time.time()
    print("Sample start...", end="")
    samples = jax.vmap(von_mises_sample)(
        jax.random.split(jax.random.key(3), num_samples), jnp.ones(num_samples) * kappa
    )
    print(f"...Done! Time taken: {time.time() - start_time:.4f}")
    print("Numpy sample start...", end="")
    numpy_samples = np.random.vonmises(0.0, kappa, size=num_samples)
    print(f"...Done! Time taken: {time.time() - start_time:.4f}")
    plt.hist(samples, bins=100, density=True, alpha=0.5, label="Samples")
    plt.hist(numpy_samples, bins=100, density=True, alpha=0.5, label="Numpy Samples")
    angles = jnp.linspace(-jnp.pi, jnp.pi, 100)
    pdf = jnp.exp(kappa * jnp.cos(angles)) / (2 * jnp.pi * jax.scipy.special.i0(kappa))
    plt.plot(angles, pdf, label="Von Mises PDF")
    plt.legend()
    plt.legend()
    plt.title("Von Mises Sampling in 2D")
    plt.show()


def test_vonmises_fisher_sample():
    """Test von Mises-Fisher sampling"""
    key = jax.random.PRNGKey(10)
    kappa = 0.1  # Concentration parameter
    num_samples = 1000000
    mu = jnp.array([1.0, 0.0])  # Mean direction in 2D
    samples = von_mises_fisher_sample(mu, kappa, num_samples, key)
    mean_direction = jnp.mean(samples, axis=0)
    mean_direction = mean_direction / jnp.linalg.norm(mean_direction)
    print("Mu:", mu)
    print("Sample mean:", mean_direction)

    ap_kappa = jax.scipy.special.i1(kappa) / jax.scipy.special.i0(kappa)
    mean_length = jnp.linalg.norm(jnp.mean(samples, axis=0))
    print("AP kappa:", ap_kappa)
    print("Mean length:", mean_length)
    # Plot 1D von Mises distribution vs. histogram of samples

    angles = jnp.arctan2(samples[..., 1], samples[..., 0])
    plt.hist(angles, bins=100, density=True, alpha=0.5, label="Samples")
    x = jnp.linspace(-jnp.pi, jnp.pi, 1000)
    pdf = jnp.exp(kappa * jnp.cos(x - jnp.arctan2(mu[1], mu[0]))) / (
        2 * jnp.pi * jax.scipy.special.i0(kappa)
    )
    plt.plot(x, pdf, label="Von Mises PDF")
    plt.legend()
    plt.title("Von Mises-Fisher Sampling in 2D")
    plt.show()


if __name__ == "__main__":
    test_vonmises_sample()
