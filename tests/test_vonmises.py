#!/usr/bin/env python3

"""Test the basic functionality of the von Mises Fisher sampling"""

import random
import math


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from dair_exploration.jax_util import von_mises_sample

def test_vonmises():
    """Test 1D von Mises sampler"""

    kappa = 1.0
    num_samples = 10000

    # Python algorithm
    # s = 0.5 / kappa
    # r = s + math.sqrt(1.0 + s * s)
    # us = [random.random() for _ in range(num_samples)]
    # zs = [math.cos(math.pi * u) for u in us]
    # q = 1.0 / r
    # f = [(q + z) / (1.0 + q * z) for z in zs]
    py_samples = [random.vonmisesvariate(math.pi, kappa)-math.pi for _ in range(num_samples)]

    # Jax Algorithm
    samples = jax.vmap(von_mises_sample)(
        jax.random.split(jax.random.key(0), num_samples), jnp.ones(num_samples) * kappa
    )

    plt.hist(py_samples, bins=100, density=True, alpha=0.5, label="Python Samples")
    plt.hist(samples, bins=100, density=True, alpha=0.5, label="JAX Samples")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_vonmises()

    


