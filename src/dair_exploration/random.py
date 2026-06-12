## Von Mises and Von Mises Fisher Distributions
## Meant to match JAX-style implementation for eventual merger.
## Based off the numpy / scipy implemenations

import math

import jax
import jax.numpy as jnp
from jax._src.random import _check_prng_key, _check_broadcast_shapes, _check_all_safe_to_cast, maybe_auto_axes, RealArray, Shape, DTypeLikeFloat
from jax._src.sharding_impls import canonicalize_sharding
from jax._src.typing import Array, ArrayLike
from jax._src.named_sharding import NamedSharding
from jax._src.partition_spec import PartitionSpec as P
from jax._src import dtypes
from jax._src.lax import lax
import numpy as np

def vonmises(key: ArrayLike,
              kappa: RealArray = np.float32(1),
              shape: Shape | None = None,
              dtype: DTypeLikeFloat | None = None,
              *,
              out_sharding: NamedSharding | P | None = None) -> Array:
  r""" Sample von Mises random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
      f(x) = \frac{1}{2*\pi I_0(\kappa)}\exp\left(\kappa\cos(x)\right)

  on the domain :math:`\pi > x > -\pi`, where :math:`\I_0(x)` is the
  zero-th order Bessel function of the first kind.

  Args:
    key: a PRNG key used as the random key.
    kappa: a float or array of floats broadcast-compatible with ``shape`` representing
      the concentration parameter of the distribution. Default 1.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. The default (None) produces a result shape equal to ``()``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: optional, specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/parallel.html>`_
      for more details.

  Returns:
    A random array with the specified dtype and with shape given by ``shape``.
  """
  key, _ = _check_prng_key("vonmises", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.inexact):
    raise ValueError(f"dtype argument to `vonmises` must be a float or complex dtype, "
                    f"got {dtype}")
  shape = _check_broadcast_shapes("vonmises", shape, kappa)
  out_sharding = canonicalize_sharding(out_sharding, "vonmises")
  _check_all_safe_to_cast("vonmises", dtype, kappa)
  return maybe_auto_axes(_vonmises, out_sharding, shape=shape, dtype=dtype)(key, kappa)

@jax.jit(static_argnums=(2, 3))
def _vonmises(key, kappa, shape, dtype) -> Array:
  kappa = lax.convert_element_type(kappa, dtype)
  kappa = jnp.broadcast_to(kappa, shape)
  # split key to match the shape of kappa
  kappa_shape = np.shape(kappa)
  split_count = math.prod(kappa_shape[key.ndim:])
  keys = key.flatten()
  keys = jax.vmap(jax.random.split, in_axes=(0, None))(keys, split_count)
  keys = keys.flatten()
  kappas = kappa.flatten()

  samples = jax.vmap(_vonmises_one)(keys, kappas)

  return jnp.reshape(samples, kappa_shape)


def _vonmises_one(key: jax.Array, kappa: jax.typing.ArrayLike):
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
    
    return jax.lax.cond(
        kappa < jnp.array(1e-8),
        small_kappa_uniform,
        lambda key, kappa: jax.lax.cond(
            kappa > 1e8, large_kappa_normal, mid_kappa_sample, key, kappa
        ),
        key,
        kappa,
    )