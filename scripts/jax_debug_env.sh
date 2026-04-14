#!/usr/bin/bash

# Better debug messages
export JAX_TRACEBACK_FILTERING=off

# Remove repeated warning in 0.9.1
# See: https://github.com/jax-ml/jax/issues/36294
export TF_CPP_MIN_LOG_LEVEL=2
