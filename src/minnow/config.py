from collections.abc import Sequence
import functools

import numpy as np
import scipy as sp

import jax
import jax.numpy
import jax.scipy
import jax.tree_util

def config(**kwargs):
    global jnp, jsp, jnparray, jnpzeros, intarray, jnpkey, jnpsplit, jnpnormal
    global matrix_factor, matrix_solve, matrix_norm

    np.logdet = lambda a: np.sum(np.log(np.abs(a)))
    jax.numpy.logdet = lambda a: jax.numpy.sum(jax.numpy.log(jax.numpy.abs(a)))

    backend = kwargs.get('backend')

    if backend == 'numpy':
        jnp, jsp = np, sp

        jnparray = lambda a: np.array(a, dtype=np.float64)
        jnpzeros = lambda a: np.zeros(a, dtype=np.float64)
        intarray = lambda a: np.array(a, dtype=np.int64)

        jnpkey    = lambda seed: np.random.default_rng(seed)
        jnpsplit  = lambda gen: (gen, gen)
        jnpnormal = lambda gen, shape: gen.normal(size=shape)

        partial = functools.partial
    elif backend == 'jax':
        jnp, jsp = jax.numpy, jax.scipy

        jnparray = lambda a: jnp.array(a, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32)
        jnpzeros = lambda a: jnp.zeros(a, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32)
        intarray = lambda a: jnp.array(a, dtype=jnp.int64)

        jnpkey    = lambda seed: jax.random.PRNGKey(seed)
        jnpsplit  = jax.random.split
        jnpnormal = jax.random.normal

        partial = jax.tree_util.Partial

    factor = kwargs.get('factor')

    if factor == 'cholesky':
        matrix_factor = jsp.linalg.cho_factor
        matrix_solve  = jsp.linalg.cho_solve
        matrix_norm   = 2.0
    elif factor == 'lu':
        matrix_factor = jsp.linalg.lu_factor
        matrix_solve  = jsp.linalg.lu_solve
        matrix_norm   = 1.0

config(backend='jax', factor='cholesky')


def rngkey(seed):
    return jnpkey(seed)