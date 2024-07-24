import jax.numpy as jnp
from kwgflows.pytypes import Array, Scalar


def compute_dtw(x: Array, y: Array) -> Scalar:
    assert x.ndim == 2 and y.ndim == 2
    n, m = x.shape[0], y.shape[0]
    M = jnp.zeros((n + 1, m + 1))
    M = M.at[:, :].set(jnp.nan)
    log_M = M
    log_M = log_M.at[0, 0].set(0)
    log_M = log_M.at[1:, 0].set(-jnp.inf)
    log_M = log_M.at[0, 1:].set(-jnp.inf)

    for i in range(1, max(n, m) + 1):
        # fill ith column
        for j in 
        log_M
