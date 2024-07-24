from typing import Optional, Protocol, Tuple, Union, cast

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from kwgflows.rkhs.kernels import MedianHeuristicKernel
from tensorflow_probability.substrates import jax as tfp

from calibration.conditional_models.gaussian_models import GaussianConditionalModel


def fill_diagonal(X: jax.Array, val: Union[float, int, jax.Array]) -> jax.Array:
    """Return a copy of matrix :params:`X` with diagonal entries set to :params:`val`.

    :param jax.Array X: Square matrix.
    :param Union[float, int, jax.Array] val: Value(s) to write on the diagonal.

    Reference: https://github.com/google/jax/issues/2680
    """
    assert X.ndim == 2
    assert X.shape[0] == X.shape[1]
    i, j = jnp.diag_indices(X.shape[0])
    return X.at[i, j].set(val, indices_are_sorted=True, unique_indices=True)


def superdiags_indices(N: int, R: int):
    """
    Return a tuple of row and column indices of the :params:`R` superdiagonals of a square matrix
    of size :param:`N` x :param:`N`.

    The indices are sorted in ascending order.
    The total number of indices is :math:`NR - R(R+1)/2`.

    :param int N: Number of rows and columns of the square matrix.
    :param int R: Number of super-diagonals (greater than 0 and less than :param:`N`).
    """
    assert R > 0
    assert R < N
    # indices of rows, in ascending order:
    # R indices in each of the top N - R rows: 0, ..., 0; 1, ..., 1; ...; N - R - 1, ..., N - R - 1
    # R - 1, R - 2, etc. indices in the following R - 1 rows: N - R, ..., N - R; ...; N - 2, ..., N - 2
    x = jnp.asarray(
        np.repeat(
            np.arange(N - 1),
            np.concatenate([np.full((N - R,), R), np.arange(R - 1, stop=0, step=-1)]),
        )
    )
    # indices of columns, in ascending order in each row:
    y = jnp.concatenate(
        [
            # R indices in each of the top N - R rows: 1, ..., R; 2, ..., R + 1; ...; N - R, ..., N - 1
            # R - 1, R - 2, etc. indices in the following R - 1 rows: N - R + 1, ..., N - 1; ...; N - 2, N - 1; N - 1
            np.arange(i + 1, min(i + 1 + R, N))
            for i in np.arange(N - 1)
        ]
    )
    return x, y


def subdiags_indices(N: int, R: int):
    """
    Return a tuple of row and column indices of the :params:`R` subdiagonals of a square matrix
    of size :param:`N` x :param:`N`.

    The indices are sorted in ascending order.
    The total number of indices is :math:`NR - R(R+1)/2`.

    :param int N: Number of rows and columns of the square matrix.
    :param int R: Number of sub-diagonals (greater than 0 and less than :param:`N`).
    """
    assert R > 0
    assert R < N
    # indices of rows, in ascending order:
    # 1, 2, ..., R-1 indices in each of the bottom R rows: 1, ..., 1; ...; R-1, ..., R-1
    # R, indices in the following N - R rows: R, ..., R; ...; N, ..., N
    x = jnp.asarray(
        np.repeat(
            np.arange(1, N), np.concatenate([np.arange(1, R), np.full((N - R,), R)])
        )
    )
    # indices of columns, in ascending order in each row:
    y = jnp.concatenate(
        [
            # R indices in each of the top N - R rows: 1, ..., R; 2, ..., R + 1; ...; N - R, ..., N - 1
            # R - 1, R - 2, etc. indices in the following R - 1 rows: N - R + 1, ..., N - 1; ...; N - 2, N - 1; N - 1
            np.arange(max(i - R, 0), i)
            for i in np.arange(1, N)
        ]
    )
    return x, y


def sub_and_supdiag_indices(N: int, R: int):
    """
    Return a tuple of row and column indices of the :params:`R` sub and superdiagonals
    of a square matrix of size :param:`N` x :param:`N`.

    The indices are sorted in ascending order.
    The total number of indices is :math:`2 * (NR - R(R+1)/2)`.

    :param int N: Number of rows and columns of the square matrix.
    :param int R: Number of sub/superdiagonals (greater than 0 and
                                                less than :param:`N`).
    """
    assert R > 0
    assert R < N
    x = jnp.asarray(
        np.repeat(
            np.arange(N),
            np.concatenate([np.arange(R), np.full((N - R,), R)])
            + np.concatenate(
                [np.full((N - R,), R), np.arange(R - 1, stop=-1, step=-1)]
            ),
        )
    )
    y = jnp.concatenate(
        [
            np.concatenate(
                [np.arange(max(i - R, 0), i), np.arange(i + 1, min(i + 1 + R, N))]
            )
            for i in np.arange(N)
        ]
    )
    return x, y


class GradLogDensity_T(Protocol):
    def __call__(self, x: jax.Array, /) -> jax.Array:
        ...


def median_euclidean_gaussians(
    p: GaussianConditionalModel, q: GaussianConditionalModel
) -> float:
    """
    Return the median Euclidean distance between isotropic Gaussian distributions
    :param:`p` and :param:`q`.
    """
    d = p.dimension
    assert q.dimension == d

    # Define the scaled components of the mixture
    scale_XX = 2 * p.sigma**2
    scale_YY = 2 * q.sigma**2
    scale_XY = p.sigma**2 + q.sigma**2
    dist_scaled_XX_YY = tfp.distributions.Chi2(df=d)
    dist_scaled_XY = tfp.distributions.NoncentralChi2(
        df=d, noncentrality=cast(float, jnp.sum(jnp.square(p.mu - q.mu)) / scale_XY)
    )

    # Compute median values of |X - X'|^2, |Y - Y'|^2 and |X - Y|^2
    median_scaled_XX_XY = dist_scaled_XX_YY.quantile(0.5)
    median_scaled_XY = dist_scaled_XY.quantile(0.5)
    median_XX = scale_XX * median_scaled_XX_XY
    median_YY = scale_YY * median_scaled_XX_XY
    median_XY = scale_XY * median_scaled_XY

    # Compute minimum and maximum of the median values
    # The median of the mixture is guaranteed to be in-between these values
    medians = jnp.asarray([median_XX, median_YY, median_XY])
    min_median = cast(float, jnp.min(medians))
    max_median = cast(float, jnp.max(medians))

    # Define function whose root is the median
    def median_root_func(x: float) -> float:
        return (
            dist_scaled_XY.cdf(x / scale_XY)
            + 0.5
            * (
                dist_scaled_XX_YY.cdf(x / scale_XX)
                + dist_scaled_XX_YY.cdf(x / scale_YY)
            )
            - 1
        )

    # Compute the median of the mixture with a bisection algorithm
    bisection = jaxopt.Bisection(
        optimality_fun=median_root_func,
        lower=min_median,
        upper=max_median,
        check_bracket=False,
    )
    median = bisection.run().params

    return cast(float, jnp.sqrt(median))


def median_median_heuristic_euclidean_gaussian(
    X: GaussianConditionalModel,
    indices: Optional[Tuple[jax.Array, jax.Array]] = None,
    quantile: Optional[float] = None,
) -> float:
    # Compute the median of the median distances in the ground space
    # for Gaussian models and a Gaussian kernel.
    #
    # More concretely, we compute the bandwidth
    #   median_{(i, j) ∈ indices} medianheuristic(X[i], X[j])
    # where medianheuristic(X[i], X[j]) is the median of the pairwise distances
    # of samples from X[i] and X[j].
    #
    # Since X[i] and X[j] are isotropic Gaussian distributions, we know the
    # distribution of |Z - Z'|₂² where Z, Z' are iid. samples from the
    # mixture of the Gaussians X[i] and X[j] with equal weights.
    # We know that
    #   |Zi - Zi'|₂² / (2 var(X[i])) ~ χ²(d)
    #   |Zj - Zj'|₂² / (2 var(X[j])) ~ χ²(d)
    #   |Zi - Zj|₂² / (var(X[i]) + var(X[j])) ~ χ²(d, |mean(X[i]) - mean(X[j])|₂² / (var(X[i]) + var(X[j])))
    # where Zi, Zi' are iid. samples from X[i]; Zj, Zj' are iid. samples from X[j];
    # and d is the dimension of the support of X[i] and X[j].
    # The distribution of |Z - Z'|₂² is a mixture of the distributions of
    # |Zi - Zi'|₂², |Zj - Zj'|₂², and |Zi - Zj|₂² with weights 1/4, 1/4, and 1/2.
    # Hence we can write the cumulative distribution function of |Z - Z'|₂²
    # as a weighted sum of the cumulative distribution functions of
    # |Zi - Zi'|₂², |Zj - Zj'|₂², and |Zi - Zj|₂².
    # The median of |Z - Z'|₂² is guaranteed to be between the
    # minimum and maximum of the medians of these mixture components,
    # which allows us to employ a bisection algorithm to compute the median
    # of |Z - Z'|₂².
    if indices is None:
        ms = jnp.asarray(
            jax.vmap(
                jax.vmap(median_euclidean_gaussians, in_axes=(0, None)),
                in_axes=(None, 0),
            )(X, X)
        )
        ms = ms[jnp.triu_indices_from(ms, k=1)]
    else:
        indices1, indices2 = indices
        X1 = jax.tree_map(lambda x: x[indices1], X)
        X2 = jax.tree_map(lambda x: x[indices2], X)
        ms = jnp.asarray(jax.vmap(median_euclidean_gaussians, in_axes=(0, 0))(X1, X2))
    if quantile is None:
        median_func = jnp.median
    else:
        median_func = lambda x: jnp.quantile(x, quantile)
    return cast(float, median_func(ms))


def median_median_heuristic_discrete(
    kernel: MedianHeuristicKernel[jax.Array],
    X: jax.Array,
    indices: Optional[Tuple[jax.Array, jax.Array]] = None,
    quantile: Optional[float] = None,
    inner_indices: Optional[Tuple[jax.Array, jax.Array]] = None,
    allow_nans: bool = False,
) -> float:
    # Compute the median of the median distances in the ground space.
    # X has to be an at least 2-dimensional array, where the first dimension
    # will be scanned over to perform individual median heuristic computations.
    # this function is used by kernels on distributions.
    # More concretely, we use the bandwidth
    #   median_{(i, j) ∈ indices} medianheuristic(X[i], X[j])
    # where medianheuristic(X[i], X[j]) is the median of the pairwise distances
    # of samples from X[i] and X[j].
    def median_estimate(
        kernel: MedianHeuristicKernel[jax.Array], x: jax.Array, y: jax.Array
    ) -> float:
        xy = jnp.concatenate([x, y])
        # print(f"using inner indices: {inner_indices}")
        return kernel.with_median_heuristic(
            xy, inner_indices, quantile=quantile, allow_nans=allow_nans
        ).bandwidth

    if indices is None:
        ms = jnp.asarray(
            jax.vmap(
                jax.vmap(median_estimate, in_axes=(None, 0, None)),
                in_axes=(None, None, 0),
            )(kernel, X, X)
        )
        ms = ms[jnp.triu_indices_from(ms, k=1)]
    else:
        indices1, indices2 = indices
        X1 = X[indices1]
        X2 = X[indices2]
        ms = jnp.asarray(
            jax.vmap(median_estimate, in_axes=(None, 0, 0))(kernel, X1, X2)
        )

    return cast(float, jnp.median(ms))
