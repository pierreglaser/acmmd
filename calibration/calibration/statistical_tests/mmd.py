from typing import Tuple

import jax
import jax.numpy as jnp
from kwgflows.base import DiscreteProbability
from kwgflows.divergences.mmd import mmd
from kwgflows.pytypes import Array, PRNGKeyArray, Scalar, T
from kwgflows.rkhs.kernels import MedianHeuristicKernel, base_kernel
from kwgflows.utils import infer_num_samples_pytree
from typing_extensions import TypeAlias, Unpack

from calibration.statistical_tests.base import (
    KernelBasedStruct,
    OneSampleTest,
    TestResults,
)
from calibration.statistical_tests.u_statistics import UStatFn

MMDOneSample_T = Tuple[T, T]


class OneSampleMMDUStatFn(UStatFn[MMDOneSample_T[T]]):
    kernel: base_kernel[T]

    def __call__(self, z: MMDOneSample_T[T], z_p: MMDOneSample_T[T]) -> Scalar:
        (x, y), (x_p, y_p) = (z, z_p)
        return (
            self.kernel(x, x_p)
            + self.kernel(y, y_p)
            - self.kernel(x, y_p)
            - self.kernel(x_p, y)
        )


MMDTestInput_T: TypeAlias = Tuple[T, T]


class mmd_test(
    OneSampleTest[MMDOneSample_T[T], Unpack[MMDTestInput_T[T]]],
    KernelBasedStruct[base_kernel[T]],
):
    """
    Perform a MMD-based two-sample test between X and Y.
    """

    def _get_u_stat_fn(
        self, X: T, Y: T, *, key: jax.random.KeyArray
    ) -> OneSampleMMDUStatFn[T]:
        # Use median heuristic if desired and possible
        kernel = self.kernel
        if self.median_heuristic and isinstance(kernel, MedianHeuristicKernel):
            print("running median heuristic on kernels")
            X_Y = jax.tree_map(lambda x, y: jnp.concatenate([x, y]), X, Y)

            # Compute indices of [X, Y] for median heuristic computations
            # and ensure that they are still sorted in ascending order
            n = infer_num_samples_pytree((X, Y))
            indices = self.generate_indices(
                key, (X, Y), max_num=self.max_num_distances // 4
            )
            indices1, indices2 = indices
            indices1 = jnp.repeat(indices1, 2)
            indices1 = jnp.concatenate([indices1, indices1 + n])
            indices2 = jnp.repeat(indices2, 2)
            indices2 = indices2.at[1:2:].add(n)
            indices2 = jnp.concatenate([indices2, indices2])

            kernel = kernel.with_median_heuristic(X_Y, (indices1, indices2))

        return OneSampleMMDUStatFn(kernel)

    def _build_one_sample(self, X: T, Y: T) -> MMDOneSample_T[T]:
        return (X, Y)

    # This method directly calls its base class `__call__` but is
    # added to explicit the test's signature.
    def __call__(self, X: T, Y: T, *, key: PRNGKeyArray) -> TestResults:
        return super(mmd_test, self).__call__(X, Y, key=key)

    def v_statistic(self, X: T, Y: T) -> Scalar:
        r"""
        Given two empirical probability distributions $\widehat{P}$ and $\widehat{Q}$,
        compute the v-statistic of the MMD, given by [1, Equation 5]:
        $$
        \widehat{\textrm{MMD}(P, Q)}^2 =
            \frac{1}{n^2} \sum_{i, j} K(x_i, x_j) +
            \frac{1}{m^2} \sum_{i, j} K(y_i, y_j) - 2
            \frac{1}{nm} \sum_{i, j} K(x_i, y_j)
             = \textrm{MMD}(\widehat{P}, \widehat{Q})^2
             = \| \widehat{\mu}_\widehat{P} - \widehat{\mu}_\widehat{Q} \|^2
        $$

        [1] Gretton, Arthur, et. Al. "A kernel two-sample test." Journal of Machine
        Learning Research (2012)
        """
        P, Q = DiscreteProbability.from_samples(X), DiscreteProbability.from_samples(Y)
        return mmd(self.kernel)(P, Q)
