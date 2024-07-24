from typing import Any, Callable, Generic, Tuple, TypeVar

import jax
import jax.numpy as jnp
from flax import struct
from jax import grad, jacobian
from kwgflows.pytypes import T1, T2, Array, PRNGKeyArray, Scalar, T
from kwgflows.rkhs.kernels import MedianHeuristicKernel, base_kernel
from typing_extensions import TypeAlias, Unpack

from calibration.statistical_tests.base import OneSampleTest, TestResults
from calibration.statistical_tests.u_statistics import UStatFn

ConditionalLogDensity_T: TypeAlias = Callable[[Array, T], Array]

KCSDOneSample_T = Tuple[T, Array]


class OneSampleKCSDStatFn(UStatFn[KCSDOneSample_T[T]]):
    """Kernel conditional Stein discrepancy (KCSD) One-Sample Statistics.

    ...

    The complete statistics takes the form
    .. math::
        \\sum_{1 \\leq i \\neq j \\leq n} k_x(x_i, x_j) h(x_i, x_j, y_i, y_j)

    Where h is
    .. math::
        h(x_i, x_j, y_i, y_j) =
        k_y(y_i, y_j) \\langle s_q(y_i, x_i), s_q(y_j, x_j) \\rangle
        + \\sum_{i=1}^{d}
          \\frac{\\partial^2 k_y(y_i, {y'}_i)}{\\partial y_i \\partial {y'}_i}
        + \\langle
              s_q(y_i, x_i),
              \\frac{\\partial k_y(y_i, y_j)}{\\partial {y'}_i}
          \\rangle
        + \\langle
            s_q(y_j, x_j),
            \\frac{\\partial k_y(y_i, y_j)}{\\partial y_i}
          \\rangle

    And :math:`s_q(y, x)` is the score of the conditional density :math:`q(y|x)`
    with respect to :math:`y`.
    .. math::
        s_q(y, x) = \\nabla_y \\log q(y|x)

    .. note::
        It also can be computed by not averaging the statistic's function over
        all pairs of indices (sampled without replacement).


    See [1, Section 3] for more details.

    [1] Testing Goodness of Fit of Conditional Density Models withb Kernels
    Jitkrittum et. al. 2020
    """

    x_kernel: base_kernel[T]
    y_kernel: base_kernel[Array]
    conditional_score_q: ConditionalLogDensity_T[T]

    def __call__(self, z: KCSDOneSample_T[T], z_p: KCSDOneSample_T[T]) -> Scalar:
        x, y = z
        x_p, y_p = z_p

        score_y_x = self.conditional_score_q(y, x)
        score_y_x_p = self.conditional_score_q(y_p, x_p)

        t1 = self.y_kernel(y, y_p) * jnp.dot(score_y_x, score_y_x_p)
        t2 = jnp.trace(jacobian(grad(self.y_kernel, argnums=1), argnums=0)(y, y_p))
        t3 = jnp.dot(score_y_x, grad(self.y_kernel, argnums=1)(y, y_p))
        t4 = jnp.dot(score_y_x_p, grad(self.y_kernel, argnums=0)(y, y_p))
        return self.x_kernel(x, x_p) * (t1 + t2 + t3 + t4)


K_T1 = TypeVar("K_T1", bound=base_kernel[Any])
K_T2 = TypeVar("K_T2", bound=base_kernel[Any])


class TwoKernelStruct(Generic[K_T1, K_T2], struct.PyTreeNode):
    x_kernel: K_T1
    y_kernel: K_T2


KCSDTestInput_T: TypeAlias = Tuple[T, Array, ConditionalLogDensity_T[T]]


class kcsd_test(
    OneSampleTest[KCSDOneSample_T[T], Unpack[KCSDTestInput_T[T]]],
    TwoKernelStruct[base_kernel[T], base_kernel[Array]],
):
    r"""Conditional goodness-of-fit test based on the kernel conditional Stein discrepancy.

    ...

    The test statistic is computed using i.i.d. samples from the joint distribution of
    random variables :math:`(X, Y)` and the score function of the model for the
    conditional distributions :math:`p(Y \,|\, X)`.

    See [1].

    [1] Testing Goodness of Fit of Conditional Density Models with Kernels,
    Jitkrittum et al. 2020
    """

    def _get_u_stat_fn(
        self,
        X: T,
        Y: Array,
        conditional_score_q: ConditionalLogDensity_T[T],
        *,
        key: jax.random.KeyArray,
    ) -> OneSampleKCSDStatFn[T]:
        # Use median heuristic if desired and possible
        x_kernel = self.x_kernel
        y_kernel = self.y_kernel
        if self.median_heuristic and (
            isinstance(x_kernel, MedianHeuristicKernel)
            or isinstance(y_kernel, MedianHeuristicKernel)
        ):
            indices = self.generate_indices(key, (X, Y))
            if isinstance(x_kernel, MedianHeuristicKernel):
                x_kernel = x_kernel.with_median_heuristic(X, indices)
            if isinstance(y_kernel, MedianHeuristicKernel):
                y_kernel = y_kernel.with_median_heuristic(Y, indices)

        return OneSampleKCSDStatFn(x_kernel, y_kernel, conditional_score_q)

    def _build_one_sample(
        self,
        X: T,
        Y: Array,
        conditional_score_q: ConditionalLogDensity_T[T],
    ) -> KCSDOneSample_T[T]:
        return (X, Y)

    # This method directly calls its base class `__call__` but is
    # added to explicit the test's signature.
    def __call__(
        self,
        X: T,
        Y: Array,
        conditional_score_q: ConditionalLogDensity_T[T],
        *,
        key: PRNGKeyArray,
    ) -> TestResults:
        return super(kcsd_test, self).__call__(X, Y, conditional_score_q, key=key)
