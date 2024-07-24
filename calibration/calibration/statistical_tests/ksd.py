from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar

import jax.numpy as jnp
from flax import struct
from jax import grad, jacobian
from kwgflows.base import DiscreteProbability
from kwgflows.pytypes import Array, PRNGKeyArray, Scalar
from kwgflows.rkhs.kernels import MedianHeuristicKernel, base_kernel, gaussian_kernel
from typing_extensions import Self, TypeAlias, Unpack

from calibration.statistical_tests.base import (
    KernelBasedStruct,
    OneSampleTest,
    TestResults,
)
from calibration.statistical_tests.u_statistics import UStatFn
from calibration.utils import GradLogDensity_T

M_co = TypeVar("M_co", bound=base_kernel[Array], covariant=True)


class steinalized_kernel(Generic[M_co], base_kernel[Array]):
    kernel: M_co
    score_fn: GradLogDensity_T

    @classmethod
    def create(
        cls: Type[Self],
        *args: List[Any],
        kernel: base_kernel[Array] = gaussian_kernel(1.0),
        score_q: Optional[GradLogDensity_T] = None,
        **kwargs: Dict[Any, Any],
    ) -> Self:
        assert (args, kwargs) == (
            (),
            {},
        ), "No positional or unknown keyword arguments allowed"
        assert score_q is not None
        return cls(kernel, score_q)

    def __call__(self, x: Array, y: Array) -> Scalar:
        term_1 = jnp.dot(self.score_fn(x), self.score_fn(y)) * self.kernel(x, y)
        term_2 = jnp.dot(self.score_fn(x), grad(self.kernel, argnums=1)(x, y))
        term_3 = jnp.dot(grad(self.kernel, argnums=0)(x, y), self.score_fn(y))
        term_4 = jnp.trace(jacobian(grad(self.kernel, argnums=1), argnums=0)(x, y))
        return term_1 + term_2 + term_3 + term_4

    def with_median_heuristic(
        self: "steinalized_kernel[MedianHeuristicKernel[Array]]",
        X: Array,
        indices: Optional[Tuple[Array, Array]] = None,
        quantile: Optional[float] = None,
        *,
        allow_nans: bool = False,
    ) -> "steinalized_kernel[MedianHeuristicKernel[Array]]":
        kernel = self.kernel.with_median_heuristic(
            X, indices, quantile, allow_nans=allow_nans
        )
        return self.replace(kernel=kernel)


class ksd(struct.PyTreeNode):
    kernel: base_kernel[Array]
    score_q: GradLogDensity_T
    squared: bool = True

    def __call__(self, P: DiscreteProbability[Array]) -> Scalar:
        k = steinalized_kernel.create(kernel=self.kernel, score_q=self.score_q)
        witness_function = P.get_mean_embedding(k)
        return witness_function.rkhs_norm(squared=self.squared)


KSDOneSample_T = Array


class OneSampleKSDStat(UStatFn[KSDOneSample_T]):
    kernel: base_kernel[KSDOneSample_T]
    score_q: GradLogDensity_T

    def __call__(self, z: KSDOneSample_T, z_p: KSDOneSample_T) -> Scalar:
        k = steinalized_kernel.create(kernel=self.kernel, score_q=self.score_q)
        return k(z, z_p)


KSDTestInput_T: TypeAlias = Tuple[Array, GradLogDensity_T]


class ksd_test(
    OneSampleTest[KSDOneSample_T, Unpack[KSDTestInput_T]],
    KernelBasedStruct[base_kernel[Array]],
):
    """
    Perform a MMD-based two-sample test between X and Y.
    """

    def _get_u_stat_fn(
        self, _: Array, score_q: GradLogDensity_T, *, key: PRNGKeyArray
    ) -> OneSampleKSDStat:
        assert self.kernel is not None
        return OneSampleKSDStat(self.kernel, score_q)

    def _build_one_sample(self, X: Array, _: GradLogDensity_T) -> Array:
        return X

    # This method directly calls its base class `__call__` but is
    # added to explicit the test's signature.
    def __call__(
        self, X: Array, score_q: GradLogDensity_T, *, key: PRNGKeyArray
    ) -> TestResults:
        return super(ksd_test, self).__call__(X, score_q, key=key)

    def v_statistic(self, X: Array, score_q: GradLogDensity_T) -> Scalar:
        return ksd(self.kernel, score_q)(DiscreteProbability.from_samples(X))
