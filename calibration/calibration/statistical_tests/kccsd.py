from typing import Any, Generic, Optional, Tuple, TypeVar

import jax.numpy as jnp
from jax import grad, jacobian, random
from jax_samplers.inference_algorithms.mcmc.base import MCMCConfig
from kwgflows.pytypes import Array, PRNGKeyArray, Scalar
from kwgflows.rkhs.kernels import MedianHeuristicKernel, base_kernel
from typing_extensions import TypeAlias, Unpack

from calibration.conditional_models.base import BaseConditionalModel
from calibration.kernels import BM2, ApproxMedianHeuristicKernel, kernel_on_models
from calibration.logging import get_logger
from calibration.statistical_tests.base import ApproximableTest, TestResults
from calibration.statistical_tests.kcsd import (
    ConditionalLogDensity_T,
    KCSDOneSample_T,
    KCSDTestInput_T,
    OneSampleKCSDStatFn,
    kcsd_test,
)
from calibration.statistical_tests.u_statistics import (
    ApproximableUStatFn,
    BaseApproximationState,
)

BM = TypeVar("BM", bound=BaseConditionalModel)

KCCSDOneSample_T: TypeAlias = KCSDOneSample_T[BM]
KCCSDTestInput_T: TypeAlias = KCSDTestInput_T[BM]


class KCCSDState(BaseApproximationState):
    approx_p: BaseConditionalModel


class OneSampleKCCSDStatFn(
    OneSampleKCSDStatFn[BM],
    ApproximableUStatFn[KCCSDOneSample_T[BM], KCCSDState],
    Generic[BM, BM2],
):
    x_kernel: kernel_on_models[BM, base_kernel[BM2], BM2]
    y_kernel: base_kernel[Array]
    conditional_score_q: ConditionalLogDensity_T[BM]

    def make_approximation_state(
        self,
        z: KCCSDOneSample_T[BM],
        num_particles: int,
        key: PRNGKeyArray,
        mcmc_config: Optional[MCMCConfig[Any, Any, Any]] = None,
    ) -> KCCSDState:
        approx_input = self.x_kernel.maybe_approximate_input(
            z[0], num_particles, key, mcmc_config
        )
        return KCCSDState(approx_input)

    def maybe_approximate_internals(
        self,
        num_particles: int,
        key: PRNGKeyArray,
        mcmc_config: Optional[MCMCConfig[Any, Any, Any]] = None,
    ) -> Any:
        return self.x_kernel.maybe_approximate_kernel(num_particles, key)

    def call_approximate(
        self,
        z_and_approx: Tuple[KCCSDOneSample_T[BM], KCCSDState],
        z_p_and_approx: Tuple[KCCSDOneSample_T[BM], KCCSDState],
        approx_internals: Any,
    ) -> Scalar:
        logger = get_logger("calibration.statistical_tests.kccsd")
        logger.info("kccsd: using approximate u-statistics evaluations")
        (p, y), (p_p, y_p) = (z_and_approx[0], z_p_and_approx[0])
        approx_k_p = approx_internals

        approx_p = z_and_approx[1].approx_p
        approx_p_p = z_p_and_approx[1].approx_p

        score_y_x = self.conditional_score_q(y, p)
        score_y_x_p = self.conditional_score_q(y_p, p_p)

        t1 = self.y_kernel(y, y_p) * jnp.dot(score_y_x, score_y_x_p)
        t2 = jnp.trace(jacobian(grad(self.y_kernel, argnums=1), argnums=0)(y, y_p))
        t3 = jnp.dot(score_y_x, grad(self.y_kernel, argnums=1)(y, y_p))
        t4 = jnp.dot(score_y_x_p, grad(self.y_kernel, argnums=0)(y, y_p))
        kp_eval = approx_k_p(approx_p, approx_p_p)
        return kp_eval * (t1 + t2 + t3 + t4)


class kccsd_test(
    ApproximableTest[KCCSDOneSample_T[BM], Unpack[KCCSDTestInput_T[BM]]],
    kcsd_test[BM],
    Generic[BM, BM2],
):
    """Specialization of Conditional goodness-of-fit tests for probability-defined
    inputs.
    """

    x_kernel: kernel_on_models[BM, base_kernel[BM2], BM2]

    def _get_u_stat_fn(
        self,
        X: BM,
        Y: Array,
        conditional_score_q: ConditionalLogDensity_T[BM],
        *,
        key: PRNGKeyArray,
    ) -> OneSampleKCCSDStatFn[BM, BM2]:
        logger = get_logger("calibration.statistical_tests.kccsd")
        # Use median heuristic if desired and possible
        x_kernel = self.x_kernel
        y_kernel = self.y_kernel
        if self.median_heuristic and (
            isinstance(x_kernel, (MedianHeuristicKernel, ApproxMedianHeuristicKernel))
            or isinstance(y_kernel, MedianHeuristicKernel)
        ):
            logger.info("kccsd: tuning kernel bandwidths using median heuristic.")
            indices = self.generate_indices(key, (X, Y))
            if isinstance(y_kernel, MedianHeuristicKernel):
                logger.info("kccsd: using median heuristic for y_kernel")
                y_kernel = y_kernel.with_median_heuristic(Y, indices)
                logger.info(f"kccsd: tuned y_kernel bandwidth: {y_kernel.bandwidth}")

            if not self.approximate:
                if isinstance(x_kernel, MedianHeuristicKernel):
                    logger.info("kccsd: using median heuristic for x_kernel")
                    x_kernel = x_kernel.with_median_heuristic(X, indices)
                    logger.info(
                        f"kccsd: tuned x_kernel bandwidth: {x_kernel.bandwidth}"
                    )
            else:
                if isinstance(x_kernel, ApproxMedianHeuristicKernel):
                    logger.info(
                        "kccsd: using approximate median heuristic for x_kernel"
                    )
                    key, subkey = random.split(key)
                    x_kernel = x_kernel.with_approximate_median_heuristic(
                        X,
                        self.approximation_num_particles,
                        subkey,
                        indices,
                        self.get_mcmc_config(),
                    )
                    logger.info(
                        f"kccsd: tuned x_kernel bandwidth: {x_kernel.bandwidth}"
                    )

        return OneSampleKCCSDStatFn(x_kernel, y_kernel, conditional_score_q)

    # This method directly calls its base class `__call__` but is
    # added to explicit the test's signature.
    def __call__(
        self,
        X: BM,
        Y: Array,
        conditional_score_q: ConditionalLogDensity_T[BM],
        *,
        key: PRNGKeyArray,
    ) -> TestResults:
        return super(kccsd_test, self).__call__(X, Y, conditional_score_q, key=key)
