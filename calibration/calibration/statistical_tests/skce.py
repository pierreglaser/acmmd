from typing import Any, Generic, Optional, Tuple, TypeVar, Union, cast

import jax
import jax.numpy as jnp
import numpyro.distributions as np_distributions
from jax._src.core import ConcreteArray
from jax_samplers.inference_algorithms.mcmc.base import MCMCAlgorithmFactory, MCMCConfig
from jax_samplers.kernels.mala import MALAConfig, MALAKernelFactory
from jax_samplers.kernels.ula import ULAConfig, ULAKernelFactory
from kwgflows.pytypes import Array, PRNGKeyArray, Scalar
from kwgflows.rkhs.kernels import MedianHeuristicKernel, base_kernel, gaussian_kernel
from typing_extensions import TypeAlias, Unpack

from calibration.conditional_models.base import (
    BaseConditionalModel,
    ConcreteLogDensityModel,
    DiscretizedModel,
    LogDensityModel,
    SampleableModel,
)
from calibration.conditional_models.gaussian_models import GaussianConditionalModel
from calibration.kernels import BM, ApproxMedianHeuristicKernel, kernel_on_models
from calibration.logging import get_logger
from calibration.statistical_tests.base import ApproximableTest, TestResults
from calibration.statistical_tests.kcsd import TwoKernelStruct
from calibration.statistical_tests.u_statistics import (
    ApproximableUStatFn,
    BaseApproximationState,
)

P = TypeVar("P", bound=BaseConditionalModel)

SKCEOneSample_T: TypeAlias = Tuple[P, Array]


class SKCEApproximationState(Generic[BM], BaseApproximationState):
    p_expectations: Union[GaussianConditionalModel, DiscretizedModel[Any]]
    p_kernel: BM


class OneSampleSKCEUStatFn(
    ApproximableUStatFn[SKCEOneSample_T[P], SKCEApproximationState[BM]],
    Generic[P, BM],
):
    r"""U-statistic for the calibration error test of [1]


    The complete U-statistics takes the form
    .. math::
        \sum_{1 \leq i \ne j \leq n} k_p(p, p') h((p_i, y_i), (p_j, y_j))

    Where :math:`h` is (assuming tensor-product kernels)
    .. math::
        h((p, y), (p', y')) = k_y(y, y')
            - \mathbb E_{Z \sim p} k_y(Z, y')
            - \mathbb E_{Z' \sim p'} k_y(y, Z')
            - \mathbb E_{Z \sim p, Z' \sim p'} k_y(Z, Z')


    [1] Calibration Tests Beyond Classification, Widmann et al., 2022
    """
    p_kernel: kernel_on_models[P, base_kernel[BM], BM]
    y_kernel: base_kernel[Array]

    def __call__(
        self,
        z: Tuple[P, Array],
        z_p: Tuple[P, Array],
    ) -> Scalar:
        (p, y), (p_p, y_p) = (z, z_p)
        logger = get_logger("calibration.statistical_tests.skce")
        if isinstance(p, GaussianConditionalModel):
            logger.info("gaussian")
            assert isinstance(p_p, GaussianConditionalModel)
            assert isinstance(self.y_kernel, gaussian_kernel)
            t1 = self.y_kernel(y, y_p)
            t2 = p.analytical_expectation_of(self.y_kernel, y_p)
            t3 = p_p.analytical_expectation_of(self.y_kernel, y)
            t4 = p.bivariate_analytical_expectation_of(self.y_kernel, p_p)
        else:
            assert isinstance(p, DiscretizedModel)
            assert isinstance(p_p, DiscretizedModel)
            t1 = self.y_kernel(y, y_p)
            t2 = p.analytical_expectation_of(self.y_kernel, y_p)
            t3 = p_p.analytical_expectation_of(self.y_kernel, y)
            t4 = p.bivariate_analytical_expectation_of(self.y_kernel, p_p)

        return cast(Scalar, self.p_kernel(p, p_p) * (t1 - t2 - t3 + t4))

    def make_approximation_state(
        self,
        z: SKCEOneSample_T[P],
        num_particles: int,
        key: PRNGKeyArray,
        mcmc_config: Optional[MCMCConfig[Any, Any, Any]] = None,
    ) -> SKCEApproximationState[BM]:
        p, _ = z
        logger = get_logger("calibration.statistical_tests.skce")
        logger.info(f"OneSampleSKCEUStatFn: approximating inputs")

        if isinstance(p, DiscretizedModel):
            logger.info(
                f"OneSampleSKCEUStatFn: p is discrete and already admits tractable "
                f"expectations, not performing any approximation."
            )
            p_expectations = p
        elif isinstance(p, GaussianConditionalModel) and isinstance(
            self.y_kernel, gaussian_kernel
        ):
            logger.info(
                f"OneSampleSKCEUStatFn: p is a GaussianConditionalModel used with "
                f"a gaussian_kernel and admits tractable expectations, not performing "
                f"any approximation."
            )
            p_expectations = p
        else:
            logger.info(
                f"OneSampleSKCEUStatFn: discretizing p ({p}) using {num_particles} "
                f"particles to compute SKCE expectations."
            )
            if isinstance(p, SampleableModel):
                logger.info(
                    f"OneSampleSKCEUStatFn: p is a SampleableModel, using "
                    f"p.non_mcmc_discretize to sample."
                )
                key, subkey = jax.random.split(key)
                p_expectations = p.non_mcmc_discretize(subkey, num_particles)
            else:
                assert isinstance(p, ConcreteLogDensityModel)
                logger.info(
                    f"OneSampleSKCEUStatFn: p is not a SampleableModel, using "
                    f"p.mcmc_discretize to sample."
                )
                assert mcmc_config is not None
                if mcmc_config.num_chains > num_particles:
                    logger.info(
                        f"OneSampleSKCEUStatFn: mcmc_config.num_chains "
                        f"({mcmc_config.num_chains}) > num_particles "
                        f"({num_particles}), reducing mcmc_config.num_chains to "
                        f"num_particles."
                    )
                    mcmc_config = mcmc_config.replace(
                        num_chains=min(mcmc_config.num_chains, num_particles)
                    )
                base_dist = np_distributions.Normal(
                    loc=jnp.zeros((p.dimension,)), scale=1.0  # type: ignore
                )
                factory = MCMCAlgorithmFactory(config=mcmc_config)
                key, subkey = jax.random.split(key)
                p_expectations = p.mcmc_discretize(
                    base_dist, subkey, factory, num_particles
                )

        key, subkey = jax.random.split(key)
        p_approx = self.p_kernel.maybe_approximate_input(
            p, num_particles, subkey, mcmc_config
        )
        return SKCEApproximationState(p_expectations, p_approx)

    def maybe_approximate_internals(self, num_particles: int, key: PRNGKeyArray) -> Any:
        return self.p_kernel.maybe_approximate_kernel(num_particles, key)

    def call_approximate(
        self,
        z_and_approx: Tuple[SKCEOneSample_T[P], SKCEApproximationState[BM]],
        z_p_and_approx: Tuple[SKCEOneSample_T[P], SKCEApproximationState[BM]],
        approx_internals: Any,
    ) -> Scalar:
        logger = get_logger("calibration.statistical_tests.skce")
        logger.info("skce: using approximate u-statistics evaluations")
        (p, y), (p_p, y_p) = (z_and_approx[0], z_p_and_approx[0])
        # used to approximate expectations of `y_kernel` under the models
        p_approx_exp = z_and_approx[1].p_expectations
        p_p_approx_exp = z_p_and_approx[1].p_expectations

        # used to approximate the kernel between the models
        p_approx_kernel = z_and_approx[1].p_kernel
        p_p_approx_kernel = z_p_and_approx[1].p_kernel

        # Approximate kernel, if the kernel contains intractable terms
        # that are not the inputs of the kernel.
        approx_kernel = approx_internals

        if isinstance(p_approx_exp, GaussianConditionalModel):
            assert isinstance(self.y_kernel, gaussian_kernel)
            assert isinstance(p_p_approx_exp, GaussianConditionalModel)
            t1 = self.y_kernel(y, y_p)
            t2 = p_approx_exp.analytical_expectation_of(self.y_kernel, y_p)
            t3 = p_p_approx_exp.analytical_expectation_of(self.y_kernel, y)
            t4 = p_approx_exp.bivariate_analytical_expectation_of(
                self.y_kernel, p_p_approx_exp
            )
        else:
            assert isinstance(p_p_approx_exp, DiscretizedModel)
            t1 = self.y_kernel(y, y_p)
            t2 = p_approx_exp.analytical_expectation_of(self.y_kernel, y_p)
            t3 = p_p_approx_exp.analytical_expectation_of(self.y_kernel, y)
            t4 = p_approx_exp.bivariate_analytical_expectation_of(
                self.y_kernel, p_p_approx_exp
            )

        p_kernel_val = approx_kernel(p_approx_kernel, p_p_approx_kernel)
        return cast(Scalar, p_kernel_val * (t1 - t2 - t3 + t4))


SKCETestInput_T: TypeAlias = Tuple[P, Array]


class skce_test(
    ApproximableTest[SKCEOneSample_T[P], Unpack[SKCETestInput_T[P]]],
    TwoKernelStruct[kernel_on_models[P, base_kernel[BM], BM], base_kernel[Array]],
):
    """Squared Kernel Calibration Error Tests.

    ...

    See [1] for details.

    [1] Calibration Tests Beyond Classification, Widmann et al., 2022
    """

    def _get_u_stat_fn(
        self,
        X: P,
        Y: Array,
        *,
        key: jax.random.KeyArray,
    ) -> OneSampleSKCEUStatFn[P, BM]:
        logger = get_logger("calibration.statistical_tests.skce")
        # Use median heuristic if desired and possible
        p_kernel = self.x_kernel
        y_kernel = self.y_kernel
        if self.median_heuristic and (
            isinstance(p_kernel, (MedianHeuristicKernel, ApproxMedianHeuristicKernel))
            or isinstance(y_kernel, MedianHeuristicKernel)
        ):
            logger.info("skce: tuning kernel bandwidths using median heuristic.")
            indices = self.generate_indices(key, (X, Y))
            if isinstance(y_kernel, MedianHeuristicKernel):
                logger.info("skce: using median heuristic for y_kernel")
                y_kernel = y_kernel.with_median_heuristic(Y, indices)
                logger.info(
                    f"skce: y_kernel bandwidth after tuning: {y_kernel.bandwidth}"
                )

            if not self.approximate:
                if isinstance(p_kernel, MedianHeuristicKernel):
                    logger.info("skce: using median heuristic for p_kernel")
                    p_kernel = p_kernel.with_median_heuristic(X, indices)
                    logger.info(
                        f"skce: p_kernel bandwidth after tuning: {p_kernel.bandwidth}"
                    )
            else:
                if isinstance(p_kernel, ApproxMedianHeuristicKernel):
                    logger.info("skce: using approximate median heuristic for p_kernel")
                    key, subkey = jax.random.split(key)
                    p_kernel = p_kernel.with_approximate_median_heuristic(
                        X,
                        self.approximation_num_particles,
                        subkey,
                        indices,
                        self.get_mcmc_config(),
                    )
                    logger.info(
                        f"skce: p_kernel bandwidth after tuning: {p_kernel.bandwidth}"
                    )

        return OneSampleSKCEUStatFn(p_kernel, y_kernel)

    def _build_one_sample(self, X: P, Y: Array) -> SKCEOneSample_T[P]:
        return (X, Y)

    # This method directly calls its base class `__call__` but is
    # added to explicit the test's signature.
    def __call__(self, X: P, Y: Array, *, key: PRNGKeyArray) -> TestResults:
        return super().__call__(X, Y, key=key)
