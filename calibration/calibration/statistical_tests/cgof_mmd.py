from typing import Any, Generic, Optional, Tuple, TypeVar, Union, cast

import jax
import jax.numpy as jnp
import numpyro.distributions as np_distributions
from jax_samplers.inference_algorithms.mcmc.base import MCMCAlgorithmFactory, MCMCConfig
from kwgflows.pytypes import Array, PRNGKeyArray, Scalar
from kwgflows.rkhs.kernels import MedianHeuristicKernel, base_kernel, gaussian_kernel
from typing_extensions import TypeAlias, Unpack

from calibration.conditional_models.base import (
    BaseConditionalModel,
    ConcreteLogDensityModel,
    DiscretizedModel,
    SampleableModel,
)
from calibration.conditional_models.gaussian_models import GaussianConditionalModel
from calibration.kernels import ApproxMedianHeuristicKernel
from calibration.logging import get_logger
from calibration.statistical_tests.base import ApproximableTest, TestResults
from calibration.statistical_tests.kcsd import TwoKernelStruct
from calibration.statistical_tests.u_statistics import (
    ApproximableUStatFn,
    BaseApproximationState,
)

P = TypeVar("P", bound=BaseConditionalModel)


class CgofMMDApproximationState(BaseApproximationState):
    p_expectations: Union[GaussianConditionalModel, DiscretizedModel[Any]]


BM = TypeVar("BM", bound=BaseConditionalModel)

CgofMMDOneSample_T: TypeAlias = Tuple[Array, Array, BM]


class OneSampleCgofMMDUStatFn(
    ApproximableUStatFn[CgofMMDOneSample_T[BM], CgofMMDApproximationState],
    Generic[BM],
):
    r"""U-statistic for the MMD Conditional Goodness-of-Fit test.

    See [1]


    The complete U-statistics takes the form
    .. math::
        \sum_{1 \leq i \ne j \leq n} k_p(x_i, x_j) h((x_i, y_i), (x_j, y_j))

    Where :math:`h` is (assuming tensor-product kernels)
    .. math::
        h((x, y), (x', y')) = k_y(y, y')
            - \mathbb E_{Y \sim \mu(x)} k_y(Y, y')
            - \mathbb E_{Y' \sim \mu(x')} k_y(y, Y')
            - \mathbb E_{Z \sim p, Z' \sim p'} k_y(Z, Z')


    [1] Kernel-Based Evaluation of Conditional Biological Sequence Models
    (2023, In Prep)
    """
    x_kernel: base_kernel[Array]
    y_kernel: base_kernel[Array]

    def __call__(
        self,
        z: CgofMMDOneSample_T[BM],
        z_p: CgofMMDOneSample_T[BM],
    ) -> Scalar:
        (x, y, p_x), (x_p, y_p, p_x_p) = (z, z_p)
        logger = get_logger("calibration.statistical_tests.cgof_mmd")
        assert isinstance(p_x, (DiscretizedModel, GaussianConditionalModel))
        assert isinstance(p_x_p, (DiscretizedModel, GaussianConditionalModel))

        if isinstance(p_x, GaussianConditionalModel):
            logger.info("gaussian")
            assert isinstance(p_x_p, GaussianConditionalModel)
            assert isinstance(self.y_kernel, gaussian_kernel)
            t1 = self.y_kernel(y, y_p)
            t2 = p_x.analytical_expectation_of(self.y_kernel, y_p)
            t3 = p_x_p.analytical_expectation_of(self.y_kernel, y)
            t4 = p_x.bivariate_analytical_expectation_of(self.y_kernel, p_x_p)
        else:
            assert isinstance(p_x, DiscretizedModel)
            assert isinstance(p_x_p, DiscretizedModel)
            t1 = self.y_kernel(y, y_p)
            t2 = p_x.analytical_expectation_of(self.y_kernel, y_p)
            t3 = p_x_p.analytical_expectation_of(self.y_kernel, y)
            t4 = p_x.bivariate_analytical_expectation_of(self.y_kernel, p_x_p)

        return cast(Scalar, self.x_kernel(x, x_p) * (t1 - t2 - t3 + t4))

    def make_approximation_state(
        self,
        z: CgofMMDOneSample_T[BM],
        num_particles: int,
        key: PRNGKeyArray,
        mcmc_config: Optional[MCMCConfig[Any, Any, Any]] = None,
    ) -> CgofMMDApproximationState:
        x, y, p_x = z
        logger = get_logger("calibration.statistical_tests.skce")
        logger.info("OneSampleSKCEUStatFn: approximating inputs")

        if isinstance(p_x, DiscretizedModel):
            # XXX: if p has a very large state space, expectations are not tractable.
            # We should allow approximation in this case.
            logger.info(
                "OneSampleSKCEUStatFn: p is discrete and already admits tractable "
                "expectations, not performing any approximation."
            )
            p_expectations = p_x
        elif isinstance(p_x, GaussianConditionalModel) and isinstance(
            self.y_kernel, gaussian_kernel
        ):
            logger.info(
                "OneSampleSKCEUStatFn: p is a GaussianConditionalModel used with "
                "a gaussian_kernel and admits tractable expectations, not performing "
                "any approximation."
            )
            p_expectations = p_x
        else:
            logger.info(
                f"OneSampleSKCEUStatFn: discretizing p ({p_x}) using {num_particles} "
                f"particles to compute SKCE expectations."
            )
            if isinstance(p_x, SampleableModel):
                logger.info(
                    "OneSampleSKCEUStatFn: p is a SampleableModel, using "
                    "p.non_mcmc_discretize to sample."
                )
                key, subkey = jax.random.split(key)
                p_expectations = p_x.non_mcmc_discretize(subkey, num_particles)
            else:
                assert isinstance(p_x, ConcreteLogDensityModel)
                logger.info(
                    "OneSampleSKCEUStatFn: p is not a SampleableModel, using "
                    "p.mcmc_discretize to sample."
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
                    loc=jnp.zeros((p_x.dimension,)), scale=1.0  # type: ignore
                )
                factory = MCMCAlgorithmFactory(config=mcmc_config)
                key, subkey = jax.random.split(key)
                p_expectations = p_x.mcmc_discretize(
                    base_dist, subkey, factory, num_particles
                )

        key, subkey = jax.random.split(key)
        return CgofMMDApproximationState(p_expectations)

    def maybe_approximate_internals(self, num_particles: int, key: PRNGKeyArray) -> Any:
        return

    def call_approximate(
        self,
        z_and_approx: Tuple[CgofMMDOneSample_T[BM], CgofMMDApproximationState],
        z_p_and_approx: Tuple[CgofMMDOneSample_T[BM], CgofMMDApproximationState],
        approx_internals: Any,
    ) -> Scalar:
        logger = get_logger("calibration.statistical_tests.skce")
        logger.info("skce: using approximate u-statistics evaluations")
        (x, y, _), (x_p, y_p, _) = (z_and_approx[0], z_p_and_approx[0])
        # used to approximate expectations of `y_kernel` under the models
        p_approx_exp = z_and_approx[1].p_expectations
        p_p_approx_exp = z_p_and_approx[1].p_expectations

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

        x_kernel_val = self.x_kernel(x, x_p)
        return cast(Scalar, x_kernel_val * (t1 - t2 - t3 + t4))


CgofMMDTestInput_T: TypeAlias = Tuple[Array, Array, BM]


class cgof_mmd_test(
    ApproximableTest[CgofMMDOneSample_T[BM], Unpack[CgofMMDTestInput_T[BM]]],
    TwoKernelStruct[base_kernel[Array], base_kernel[Array]],
):
    """Conditional Goodness-of-fit MMD tests

    ...

    See [1] for details.

    [1] Kernel-Based Evaluation of Conditional Biological Sequence Models
    (2023, In Prep)
    """

    def _get_u_stat_fn(
        self,
        X: Array,
        Y: Array,
        models: BM,
        *,
        key: jax.random.KeyArray,
    ) -> OneSampleCgofMMDUStatFn[BM]:
        logger = get_logger("calibration.statistical_tests.skce")
        # Use median heuristic if desired and possible
        x_kernel = self.x_kernel
        y_kernel = self.y_kernel
        if self.median_heuristic and (
            isinstance(x_kernel, (MedianHeuristicKernel, ApproxMedianHeuristicKernel))
            or isinstance(y_kernel, MedianHeuristicKernel)
        ):
            logger.info("skce: tuning kernel bandwidths using median heuristic.")
            indices = self.generate_indices(key, (X, Y, models))
            if isinstance(y_kernel, MedianHeuristicKernel):
                logger.info("skce: using median heuristic for y_kernel")
                y_kernel = y_kernel.with_median_heuristic(Y, indices)
                logger.info(
                    f"skce: y_kernel bandwidth after tuning: {y_kernel.bandwidth}"
                )

            if not self.approximate:
                if isinstance(x_kernel, MedianHeuristicKernel):
                    logger.info("skce: using median heuristic for p_kernel")
                    x_kernel = x_kernel.with_median_heuristic(X, indices)
                    logger.info(
                        f"skce: p_kernel bandwidth after tuning: {x_kernel.bandwidth}"
                    )
            else:
                if isinstance(x_kernel, ApproxMedianHeuristicKernel):
                    logger.info("skce: using approximate median heuristic for p_kernel")
                    key, subkey = jax.random.split(key)
                    x_kernel = x_kernel.with_approximate_median_heuristic(
                        X,
                        self.approximation_num_particles,
                        subkey,
                        indices,
                        self.get_mcmc_config(),
                    )
                    logger.info(
                        f"skce: p_kernel bandwidth after tuning: {x_kernel.bandwidth}"
                    )

        return OneSampleCgofMMDUStatFn(x_kernel, y_kernel)

    def _build_one_sample(
        self, X: Array, Y: Array, models: BM
    ) -> CgofMMDOneSample_T[BM]:
        return (X, Y, models)

    # This method directly calls its base class `__call__` but is
    # added to explicit the test's signature.
    def __call__(
        self, X: Array, Y: Array, models: BM, *, key: PRNGKeyArray
    ) -> TestResults:
        return super().__call__(X, Y, models, key=key)
