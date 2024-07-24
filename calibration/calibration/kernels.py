from abc import abstractmethod
from typing import Any, Generic, List, Optional, Tuple, Type, TypeVar, cast

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array, vmap
from jax_samplers.inference_algorithms.mcmc.base import MCMCAlgorithmFactory, MCMCConfig
from kwgflows.divergences.mmd import mmd as scalar_mmd
from kwgflows.pytypes import Array, PRNGKeyArray, Scalar
from kwgflows.rkhs.kernels import (
    BandwidthBasedKernel,
    GaussianRBFKernel,
    MedianHeuristicKernel,
    base_kernel,
    gaussian_kernel,
)
from kwgflows.rkhs.rkhs import rkhs_element
from kwgflows.utils import infer_num_samples_pytree
from numpyro import distributions as np_distributions
from typing_extensions import Self

from calibration.conditional_models.base import (
    BaseConditionalModel,
    ConcreteLogDensityModel,
    DiscretizedModel,
    LogDensityModel,
    SampleableModel,
)
from calibration.conditional_models.gaussian_models import (
    GaussianConditionalModel,
    generalized_fisher_divergence_gaussian_model,
    kernelized_generalized_fisher_divergence_gaussian_model,
    mmd_gaussian_kernel,
)
from calibration.logging import get_logger
from calibration.utils import (
    GradLogDensity_T,
    median_median_heuristic_discrete,
    median_median_heuristic_euclidean_gaussian,
)


class discretized_generalized_fisher_divergence_base(struct.PyTreeNode):
    samples: Array
    base_dist: np_distributions.Distribution = struct.field(pytree_node=False)
    squared: bool = True


class discretized_generalized_fisher_divergence(
    discretized_generalized_fisher_divergence_base
):
    r"""
    Class that represents a discretized approximation of the generalized Fisher
    divergence with respect to some given base measure.

    ...

    The generalized Fisher divergence between two densities :math:`p` and :math:`q`
    with respect to a base measure :math:`\nu` is defined as

    .. math:: \int \|\nabla \log q(x) - \nabla \log p(x)\|^2 \nu(\mathrm{d}x).

    Its computable approximation is implemented by sampling from the base measure,
    and averaging the integrand over the samples.
    """

    @classmethod
    def create(
        cls: Type[Self],
        base_dist: np_distributions.Distribution,
        key: PRNGKeyArray,
        num_samples: int = 1000,
        squared: bool = True,
    ) -> Self:
        """Create a :class:`generalized_fisher_divergence` instance with base measure
        :param:`base_dist` that is approximated using :param:`num_samples` samples.

        :param numpyro.distributions.Distribution base_dist: Base distribution that is approximated
          by samples.
        :param jax.random.KeyArray key: Pseudo-random number generator (PRNG) key used for drawing
          samples from the base distribution.
        :param int num_samples: Number of samples drawn from the base distribution. Defaults to
          1000.
        """
        # discretize the base measure
        samples = cast(Array, base_dist.sample(key, (num_samples,)))

        return cls(samples, base_dist, squared)

    def __call__(self, score_p: GradLogDensity_T, score_q: GradLogDensity_T) -> Scalar:
        """Approximate the generalized Fisher divergence between distributions :math:`p` and
        :math:`q` with score functions :param:`score_p` and :param:`score_q`.

        :param calibration.utils.GradLogDensity_T score_p: Score function of distribution :math:`p`.
        :param calibration.utils.GradLogDensity_T score_q: Score function of distribution :math:`q`.
        """
        ret = jnp.average(
            vmap(lambda x: jnp.sum(jnp.square(score_p(x) - score_q(x))))(self.samples),
        )
        return jax.lax.cond(self.squared, lambda x: x, lambda x: jnp.sqrt(x), ret)


SM = TypeVar("SM", bound=SampleableModel)
BM = TypeVar("BM", bound=BaseConditionalModel)
BM2 = TypeVar("BM2", bound=BaseConditionalModel)
K_co = TypeVar("K_co", bound=base_kernel[Any], covariant=True)


class kernelized_discretized_generalized_fisher_divergence(
    discretized_generalized_fisher_divergence_base,
):
    r"""
    Class that represents a discretized approximation of the kernelized generalized
    Fisher divergence with respect to some given base measure :math:`\nu`.

    ...

    The kernelized generalized Fisher divergence between two densities :math:`p` and
    :math:`q` with respect to a base measure :math:`\nu` is defined as

    .. math:: \|\phi p - \phi q \|^2_{\mathcal H}

    Where \phi: p \mapsto \int \nabla \log p(x) k(x, \cdot)\nu(\mathrm{d}x)
    is a feature map with outputs in a reproducing kernel Hilbert space (RKHS)
    with positive definite kernel :math:`k`.

    Its computable approximation is implemented by sampling from the base measure
    :math:`\nu`, replacing the population integral within :math:`\phi` by its sample
    average, and computing the RKHS norm of the diffrence feature map using RKHS algebra
    rules.
    """
    ground_space_kernel: base_kernel[Array] = gaussian_kernel.create()

    @classmethod
    def create(
        cls: Type[Self],
        base_dist: np_distributions.Distribution,
        key: PRNGKeyArray,
        ground_space_kernel: base_kernel[Array] = gaussian_kernel.create(),
        num_samples: int = 1000,
        squared: bool = True,
    ) -> Self:
        """Create a :class:`generalized_fisher_divergence` instance with base measure
        :param:`base_dist` that is approximated using :param:`num_samples` samples.

        :param numpyro.distributions.Distribution base_dist: Base distribution that is approximated
          by samples.
        :param jax.random.KeyArray key: Pseudo-random number generator (PRNG) key used for drawing
          samples from the base distribution.
        :param int num_samples: Number of samples drawn from the base distribution. Defaults to
          1000.
        """
        # discretize the base measure
        samples = cast(Array, base_dist.sample(key, (num_samples,)))
        return cls(samples, base_dist, squared, ground_space_kernel)

    def __call__(self, score_p: GradLogDensity_T, score_q: GradLogDensity_T) -> Scalar:
        """Approximate the kernelized generalized Fisher divergence between distributions
        :math:`p` and :math:`q` with score functions :param:`score_p` and :param:`score_q`.

        :param calibration.utils.GradLogDensity_T score_p: Score function of distribution :math:`p`.
        :param calibration.utils.GradLogDensity_T score_q: Score function of distribution :math:`q`.
        """
        # score functions of p and q
        score_diff = vmap(lambda x: score_p(x) - score_q(x))(self.samples).reshape(
            self.samples.shape[0], -1
        )
        kernelized_diff_per_dim = vmap(
            lambda ws: rkhs_element(
                self.samples, ws / self.samples.shape[0], self.ground_space_kernel
            ).rkhs_norm(squared=True),
            in_axes=1,
        )(score_diff)
        return jax.lax.cond(
            self.squared,
            lambda x: x,
            lambda x: jnp.sqrt(x),
            jnp.sum(kernelized_diff_per_dim),
        )


class kernel_on_models(base_kernel[BM], Generic[BM, K_co, BM2]):
    """Approximable kernels between probability distributions

    ...

    `kernel_on_models` are kernels that take inputs of type `BM`, and whose evaluations
    k(x, y) can be approximated to h(x', y'), where h is a kernel of type `K_co` and
    x' and y' are of type BM2.
    K_co is made generic to make it specialize to `MedianHeuristicKernel` when needed.
    A more compact and precise typing rule would be `Generic[BM, K_co[BM2]]`, however
    python does not support type variables parametrized by other type variables.
    Thus, we decorrelate `K_co` and `BM2`, and re-correlate them when needed, as in
    `maybe_approximate_input`, or in subclasses declarations.
    In most cases `kernel_on_models[BM, base_kernel[BM2], BM2]` is enough to provide
    type safety.
    """

    @abstractmethod
    def maybe_approximate_input(
        self: "kernel_on_models[BM, base_kernel[BM2], BM2]",
        x: BM,
        num_particles: int,
        key: PRNGKeyArray,
        mcmc_config: Optional[MCMCConfig[Any, Any, Any]] = None,
    ) -> BM2:
        raise NotImplementedError

    @abstractmethod
    def maybe_approximate_kernel(self, num_particles: int, key: PRNGKeyArray) -> K_co:
        raise NotImplementedError


class TractableKernelBase(kernel_on_models[BM, K_co, BM], struct.PyTreeNode):
    def maybe_approximate_kernel(self, num_particles: int, key: PRNGKeyArray) -> K_co:
        logger = get_logger("calibration.kernels")
        logger.warning(
            f"maybe_approximate_kernel: calling an approximation procedure on a "
            f"tractable kernel {self} is valid but discouraged as it does not perform any "
            f"approximation."
        )
        return cast(K_co, self)

    def maybe_approximate_input(
        self,
        x: BM,
        num_particles: int,
        key: PRNGKeyArray,
        mcmc_config: Optional[MCMCConfig[Any, Any, Any]] = None,
    ) -> BM:
        logger = get_logger("calibration.kernels")
        logger.warning(
            "maybe_approximate_input: calling an approximation procedure on a "
            "tractable kernel is valid but discouraged as it does not perform any"
            "approximation."
        )
        return x


class ApproxMedianHeuristicKernel(
    kernel_on_models[BM, MedianHeuristicKernel[BM2], BM2], BandwidthBasedKernel[BM]
):
    def with_approximate_median_heuristic(
        self,
        X: BM,
        num_particles: int,
        key: PRNGKeyArray,
        indices: Optional[Tuple[Array, Array]] = None,
        mcmc_config: Optional[MCMCConfig[Any, Any, Any]] = None,
        allow_nans: bool = False,
    ) -> Self:
        logger = get_logger("calibration.kernels")
        if isinstance(self, TractableKernelBase):
            logger.warning(
                "with_approximate_median_heuristic: calling an approximation "
                "procedure on a tractable kernel is valid but discouraged as it "
                "does not perform any approximation."
            )

        key, subkey = jax.random.split(key)
        approx_k = self.maybe_approximate_kernel(num_particles, subkey)
        key, subkey = jax.random.split(key)
        n = infer_num_samples_pytree(X)
        subkeys = jax.random.split(subkey, n)
        Xs = vmap(self.maybe_approximate_input, in_axes=(0, None, 0, None))(
            X, num_particles, subkeys, mcmc_config
        )
        new_sigma = approx_k.with_median_heuristic(
            Xs, indices, allow_nans=allow_nans
        ).bandwidth
        return self.set_bandwidth(new_sigma)


class ExpMMDKernelBase(GaussianRBFKernel[BM]):
    ground_space_kernel: base_kernel[Array]

    @classmethod
    def create(
        cls: Type[Self],
        *args: List[Any],
        sigma: float = 1.0,
        ground_space_kernel: Optional[base_kernel[Array]] = None,
        **kwargs: dict[str, Any],
    ) -> Self:
        assert ground_space_kernel is not None
        assert (args, kwargs) == ((), {}), "No positional or keyword arguments allowed"
        return cls(sigma, ground_space_kernel)


class ExpMMDKernel(
    ApproxMedianHeuristicKernel[SM, DiscretizedModel[Any]], ExpMMDKernelBase[SM]
):
    ground_space_kernel: base_kernel[Array]

    def squared_distance(self, x: SM, y: SM) -> Scalar:
        raise NotImplementedError

    def maybe_approximate_kernel(
        self, num_particles: int, key: PRNGKeyArray
    ) -> "DiscreteExpMMDKernel":
        logger = get_logger("calibration.kernels")
        logger.info("ExpMMDGaussianKernel: approximating into DiscreteExpMMDKernel")
        return DiscreteExpMMDKernel.create(
            sigma=self.sigma, ground_space_kernel=self.ground_space_kernel
        )

    def maybe_approximate_input(
        self,
        x: BaseConditionalModel,
        num_particles: int,
        key: PRNGKeyArray,
        mcmc_config: Optional[MCMCConfig[Any, Any, Any]] = None,
    ) -> DiscretizedModel[Any]:
        logger = get_logger("calibration.kernels")
        logger.info(f"ExpMMDKernel: discretizing x using {num_particles} particles")
        if isinstance(x, SampleableModel):
            logger.info("ExpMMDKernel: approximating input by sampling from the model")
            return x.non_mcmc_discretize(key, num_particles)
        else:
            assert isinstance(x, ConcreteLogDensityModel)
            logger.info(
                "ExpMMDKernel: model is not sampleable, approximating input using "
                "MCMC"
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
                loc=jnp.zeros((x.dimension,)), scale=1.0  # type: ignore
            )
            factory = MCMCAlgorithmFactory(config=mcmc_config)
            key, subkey = jax.random.split(key)
            return x.mcmc_discretize(base_dist, subkey, factory, num_particles)


class DiscreteExpMMDKernel(
    ApproxMedianHeuristicKernel[DiscretizedModel[Any], DiscretizedModel[Any]],
    TractableKernelBase[DiscretizedModel[Any], "DiscreteExpMMDKernel"],
    ExpMMDKernelBase[DiscretizedModel[Any]],
):
    def with_median_heuristic(
        self,
        X: DiscretizedModel[Any],
        indices: Optional[Tuple[Array, Array]] = None,
        quantile: Optional[float] = None,
        inner_indices: Optional[Tuple[Array, Array]] = None,
        *,
        allow_nans: bool = False,
    ) -> Self:
        kernel = self.ground_space_kernel
        if isinstance(kernel, MedianHeuristicKernel):
            # We also apply the median heuristic to the ground-space kernel
            # if possible.
            sigma = median_median_heuristic_discrete(
                kernel, X.X, indices, quantile, inner_indices, allow_nans=allow_nans
            )
            kernel = kernel.set_bandwidth(sigma)
        return super(
            DiscreteExpMMDKernel, self.replace(ground_space_kernel=kernel)
        ).with_median_heuristic(X, indices, quantile, allow_nans=allow_nans)

    def squared_distance(
        self, x: DiscretizedModel[Any], y: DiscretizedModel[Any]
    ) -> Scalar:
        p, q = x, y
        return (
            scalar_mmd(self.ground_space_kernel, squared=True)(p, q) / self.sigma**2
        )


class ExpMMDGaussianKernel(
    ApproxMedianHeuristicKernel[GaussianConditionalModel, GaussianConditionalModel],
    TractableKernelBase[GaussianConditionalModel, "ExpMMDGaussianKernel"],
    ExpMMDKernelBase[GaussianConditionalModel],
):
    ground_space_kernel: gaussian_kernel

    def with_median_heuristic(
        self,
        X: GaussianConditionalModel,
        indices: Optional[Tuple[Array, Array]] = None,
        quantile: Optional[float] = None,
        *,
        allow_nans: bool = False,
    ) -> Self:
        # We also apply the median heuristic to the ground-space kernel.
        sigma = median_median_heuristic_euclidean_gaussian(X, indices, quantile)
        kernel = gaussian_kernel.create(sigma=sigma)
        return super(
            ExpMMDGaussianKernel, self.replace(ground_space_kernel=kernel)
        ).with_median_heuristic(X, indices, quantile, allow_nans=allow_nans)

    def squared_distance(
        self, x: GaussianConditionalModel, y: GaussianConditionalModel
    ) -> Scalar:
        p, q = x, y
        return (
            mmd_gaussian_kernel(self.ground_space_kernel, squared=True)(p, q)
            / self.sigma**2
        )


M = TypeVar("M", bound=LogDensityModel)


class ExpFisherKernel(
    ApproxMedianHeuristicKernel[M, M],
    GaussianRBFKernel[M],
):
    base_dist: np_distributions.Distribution = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls: Type[Self],
        *args: List[Any],
        base_dist: np_distributions.Normal,
        sigma: float = 1.0,
        **kwargs: dict[str, Any],
    ) -> Self:
        assert (args, kwargs) == ((), {}), "No positional or keyword arguments allowed"
        return cls(sigma, base_dist)

    def squared_distance(self, x: M, y: M) -> Scalar:
        raise NotImplementedError

    def maybe_approximate_kernel(
        self, num_particles: int, key: PRNGKeyArray
    ) -> "DiscreteExpFisherKernel[M]":
        return DiscreteExpFisherKernel.create(
            base_dist=self.base_dist,
            key=key,
            sigma=self.sigma,
            num_samples=num_particles,
        )

    def maybe_approximate_input(
        self,
        x: M,
        num_particles: int,
        key: PRNGKeyArray,
        mcmc_config: Optional[MCMCConfig[Any, Any, Any]] = None,
    ) -> M:
        return x


class ExpKernelizedFisherKernel(
    ApproxMedianHeuristicKernel[M, M],
    GaussianRBFKernel[M],
):
    base_dist: np_distributions.Distribution = struct.field(pytree_node=False)
    ground_space_kernel: base_kernel[Array]

    @classmethod
    def create(
        cls: Type[Self],
        *args: List[Any],
        base_dist: np_distributions.Normal,
        ground_space_kernel: base_kernel[Array],
        sigma: float = 1.0,
        **kwargs: dict[str, Any],
    ) -> Self:
        assert (args, kwargs) == ((), {}), "No positional or keyword arguments allowed"
        return cls(sigma, base_dist, ground_space_kernel)

    def squared_distance(self, x: M, y: M) -> Scalar:
        raise NotImplementedError

    def maybe_approximate_kernel(
        self, num_particles: int, key: PRNGKeyArray
    ) -> "DiscreteExpKernelizedFisherKernel[M]":
        return DiscreteExpKernelizedFisherKernel.create(
            base_dist=self.base_dist,
            key=key,
            sigma=self.sigma,
            num_samples=num_particles,
            ground_space_kernel=self.ground_space_kernel,
        )

    def maybe_approximate_input(
        self,
        x: M,
        num_particles: int,
        key: PRNGKeyArray,
        mcmc_config: Optional[MCMCConfig[Any, Any, Any]] = None,
    ) -> M:
        return x


class DiscreteExpFisherKernel(
    ApproxMedianHeuristicKernel[M, M],
    TractableKernelBase[M, "DiscreteExpFisherKernel"],
    GaussianRBFKernel[M],
):
    fd: discretized_generalized_fisher_divergence

    @classmethod
    def create(
        cls: Type[Self],
        *args: List[Any],
        base_dist: np_distributions.Distribution,
        key: PRNGKeyArray,
        sigma: float = 1.0,
        num_samples: int = 1000,
        **kwargs: dict[str, Any],
    ) -> Self:
        assert (args, kwargs) == ((), {}), "No positional or keyword arguments allowed"
        fd = discretized_generalized_fisher_divergence.create(
            base_dist, key, num_samples, squared=True
        )
        return cls(sigma, fd)

    def squared_distance(self, x: M, y: M) -> Scalar:
        assert self.fd is not None
        score_p = x.score
        score_q = y.score
        return cast(Scalar, self.fd(score_p, score_q) / self.sigma**2)


class DiscreteExpKernelizedFisherKernel(
    ApproxMedianHeuristicKernel[M, M],
    TractableKernelBase[M, "DiscreteExpKernelizedFisherKernel"],
    GaussianRBFKernel[M],
):
    fd: kernelized_discretized_generalized_fisher_divergence

    @classmethod
    def create(
        cls: Type[Self],
        *args: List[Any],
        base_dist: np_distributions.Distribution,
        key: PRNGKeyArray,
        ground_space_kernel: base_kernel[Array] = gaussian_kernel.create(),
        sigma: float = 1.0,
        num_samples: int = 1000,
        **kwargs: dict[str, Any],
    ) -> Self:
        assert (args, kwargs) == ((), {}), "No positional or keyword arguments allowed"
        fd = kernelized_discretized_generalized_fisher_divergence.create(
            base_dist, key, ground_space_kernel, num_samples, squared=True
        )
        return cls(sigma, fd)

    def squared_distance(self, x: M, y: M) -> Scalar:
        assert self.fd is not None
        score_p = x.score
        score_q = y.score
        return cast(Scalar, self.fd(score_p, score_q) / self.sigma**2)


class GaussianExpFisherKernel(
    ApproxMedianHeuristicKernel[GaussianConditionalModel, LogDensityModel],
    TractableKernelBase[GaussianConditionalModel, "GaussianExpFisherKernel"],
    GaussianRBFKernel[GaussianConditionalModel],
):
    fd: generalized_fisher_divergence_gaussian_model

    @classmethod
    def create(
        cls: Type[Self],
        *args: List[Any],
        base_dist: np_distributions.Normal,
        sigma: float = 1.0,
        **kwargs: dict[str, Any],
    ) -> Self:
        assert (args, kwargs) == ((), {}), "No positional or keyword arguments allowed"
        fd = generalized_fisher_divergence_gaussian_model.create(
            base_dist, squared=True
        )
        return cls(sigma, fd)

    def squared_distance(
        self, x: GaussianConditionalModel, y: GaussianConditionalModel
    ) -> Scalar:
        log_p, log_q = x, y
        assert self.fd is not None
        return cast(Scalar, self.fd(log_p, log_q) / self.sigma**2)


class GaussianExpKernelizedFisherKernel(
    ApproxMedianHeuristicKernel[GaussianConditionalModel, GaussianConditionalModel],
    TractableKernelBase[GaussianConditionalModel, "GaussianExpKernelizedFisherKernel"],
    GaussianRBFKernel[GaussianConditionalModel],
):
    r"""
    Class that computes closed form values for the kernelized generalized Fisher
    divergence when the ground space kernel is a gaussian kernel, the base measure
    :math:`\nu` is a Gaussian measure, and the probabilities :math:`p` and :math:`q`
    are Gaussian.

    ...

    The kernelized generalized Fisher divergence between two densities :math:`p` and
    :math:`q` with respect to a base measure :math:`\nu` is defined as

    .. math:: \|\phi q - \phi \|^2_{\mathcal H}

    Where

    .. math:: \phi: p \mapsto \int \nabla \log p(x) k(x, \cdot)\nu(\mathrm{d}x)

    is a feature map with outputs in a reproducing kernel Hilbert space (RKHS).
    In the case where all quantities (:math:`p`, :math:`q`, :math:`\nu`, :math:`k`)
    are Gaussian, the kernelized generalized Fisher divergence admits a closed-form
    expression that does not require any approximation to be computed.
    """
    fd: kernelized_generalized_fisher_divergence_gaussian_model

    @classmethod
    def create(
        cls: Type[Self],
        *args: List[Any],
        base_dist: np_distributions.Normal,
        ground_space_kernel: gaussian_kernel = gaussian_kernel.create(),
        sigma: float = 1.0,
        **kwargs: dict[str, Any],
    ) -> Self:
        assert (args, kwargs) == ((), {}), "No positional or keyword arguments allowed"
        fd = kernelized_generalized_fisher_divergence_gaussian_model.create(
            base_dist, ground_space_kernel, squared=True
        )
        return cls(sigma, fd)

    def with_median_heuristic(
        self,
        X: GaussianConditionalModel,
        indices: Optional[Tuple[Array, Array]] = None,
        quantile: Optional[float] = None,
        *,
        allow_nans: bool = False,
    ) -> Self:
        # We also apply the median heuristic to the kernel of the
        # kernelized generalized Fisher divergence.
        sigma = median_median_heuristic_euclidean_gaussian(X, indices, quantile)
        kernel = gaussian_kernel.create(sigma=sigma)
        fd = self.fd.replace(ground_space_kernel=kernel)
        return super(
            GaussianExpKernelizedFisherKernel, self.replace(fd=fd)
        ).with_median_heuristic(X, indices, quantile, allow_nans=allow_nans)

    def squared_distance(
        self, x: GaussianConditionalModel, y: GaussianConditionalModel
    ) -> Scalar:
        log_p, log_q = x, y
        assert self.fd is not None
        return cast(Scalar, self.fd(log_p, log_q) / self.sigma**2)


class GaussianExpWassersteinKernel(
    ApproxMedianHeuristicKernel[GaussianConditionalModel, GaussianConditionalModel],
    TractableKernelBase[GaussianConditionalModel, "GaussianExpWassersteinKernel"],
    GaussianRBFKernel[GaussianConditionalModel],
):
    r"""
    Class that represents an Gaussian kernel with the 2-Wasserstein distance between isotropic
    Gaussian distributions.

    ...

    The 2-Wasserstein distance (with respect to the Euclidean distance) between isotropic
    Gaussian distributions :math:`P = \mathcal{N}(\mu_1, \sigma_1^2 I)` and
    :math:`Q = \mathcal{N}(\mu_2, \sigma_2^2 I)` on :math:`\mathbb{R}^d` is given by

    .. math:: W_2(P, Q) = \sqrt{\|\mu_1 - \mu_2\|_2^2 + d (\sigma_1 - \sigma_2)^2}.
    """

    @classmethod
    def create(
        cls: Type[Self],
        *args: List[Any],
        sigma: float = 1.0,
        **kwargs: dict[str, Any],
    ) -> Self:
        """Create a :class:`ExpWassersteinKernel` instance with length scale :param:`sigma`.

        :param float sigma: Length scale of the Gaussian kernel. Defaults to 1.0.
        """
        assert (args, kwargs) == ((), {}), "No positional or keyword arguments allowed"
        return cls(sigma)

    def squared_distance(
        self, P: GaussianConditionalModel, Q: GaussianConditionalModel
    ) -> Scalar:
        assert P.dimension == Q.dimension
        return cast(
            Scalar,
            jnp.sum(jnp.square((P.mu - Q.mu) / self.sigma))
            + P.dimension * ((P.sigma - Q.sigma) / self.sigma) ** 2,
        )
