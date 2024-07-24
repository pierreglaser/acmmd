from typing import cast

import jax
import jax.numpy as jnp
from flax import struct
from kwgflows.pytypes import Array, Scalar
from kwgflows.rkhs.kernels import base_kernel, gaussian_kernel
from numpyro import distributions as np_distributions
from typing_extensions import Self

from calibration.conditional_models.base import (
    DistributionModel,
    LogDensityModel,
    SampleableModel,
)


class GaussianConditionalModel(LogDensityModel, SampleableModel):
    mu: Array
    sigma: float = 1.0

    @property
    def dimension(self) -> int:
        return self.mu.shape[0]

    def sample_from_conditional(
        self, key: jax.random.KeyArray, num_samples: int
    ) -> Array:
        return jax.random.multivariate_normal(
            key,
            mean=self.mu,
            cov=self.sigma**2 * jnp.identity(self.dimension),
            shape=(num_samples,),
        )

    def __call__(self, y: Array) -> Scalar:
        return cast(
            Scalar,
            jax.scipy.stats.multivariate_normal.logpdf(
                y, mean=self.mu, cov=self.sigma**2
            ),
        )

    def score(self, y: Array) -> Array:
        return (self.mu - y) / self.sigma**2

    def analytical_expectation_of(self, k: gaussian_kernel, y: Array) -> Scalar:
        r"""Compute the analytical expectation of a Gaussian kernel :param:`k`
        with second argument fixed to :params:`y` with respect to the first argument
        which follows a Gaussian distribution :param:`self`.

        ...

        Let :math:`X \sim \mathcal{N}(\mu, \sigma^2 I)` take values in
        :math:`\mathbb{R}^d`, let :math:`y \in \mathbb{R}^d`, and consider a
        Gaussian kernel :math:`k(x, y) = \exp{(- \|x - y\|^2 / (2 \sigma_k^2))}`.
        Then we have
        .. math::
            \mathbb{E} k(X, y) =
                \frac{\exp{(- \|\mu - y\|^2 / (2 (\sigma^2 + \sigma_k^2)))}}{
                    {(1 + (\sigma / \sigma_k)^2)}^{d/2}}
        """
        s = jnp.hypot(self.sigma, k.sigma)
        log_scaling = self.dimension * jnp.log1p((self.sigma / k.sigma) ** 2)
        return cast(
            Scalar, jnp.exp(-(log_scaling + jnp.sum(jnp.square((self.mu - y) / s))) / 2)
        )

    def bivariate_analytical_expectation_of(
        self, k: gaussian_kernel, other: Self
    ) -> Scalar:
        r"""Compute the analytical expectation of a Gaussian kernel :param:`k`
        with respect to both arguments where the first argument follows a Gaussian
        distribution :param:`self` and the second argument follows a Gaussian
        distribution :param:`other`.

        ...

        Let :math:`X \sim \mathcal{N}(\mu_X, \sigma_X^2 I)` and
        :math:`Y \sim \mathcal{N}(\mu_Y, \sigma_Y^2 I)` be independent random variables
        that take values in :math:`\mathbb{R}^d`, and consider a Gaussian kernel
        :math:`k(x, y) = \exp{(- \|x - y\|^2 / (2 \sigma_k^2))}`.
        Then we have
        .. math::
            \mathbb{E} k(X, Y) =
                \frac{\exp{(- \|\mu_X - \mu_Y\|^2 / (2 (\sigma_X^2 + \sigma_Y^2 + \sigma_k^2)))}}{
                    {(1 + (\sigma_X / \sigma_k)^2 + (\sigma_Y / \sigma_k)^2)}^{d/2}}
        """
        # `hypot` is not defined for > 2 arguments
        s = jnp.sqrt(self.sigma**2 + k.sigma**2 + other.sigma**2)
        log_scaling = self.dimension * jnp.log1p(
            (self.sigma / k.sigma) ** 2 + (other.sigma / k.sigma) ** 2
        )
        return cast(
            Scalar,
            jnp.exp(-(log_scaling + jnp.sum(jnp.square((self.mu - other.mu) / s))) / 2),
        )

    def as_distribution_model(self) -> DistributionModel:
        return DistributionModel.create(
            np_distributions.Normal(self.mu, self.sigma).to_event(1)  # type: ignore
        )


class mmd_gaussian_kernel(struct.PyTreeNode):
    kernel: gaussian_kernel
    squared: bool = True

    def __call__(
        self, P: GaussianConditionalModel, Q: GaussianConditionalModel
    ) -> Scalar:
        kpp = P.bivariate_analytical_expectation_of(self.kernel, P)
        kqq = Q.bivariate_analytical_expectation_of(self.kernel, Q)
        kqp = Q.bivariate_analytical_expectation_of(self.kernel, P)
        mmd_sq = kpp + kqq - 2 * kqp
        return jax.lax.cond(
            self.squared, lambda _: mmd_sq, lambda _: jnp.sqrt(mmd_sq), None
        )


class generalized_fisher_divergence_gaussian_model(struct.PyTreeNode):
    r"""
    A special case of the generalized Fisher divergence where the base measure
    is Gaussian and that only accepts Gaussian conditional models as inputs.

    ...

    The computed divergence is not approximate in that case, because the integral
    constituting the generalized Fisher divergence admits an explicit closed-form
    expression in the case where all models involved are Gaussian.
    """
    base_dist: np_distributions.Normal = struct.field(pytree_node=False)
    squared: bool = True

    @classmethod
    def create(cls, base_dist: np_distributions.Normal, squared: bool = True) -> Self:
        assert len(base_dist.event_shape) == 0
        assert len(base_dist.batch_shape) == 1
        return cls(base_dist, squared)

    def __call__(
        self, log_p: GaussianConditionalModel, log_q: GaussianConditionalModel
    ) -> Scalar:
        assert self.base_dist.scale.shape == (1,)
        d = self.base_dist.batch_shape[0]
        t1 = (
            d * self.base_dist.scale[0] ** 2 + jnp.sum(jnp.square(self.base_dist.loc))
        ) * (1 / log_p.sigma**2 - 1 / log_q.sigma**2) ** 2
        t2 = (1 / log_p.sigma**2 - 1 / log_q.sigma**2) * jnp.dot(
            self.base_dist.loc,
            (log_p.mu / log_p.sigma**2 - log_q.mu / log_q.sigma**2),
        )
        t3 = jnp.sum(
            jnp.square(log_p.mu / log_p.sigma**2 - log_q.mu / log_q.sigma**2)
        )
        ret = t1 - 2 * t2 + t3
        return jax.lax.cond(self.squared, lambda x: x, lambda x: jnp.sqrt(x), ret)


class kernelized_generalized_fisher_divergence_gaussian_model(struct.PyTreeNode):
    r"""
    A special case of the generalized Fisher divergence where the base measure
    is Gaussian and that only accepts Gaussian conditional models as inputs.

    ...

    The computed divergence is not approximate in that case, because the integral
    constituting the generalized Fisher divergence admits an explicit closed-form
    expression in the case where all models involved are Gaussian.
    """
    base_dist: np_distributions.Normal = struct.field(pytree_node=False)
    squared: bool = True
    ground_space_kernel: gaussian_kernel = gaussian_kernel.create()

    @classmethod
    def create(
        cls,
        base_dist: np_distributions.Normal,
        ground_space_kernel: gaussian_kernel = gaussian_kernel.create(),
        squared: bool = True,
    ) -> Self:
        assert len(base_dist.event_shape) == 0
        assert len(base_dist.batch_shape) == 1

        return cls(base_dist, squared, ground_space_kernel)

    def __call__(
        self, log_p: GaussianConditionalModel, log_q: GaussianConditionalModel
    ) -> Scalar:
        sigma_p = log_p.sigma
        sigma_q = log_q.sigma
        sigma_K = self.ground_space_kernel.sigma
        sigma_b = self.base_dist.scale[0]
        mu_b = self.base_dist.loc
        mu_p = log_p.mu
        mu_q = log_q.mu

        assert mu_b.shape == mu_p.shape == mu_q.shape
        d = mu_b.shape[0]

        resolved_cross_second_moment = mu_b**2 + sigma_b**2 / (
            2 + sigma_K**2 / sigma_b**2
        )

        tpp = (resolved_cross_second_moment - mu_b * (mu_p + mu_p) + mu_p * mu_p) / (
            sigma_p**2 * sigma_p**2
        )
        tpq = (resolved_cross_second_moment - mu_b * (mu_p + mu_q) + mu_p * mu_q) / (
            sigma_p**2 * sigma_q**2
        )

        tqq = (resolved_cross_second_moment - mu_b * (mu_q + mu_q) + mu_q * mu_q) / (
            sigma_q**2 * sigma_q**2
        )
        det = (sigma_b**2 * sigma_K**2) / (2 + sigma_K**2 / sigma_b**2)

        norm_constant = 2 * jnp.pi * jnp.sqrt(det)
        norm_constant_base = 1 / jnp.sqrt((2 * jnp.pi * sigma_b**2))

        total = (tpp - 2 * tpq + tqq) * norm_constant * norm_constant_base**2

        s = jnp.sqrt(2 * sigma_b**2 + sigma_K**2)
        log_scaling = (d - 1) * jnp.log1p(2 * (sigma_b / sigma_K) ** 2)
        scaling_f = jnp.exp(-(log_scaling) / 2)
        total = jnp.sum(scaling_f * total)
        return jax.lax.cond(self.squared, lambda x: x, lambda x: jnp.sqrt(x), total)
