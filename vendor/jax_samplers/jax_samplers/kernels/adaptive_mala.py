from typing import Callable, Optional, Type, Union, cast

import jax
import jax.numpy as jnp
from flax import struct
from jax import grad, random
from jax.numpy.linalg import eigh
from typing_extensions import Self

from jax_samplers.pytypes import Array, LogDensity_T, Numeric, PRNGKeyArray
from jax_samplers.kernels.base import (Array_T, Info, Kernel, KernelConfig,
                                           MHKernelFactory, Result, State)


class AdaptiveMALAConfig(KernelConfig):
    step_size: float = struct.field(pytree_node=True)
    update_sigma: bool = struct.field(pytree_node=False, default=True)
    update_cov: bool = struct.field(pytree_node=False, default=False)
    use_exponential_average: bool = struct.field(pytree_node=False, default=True)
    use_dense_cov: bool = struct.field(pytree_node=False, default=False)


def get_schedule(use_exponential_average: bool) -> Callable[[int], float]:
    if use_exponential_average:
        return lambda _: 0.01
    else:
        return lambda iter_no: 1 / iter_no


class AdaptiveMALAState(State):
    iter_no: int
    mu: Array
    C: Array
    sigma: Numeric
    target_accept_rate: float = 0.5

    @property
    def has_diagonal_cov(self) -> bool:
        return len(self.C.shape) == 1

    @staticmethod
    def p1(sigma: Numeric):
        C1 = 1e-4
        C2 = 1
        return jnp.clip(sigma, a_min=C1, a_max=C2)

    @staticmethod
    def p2(C):
        C1 = 1e-8
        C2 = 1.
        if len(C.shape) == 2:
            eigvals, eigvecs = eigh(C)
            return (
                eigvecs
                @ jnp.diag(jnp.clip(jnp.real(eigvals), a_min=C1, a_max=C2))
                @ eigvecs.T
            )
        else:
            return jnp.clip(C, a_min=C1, a_max=C2)

    @staticmethod
    def p3(mu):
        C1 = -40.0
        C2 = 40.0
        return jnp.clip(mu, a_min=C1, a_max=C2)

    def _update_stats(
        self,
        x: Array,
        log_alpha: Numeric,
        gamma_n: Numeric,
        update_cov: bool = False,
        update_sigma: bool = False,
    ) -> Self:
        new_state = self

        if update_sigma:
            alpha = jnp.exp(jnp.clip(log_alpha, a_max=0.0, a_min=-50))
            alpha = jnp.nan_to_num(alpha, 0.001)

            # new_sigma = self.sigma
            # new_sigma = jax.lax.cond((alpha < 1e-3), lambda: 0.1 * new_sigma, lambda: new_sigma)
            # new_sigma = jax.lax.cond((alpha >= 1e-3) * (alpha < 0.1), lambda: 0.5 * new_sigma, lambda: new_sigma)
            # new_sigma = jax.lax.cond((alpha >= 0.1) * (alpha < 0.4), lambda: 0.9 * new_sigma, lambda: new_sigma)
            # new_sigma = jax.lax.cond((alpha >= 0.6) * (alpha < 0.9), lambda: 1.1 * new_sigma, lambda: new_sigma)
            # new_sigma = jax.lax.cond((alpha >= 0.9) * (alpha < 0.999), lambda: 2 * new_sigma, lambda: new_sigma)
            # new_sigma = jax.lax.cond((alpha >= 0.999), lambda: 10 * new_sigma, lambda: new_sigma)

            new_sigma = new_state.p1(new_state.sigma + gamma_n * (alpha - self.target_accept_rate))
            # new_sigma = new_state.p1(new_state.sigma + gamma_n * (alpha - 0.5))
            new_state = new_state.replace(sigma=new_sigma)

        if update_cov:
            new_mu = new_state.p3((1 - gamma_n) * self.mu + gamma_n * x)
            if self.has_diagonal_cov:
                # new_C = new_state.p3((1 - gamma_n) * self.C + gamma_n * (x - self.mu) ** 2 )
                new_C = new_state.p2(
                    (1 - gamma_n) * self.C + gamma_n * (x - self.mu) ** 2
                )
            else:
                new_C = new_state.p2(
                    (1 - gamma_n) * self.C
                    + gamma_n * (jnp.outer(x - self.mu, x - self.mu))
                )
            new_state = new_state.replace(C=new_C, mu=new_mu)

        return new_state

    def update(
        self,
        x: Array,
        log_alpha: Numeric,
        gamma_n: Numeric,
        update_cov: bool = False,
        update_sigma: bool = False,
    ) -> Self:
        self = self._update_stats(x, log_alpha, gamma_n, update_cov, update_sigma)
        self = self.replace(x=x, iter_no=self.iter_no + 1)
        return self

    def update_sigma(self, log_alpha: Numeric, gamma_n: Numeric):
        return self._update_stats(self.x, log_alpha, gamma_n, update_sigma=True)

    def update_cov(self, x: Array, gamma_n):
        return self._update_stats(x, 0.0, gamma_n, update_cov=True)


class AdaptiveMALAInfo(Info):
    accept: Numeric
    log_alpha: Numeric


def sqrtm(m: Array) -> Array:
    eigvals, eigvecs = eigh(m)
    return eigvecs @ jnp.diag(jnp.real(eigvals) ** 0.5) @ eigvecs.T


class PreconditionedMALAProposalDist(struct.PyTreeNode):
    """
    From Atchade, 2006
    Conditional distribution q(xᵢ₊₁|xᵢ) = 𝒩(xᵢ + σ²ᵢΛᵢ∇log p(xᵢ), 2σ²ᵢΛᵢI))
    """

    target_log_prob: LogDensity_T
    mu: Array
    C: Array
    sigma: Numeric

    def mean(self, x_cond: Array_T) -> Array_T:
        return x_cond + self.sigma**2 * self.C @ grad(self.target_log_prob)(x_cond)

    def cov_mat(self, x_cond: Array_T) -> Array_T:
        return 2 * self.sigma**2 * (self.C + 1e-4 * jnp.eye(self.C.shape[0]))

    def sample(self, x_cond: Array_T, key: PRNGKeyArray) -> Array_T:
        """Sample from q(.|x_cond)"""
        noise = random.normal(key, x_cond.shape)
        return self.mean(x_cond) + jnp.real(sqrtm(self.cov_mat(x_cond))) @ noise

    def log_prob(self, x: Array_T, x_cond: Array_T) -> Numeric:
        """Evaluate q(x|x_cond)"""
        # XXX: no normalizer - does not matter for now since accept prob relies
        # on a ratio of this log_prob
        mean = self.mean(x_cond)
        inv_cov_mat = jnp.linalg.inv(self.cov_mat(x_cond))

        return -1 / 2 * (x - mean) @ inv_cov_mat @ (x - mean)


class PreconditionedMALAProposalDistDiagCov(struct.PyTreeNode):
    """
    From Atchade, 2006
    Conditional distribution q(xᵢ₊₁|xᵢ) = 𝒩(xᵢ + σ²ᵢΛᵢ∇log p(xᵢ), 2σ²ᵢΛᵢI))
    """

    target_log_prob: LogDensity_T
    mu: Array
    C: Array
    sigma: Numeric

    @property
    def reg_C(self) -> Array_T:
        return self.C + 1e-6

    def mean(self, x_cond: Array_T) -> Array_T:
        return x_cond + self.sigma**2 * self.reg_C * grad(self.target_log_prob)(x_cond)

    def var(self, x_cond: Array_T) -> Numeric:
        return 2 * self.sigma**2 * self.reg_C

    def sample(self, x_cond: Array_T, key: PRNGKeyArray) -> Array_T:
        """Sample from q(.|x_cond)"""
        noise = random.normal(key, x_cond.shape)
        return self.mean(x_cond) + jnp.sqrt(self.var(x_cond)) * noise

    def log_prob(self, x: Array_T, x_cond: Array_T) -> Numeric:
        """Evaluate q(x|x_cond)"""
        # XXX: no normalizer - does not matter for now since accept prob relies
        # on a ratio of this log_prob
        mean = self.mean(x_cond)
        var = self.var(x_cond)
        return -1 / 2 * jnp.dot((x - mean) / var, (x - mean))





class PreconditionedMALAProposalDistNoCov(struct.PyTreeNode):
    """
    From Atchade, 2006
    Conditional distribution q(xᵢ₊₁|xᵢ) = 𝒩(xᵢ + σ²ᵢΛᵢ∇log p(xᵢ), 2σ²ᵢΛᵢI))
    """

    target_log_prob: LogDensity_T
    mu: Array
    sigma: Numeric

    def mean(self, x_cond: Array_T) -> Array_T:
        return x_cond + self.sigma**2 * grad(self.target_log_prob)(x_cond)

    def var(self, x_cond: Array_T) -> Numeric:
        return 2 * self.sigma**2

    def sample(self, x_cond: Array_T, key: PRNGKeyArray) -> Array_T:
        """Sample from q(.|x_cond)"""
        noise = random.normal(key, x_cond.shape)
        return self.mean(x_cond) + jnp.sqrt(self.var(x_cond)) * noise

    def log_prob(self, x: Array_T, x_cond: Array_T) -> Numeric:
        """Evaluate q(x|x_cond)"""
        # XXX: no normalizer - does not matter for now since accept prob relies
        # on a ratio of this log_prob
        mean = self.mean(x_cond)
        var = self.var(x_cond)
        return -1 / (2 * var) * jnp.dot((x - mean), (x - mean))


class AdaptiveMALAKernel(
    Kernel[AdaptiveMALAConfig, AdaptiveMALAState, AdaptiveMALAInfo]
):
    schedule: Callable[[int], float]

    @classmethod
    def create(
        cls: Type[Self], target_log_prob: LogDensity_T, config: AdaptiveMALAConfig
    ) -> Self:
        schedule = get_schedule(config.use_exponential_average)
        return cls(target_log_prob=target_log_prob, config=config, schedule=schedule)

    def init_state(self, x0: Array):
        if self.config.update_cov:
            if self.config.use_dense_cov:
                return AdaptiveMALAState(
                    x0, 0, x0, jnp.eye(x0.shape[0]), jnp.sqrt(self.config.step_size)
                )
            else:
                return AdaptiveMALAState(
                    x0, 0, x0, jnp.ones((x0.shape[0],)), jnp.sqrt(self.config.step_size)
                )
        else:
            return AdaptiveMALAState(
                x0, 0, None, None, jnp.sqrt(self.config.step_size)
            )

    def get_proposal_dist(
        self, mu: Array, C: Array, sigma: Optional[Numeric] = None
    ) -> Union[PreconditionedMALAProposalDist, PreconditionedMALAProposalDistNoCov, PreconditionedMALAProposalDistDiagCov]:
        if self.config.update_cov:
            assert self.config.update_sigma
            if self.config.use_dense_cov:
                return PreconditionedMALAProposalDist(
                    target_log_prob=self.target_log_prob, mu=mu, C=C, sigma=sigma
                )
            else:
                return PreconditionedMALAProposalDistDiagCov(
                    target_log_prob=self.target_log_prob, mu=mu, C=C, sigma=sigma
                )
        else:
            return PreconditionedMALAProposalDistNoCov(
                target_log_prob=self.target_log_prob,
                mu=mu,
                sigma=sigma,
            )

    def _compute_accept_prob(
        self,
        q: Union[PreconditionedMALAProposalDist, PreconditionedMALAProposalDistNoCov, PreconditionedMALAProposalDistDiagCov],
        proposal: Array,
        x: Array_T,
    ) -> Numeric:
        """Compute α = min(1, (p(xᵢ₊₁)q(xᵢ | xᵢ₊₁)) / (p(xᵢ) q(xᵢ₊₁ | xᵢ)))"""
        log_q_prev_given_new = q.log_prob(x=x, x_cond=proposal)
        log_q_new_given_prev = q.log_prob(x=proposal, x_cond=x)

        log_alpha = (
            self.target_log_prob(proposal)
            + log_q_prev_given_new
            - self.target_log_prob(x)
            - log_q_new_given_prev
        )
        log_alpha = jnp.nan_to_num(log_alpha, neginf=-1e80, posinf=-1e80, nan=-1e80)
        log_alpha = jnp.clip(log_alpha, a_min=-500, a_max=0)
        return log_alpha

    def maybe_accept_proposal(
        self,
        q: Union[PreconditionedMALAProposalDist, PreconditionedMALAProposalDistNoCov, PreconditionedMALAProposalDistDiagCov],
        proposal: Array,
        x: AdaptiveMALAState,
        key: PRNGKeyArray,
    ) -> Result[AdaptiveMALAState, AdaptiveMALAInfo]:
        log_alpha = self._compute_accept_prob(q, proposal=proposal, x=x.x)
        accept = jnp.log(random.uniform(key=key)) < log_alpha
        accept = cast(int, accept)

        new_x = jax.lax.cond(accept, lambda: proposal, lambda: x.x)

        gamma_n = self.schedule(x.iter_no)
        new_state = x.update(
            new_x, log_alpha, gamma_n, self.config.update_cov, self.config.update_sigma
        )
        return Result(new_state, AdaptiveMALAInfo(accept=accept, log_alpha=log_alpha))

    def one_step(
        self, x: AdaptiveMALAState, key: PRNGKeyArray
    ) -> Result[AdaptiveMALAState, AdaptiveMALAInfo]:
        key_proposal, key_accept, key = random.split(key, num=3)
        q = self.get_proposal_dist(x.mu, x.C, x.sigma)
        proposal = q.sample(x_cond=x.x, key=key_proposal)
        return self.maybe_accept_proposal(q, proposal, x, key=key_accept)


class AdaptiveMALAKernelFactory(
    MHKernelFactory[AdaptiveMALAConfig, AdaptiveMALAState, AdaptiveMALAInfo]
):
    kernel_cls: Type[AdaptiveMALAKernel] = struct.field(
        pytree_node=False, default=AdaptiveMALAKernel
    )

    def build_kernel(self, log_prob: LogDensity_T) -> AdaptiveMALAKernel:
        return self.kernel_cls.create(log_prob, self.config)
