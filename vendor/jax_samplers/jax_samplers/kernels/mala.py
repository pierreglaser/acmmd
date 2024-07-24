from typing import Optional, Type, Union

import jax.numpy as jnp
from flax import struct
from jax import grad, random
from jax.scipy.linalg import sqrtm  # type: ignore
from typing_extensions import Self, TypeAlias

from jax_samplers.pytypes import Array, LogDensity_T, Numeric, PRNGKeyArray
from jax_samplers.kernels.base import (Array_T, Info, Kernel, KernelConfig, KernelFactory, MHKernel, State, State_T, TunableKernel, TunableMHKernelFactory)

MHSample: TypeAlias = Array


class MALAProposalDist(struct.PyTreeNode):
    """
    Conditional distribution q(xᵢ₊₁|xᵢ) = 𝒩(xᵢ + γ∇log p(xᵢ), √(2 * γ)² I)
    ⇒ log q(xᵢ₊₁|xᵢ) = - 1 / (4 * γ) || xᵢ₊₁ - xᵢ - γ∇log p(xᵢ)||²
    """

    target_log_prob: LogDensity_T
    step_size: float

    def mean(self, x_cond: MHSample) -> MHSample:
        return x_cond + self.step_size * grad(self.target_log_prob)(x_cond)

    def std(self) -> MHSample:
        return jnp.sqrt(2 * self.step_size)

    def sample(self, x_cond: MHSample, key: PRNGKeyArray) -> MHSample:
        """Sample from q(.|x_cond)"""
        noise = random.normal(key, x_cond.shape)
        return self.mean(x_cond) + self.std() * noise

    def log_prob(self, x: MHSample, x_cond: MHSample) -> Numeric:
        """Evaluate q(x|x_cond)"""
        # XXX: no normalizer - does not matter for now since accept prob relies
        # on a ratio of this log_prob
        mean = self.mean(x_cond)
        return -jnp.dot(x - mean, x - mean) / (2 * self.std() ** 2)


class PreconditionedMALAProposalDist(struct.PyTreeNode):
    """
    From Atchade, 2006
    Conditional distribution q(xᵢ₊₁|xᵢ) = 𝒩(xᵢ + σ²ᵢΛᵢ∇log p(xᵢ), 2σ²ᵢΛᵢI))
    """

    target_log_prob: LogDensity_T
    C: Array
    sigma: Numeric

    @property
    def reg_C(self) -> Array_T:
        return self.C + 1e-6 * jnp.eye(self.C.shape[0])

    def mean(self, x_cond: Array_T) -> Array_T:
        return x_cond + self.sigma**2 * self.reg_C @ grad(self.target_log_prob)(x_cond)

    def cov_mat(self) -> Array_T:
        return 2 * self.sigma**2 * self.reg_C

    def sample(self, x_cond: Array_T, key: PRNGKeyArray) -> Array_T:
        """Sample from q(.|x_cond)"""
        noise = random.normal(key, x_cond.shape)
        return self.mean(x_cond) + jnp.real(sqrtm(self.cov_mat())) @ noise

    def log_prob(self, x: Array_T, x_cond: Array_T) -> Numeric:
        """Evaluate q(x|x_cond)"""
        # XXX: no normalizer - does not matter for now since accept prob relies
        # on a ratio of this log_prob
        mean = self.mean(x_cond)
        inv_cov_mat = jnp.linalg.inv(self.cov_mat())

        return -1 / 2 * (x - mean) @ inv_cov_mat @ (x - mean)


class PreconditionedMALAProposalDistDiagCov(struct.PyTreeNode):
    """
    From Atchade, 2006
    Conditional distribution q(xᵢ₊₁|xᵢ) = 𝒩(xᵢ + σ²ᵢΛᵢ∇log p(xᵢ), 2σ²ᵢΛᵢI))
    """

    target_log_prob: LogDensity_T
    sigma: Numeric
    C: Array

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


class MALAConfig(KernelConfig):
    step_size: float
    C: Optional[Array_T] = None


class MALAInfo(Info):
    accept: Numeric
    log_alpha: Numeric


class MALAState(State):
    pass


class MALAKernel(TunableKernel[MALAConfig, MALAState, MALAInfo]):
    supports_diagonal_mass: bool = True

    @classmethod
    def create(
        cls: Type[Self], target_log_prob: LogDensity_T, config: MALAConfig
    ) -> Self:
        return cls(target_log_prob, config)

    def get_step_size(self) -> Array_T:
        return self.config.step_size

    def get_inverse_mass_matrix(self) -> Array_T:
        return self.config.C

    def set_step_size(self, step_size) -> Self:
        return self.replace(config=self.config.replace(step_size=step_size))

    def _set_dense_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(config=self.config.replace(C=inverse_mass_matrix))

    def _set_diag_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(config=self.config.replace(C=inverse_mass_matrix))

    def init_state(self, x: Array_T) -> MALAState:
        return MALAState(x=x)

    def get_proposal(self) -> Union[PreconditionedMALAProposalDist, PreconditionedMALAProposalDistDiagCov, PreconditionedMALAProposalDistNoCov]:
        # return MALAProposalDist(
        #     target_log_prob=self.target_log_prob, step_size=self.config.step_size
        # )
        if self.config.C is None:
            return PreconditionedMALAProposalDistNoCov(
                target_log_prob=self.target_log_prob, sigma=self.config.step_size
            )
        elif len(self.config.C.shape) == 1:
            return PreconditionedMALAProposalDistDiagCov(
                target_log_prob=self.target_log_prob, sigma=self.config.step_size, C=self.config.C
            )
        elif len(self.config.C.shape) == 2:
            return PreconditionedMALAProposalDist(
                target_log_prob=self.target_log_prob, sigma=self.config.step_size,
                C=self.config.C
            )
        else:
            raise ValueError

    def _build_info(self, accept: Numeric, log_alpha: Numeric) -> MALAInfo:
        return MALAInfo(accept=accept, log_alpha=log_alpha)

    def _compute_accept_prob(self, proposal: MALAState, x: MALAState) -> Numeric:
        """Compute α = min(1, (p(xᵢ₊₁)q(xᵢ | xᵢ₊₁)) / (p(xᵢ) q(xᵢ₊₁ | xᵢ)))"""
        q = self.get_proposal()
        log_q_new_given_prev = q.log_prob(x=proposal.x, x_cond=x.x)
        log_q_prev_given_new = q.log_prob(x=x.x, x_cond=proposal.x)

        log_alpha = (
            self.target_log_prob(proposal.x)
            + log_q_prev_given_new
            - self.target_log_prob(x.x)
            - log_q_new_given_prev
        )
        log_alpha = jnp.nan_to_num(log_alpha, neginf=-1e80, posinf=-1e80, nan=-1e80)
        # log_alpha = jnp.clip(log_alpha, a_min=-500, a_max=0)
        # log_alpha = jnp.nan_to_num(log_alpha, posinf=-1e20, neginf=-1e20, nan=-1e20)
        return log_alpha

    def _sample_from_proposal(self, key: PRNGKeyArray, x: MALAState) -> MALAState:
        q = self.get_proposal()
        proposal = q.sample(x_cond=x.x, key=key)
        return x.replace(x=proposal)



class MALAKernelFactory(TunableMHKernelFactory[MALAConfig, MALAState, MALAInfo]):
    kernel_cls: Type[MALAKernel] = struct.field(pytree_node=False, default=MALAKernel)

    def build_kernel(self, log_prob: LogDensity_T) -> MALAKernel:
        return self.kernel_cls.create(log_prob, self.config)
