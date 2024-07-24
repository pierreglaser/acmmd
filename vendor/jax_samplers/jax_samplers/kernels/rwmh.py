from typing import Optional, Tuple, Type, Union, cast
from blackjax.base import Info

import jax.numpy as jnp
from flax import struct
import jax
from jax import grad, random, vmap
# from jax.scipy.linalg import sqrtm
from jax.numpy.linalg import eigh
from jax.tree_util import tree_map
from jax.lax import scan  # type: ignore
from typing_extensions import Self, TypeAlias

from jax_samplers.kernels.base import Array_T, KernelConfig, MHKernel, MHKernelFactory, State, Info, TunableKernel, TunableMHKernelFactory

MHSample: TypeAlias = Array_T

from jax_samplers.pytypes import Array, LogDensity_T, Numeric, PRNGKeyArray

class RWConfig(KernelConfig):
    step_size: float
    C: Optional[Array_T] = None


class RWInfo(Info):
    accept: Numeric
    log_alpha: Numeric


class RWState(State):
    pass


def sqrtm(m: Array) -> Array:
    eigvals, eigvecs = eigh(m)
    return eigvecs @ jnp.diag(jnp.real(eigvals) ** 0.5) @ eigvecs.T


class RWProposalDist(struct.PyTreeNode):
    step_size: float
    C: Optional[Array_T] = None

    def mean(self, x_cond: MHSample) -> MHSample:
        return x_cond

    def std(self) -> MHSample:
        return jnp.sqrt(self.step_size)

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


class PreconditonnedRWProposalDistDiagCov(struct.PyTreeNode):
    step_size: float
    C: Array_T

    def mean(self, x_cond: MHSample) -> MHSample:
        return x_cond

    def std(self) -> MHSample:
        return jnp.sqrt(self.step_size) * jnp.sqrt(self.C)

    def sample(self, x_cond: MHSample, key: PRNGKeyArray) -> MHSample:
        """Sample from q(.|x_cond)"""
        noise = random.normal(key, x_cond.shape)
        return self.mean(x_cond) + self.std() * noise

    def log_prob(self, x: MHSample, x_cond: MHSample) -> Numeric:
        """Evaluate q(x|x_cond)"""
        # XXX: no normalizer - does not matter for now since accept prob relies
        # on a ratio of this log_prob
        mean = self.mean(x_cond)
        return -jnp.dot((x - mean) / self.std() ** 2, x - mean) / 2

class PreConditionedRWProposalDist(struct.PyTreeNode):
    step_size: float
    C: Array_T

    def mean(self, x_cond: MHSample) -> MHSample:
        return x_cond

    def std(self) -> MHSample:
        return jnp.real(sqrtm(self.cov_mat()))

    def cov_mat(self) -> Array_T:
        return self.step_size * (self.C + 1e-4 * jnp.eye(self.C.shape[0]))

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

class RWKernel(TunableKernel[RWConfig, RWState, RWInfo]):
    supports_diagonal_mass: bool = True

    @classmethod
    def create(cls: Type[Self], target_log_prob: LogDensity_T,  config: RWConfig) -> Self:
        return cls(target_log_prob, config)

    def init_state(self, x: Array_T) -> RWState:
        return RWState(x=x)

    def _build_info(self, accept: Numeric, log_alpha: Numeric) -> RWInfo:
        return RWInfo(accept=accept, log_alpha=log_alpha)

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

    def get_proposal(self) -> Union[RWProposalDist, PreConditionedRWProposalDist, PreconditonnedRWProposalDistDiagCov]:
        if self.config.C is None:
            return RWProposalDist(step_size=self.config.step_size)
        elif len(self.config.C.shape) == 1:
            return PreconditonnedRWProposalDistDiagCov(step_size=self.config.step_size, C=self.config.C)
        else:
            return PreConditionedRWProposalDist(step_size=self.config.step_size, C=self.config.C)

    def _compute_accept_prob(
        self, proposal: RWState, x: RWState
    ) -> Numeric:
        """Compute α = min(1, (p(xᵢ₊₁)q(xᵢ | xᵢ₊₁)) / (p(xᵢ) q(xᵢ₊₁ | xᵢ)))"""
        log_alpha = self.target_log_prob(proposal.x) - self.target_log_prob(x.x)
        return jnp.nan_to_num(log_alpha, neginf=-1e80, posinf=-1e80, nan=-1e80)

    def _sample_from_proposal(self, key: PRNGKeyArray, x: RWState) -> RWState:
        q = self.get_proposal()
        proposal = q.sample(x_cond=x.x, key=key)
        return x.replace(x=proposal)


class RWKernelFactory(TunableMHKernelFactory[RWConfig, RWState, RWInfo]):
    kernel_cls: Type[RWKernel] = struct.field(pytree_node=False, default=RWKernel)

    def build_kernel(self, log_prob: LogDensity_T) -> RWKernel:
        return self.kernel_cls.create(log_prob, self.config)
