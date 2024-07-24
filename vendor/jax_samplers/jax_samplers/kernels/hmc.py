from typing import Type

import blackjax
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.hmc import HMCInfo as BHMCInfo
from blackjax.mcmc.hmc import HMCState as BHMCState
from blackjax.mcmc.hmc import init as hmc_init
from blackjax.mcmc.hmc import kernel as hmc_kernel
from flax import struct
from typing_extensions import Self

from jax_samplers.pytypes import Array, LogDensity_T, Numeric, PRNGKeyArray
from jax_samplers.kernels.base import (Array_T, Info, Kernel, KernelConfig, MHKernel, MHKernelFactory,
                                           Result, State, TunableKernel, TunableMHKernelFactory)

import jax.numpy as jnp


class HMCConfig(KernelConfig):
    step_size: float
    inverse_mass_matrix: Array
    num_integration_steps: int = struct.field(pytree_node=False)


class HMCInfo(Info):
    accept: Numeric
    log_alpha: Numeric


class HMCState(State):
    _blackjax_state: BHMCState


class HMCKernel(TunableKernel[HMCConfig, HMCState, HMCInfo]):
    _kernel_fun = staticmethod(hmc_kernel())
    supports_diagonal_mass: bool = True

    @classmethod
    def create(
        cls: Type[Self], target_log_prob: LogDensity_T, config: HMCConfig
    ) -> Self:
        return cls(target_log_prob=target_log_prob, config=config)

    def _sample_from_proposal(self, key: PRNGKeyArray, x: HMCState) -> HMCState:
        raise NotImplementedError

    def _build_info(self, accept: Numeric, log_alpha: Numeric) -> HMCInfo:
        return HMCInfo(accept=accept, log_alpha=log_alpha)

    def _compute_accept_prob(self, proposal: HMCState, x: HMCState) -> Numeric:
        raise NotImplementedError

    def _set_dense_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(config=self.config.replace(inverse_mass_matrix=inverse_mass_matrix))

    def _set_diag_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(config=self.config.replace(inverse_mass_matrix=inverse_mass_matrix))

    def get_inverse_mass_matrix(self) -> Array_T:
        return self.config.inverse_mass_matrix

    def get_step_size(self) -> Numeric:
        return self.config.step_size

    def set_step_size(self, step_size: Array_T) -> Self:
        return self.replace(config=self.config.replace(step_size=step_size))

    def init_state(self, x0: Array) -> HMCState:
        _blackjax_state = hmc_init(x0, self.target_log_prob)
        # assert isinstance(_blackjax_state.position, Array_T)
        return HMCState(_blackjax_state.position, _blackjax_state)

    def one_step(
        self,
        state: HMCState,
        key: PRNGKeyArray,
    ) -> Result[HMCState, HMCInfo]:
        _new_state, _new_info = self._kernel_fun(
            rng_key=key,
            state=state._blackjax_state,
            logprob_fn=self.target_log_prob,
            inverse_mass_matrix=self.config.inverse_mass_matrix,
            step_size=self.config.step_size,
            num_integration_steps=self.config.num_integration_steps,
        )
        # assert isinstance(_new_state.position, Array_T)
        ret = Result(
            HMCState(_new_state.position, _new_state),
            self._build_info(_new_info.is_accepted, jnp.log(_new_info.acceptance_probability)),
        )
        return ret


class HMCKernelFactory(TunableMHKernelFactory[HMCConfig, HMCState, HMCInfo]):
    kernel_cls: Type[HMCKernel] = struct.field(pytree_node=False, default=HMCKernel)

    def build_kernel(self, log_prob: LogDensity_T) -> HMCKernel:
        return self.kernel_cls.create(log_prob, self.config)
