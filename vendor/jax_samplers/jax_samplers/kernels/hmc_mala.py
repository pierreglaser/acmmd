from typing import Type

import blackjax
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.hmc import HMCInfo as BHMCInfo
from blackjax.mcmc.hmc import HMCState as BHMCState
from blackjax.mcmc.hmc import init as hmc_init
from blackjax.mcmc.hmc import kernel as hmc_kernel
from flax import struct
from typing_extensions import Self
import jax.numpy as jnp

from jax_samplers.pytypes import Array, LogDensity_T, Numeric, PRNGKeyArray
from jax_samplers.kernels.base import (Array_T, Info, Kernel, KernelConfig, MHKernelFactory,
                                           Result, State)


class HMCConfig(KernelConfig):
    step_size: float


class HMCInfo(Info):
    accept_freq: Numeric


class HMCState(State):
    _blackjax_state: BHMCState


class HMCKernel(Kernel[HMCConfig, HMCState, HMCInfo]):
    _kernel_fun = hmc_kernel()

    @classmethod
    def create(
        cls: Type[Self], target_log_prob: LogDensity_T, config: HMCConfig
    ) -> Self:
        return cls(target_log_prob=target_log_prob, config=config)

    def init(self, x0: Array) -> HMCState:
        _blackjax_state = hmc_init(x0, self.target_log_prob)
        assert isinstance(_blackjax_state.position, Array_T)
        return HMCState(_blackjax_state.position, _blackjax_state)

    def one_step(
        self,
        state: HMCState,
        key: PRNGKeyArray,
    ) -> Result[HMCState, HMCInfo]:
        _new_state, _new_info = self._kernel_fun(
            key,
            state._blackjax_state,
            self.target_log_prob,
            self.config.step_size,
            inverse_mass_matrix=jnp.ones_like(state._blackjax_state.position),
            num_integration_steps=1
        )
        assert isinstance(_new_state.position, Array_T)
        ret = Result(
            HMCState(_new_state.position, _new_state),
            HMCInfo(accept_freq=_new_info.acceptance_probability),
        )
        return ret


class HMCKernelFactory(MHKernelFactory[HMCConfig, HMCState, HMCInfo]):
    kernel_cls: Type[HMCKernel] = HMCKernel

    def build_kernel(self, log_prob: LogDensity_T) -> HMCKernel:
        return self.kernel_cls.create(log_prob, self.config)
