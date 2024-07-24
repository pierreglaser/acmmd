from typing import Type

from blackjax.mcmc.hmc import HMCState as BNUTSState
from blackjax.mcmc.nuts import init as nuts_init
from blackjax.mcmc.nuts import kernel as nuts_kernel
from typing_extensions import Self

from flax import struct

from jax_samplers.pytypes import Array, LogDensity_T, Numeric, PRNGKeyArray
from jax_samplers.kernels.base import (Array_T, Info, Kernel, KernelConfig, MHKernelFactory,
                                           Result, State, TunableKernel, TunableMHKernelFactory)


import jax.numpy as jnp


class NUTSConfig(KernelConfig):
    # staying conservative w.r.t
    step_size: float
    inverse_mass_matrix: Array


class NUTSInfo(Info):
    accept: Numeric
    log_alpha: Numeric


class NUTSState(State):
    _blackjax_state: BNUTSState


class NUTSKernel(TunableKernel[NUTSConfig, NUTSState, NUTSInfo]):
    _kernel_fun = staticmethod(nuts_kernel())
    supports_diagonal_mass: bool  = True

    @classmethod
    def create(
        cls: Type[Self], target_log_prob: LogDensity_T, config: NUTSConfig
    ) -> Self:
        return cls(target_log_prob=target_log_prob, config=config)

    def _sample_from_proposal(self, key: PRNGKeyArray, x: NUTSState) -> NUTSState:
        raise NotImplementedError

    def _build_info(self, accept: Numeric, log_alpha: Numeric) -> NUTSInfo:
        return NUTSInfo(accept=accept, log_alpha=log_alpha)

    def _compute_accept_prob(self, proposal: NUTSState, x: NUTSState) -> Numeric:
        raise NotImplementedError

    def set_step_size(self, step_size) -> Self:
        return self.replace(config=self.config.replace(step_size=step_size))

    def _set_dense_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(config=self.config.replace(inverse_mass_matrix=inverse_mass_matrix))

    def _set_diag_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(config=self.config.replace(inverse_mass_matrix=inverse_mass_matrix))

    def get_inverse_mass_matrix(self) -> Array_T:
        return self.config.inverse_mass_matrix

    def get_step_size(self) -> Numeric:
        return self.config.step_size

    def init_state(self, x0: Array) -> NUTSState:
        _blackjax_state = nuts_init(x0, self.target_log_prob)
        # assert isinstance(_blackjax_state.position, Array_T)
        return NUTSState(_blackjax_state.position, _blackjax_state)

    def one_step(
        self,
        state: NUTSState,
        key: PRNGKeyArray,
    ) -> Result[NUTSState, NUTSInfo]:
        _new_state, _new_info = self._kernel_fun(
            rng_key=key,
            state=state._blackjax_state,
            logprob_fn=self.target_log_prob,
            step_size=self.config.step_size,
            inverse_mass_matrix=self.config.inverse_mass_matrix,
        )

        # assert isinstance(_new_state.position, Array_T)
        ret = Result(
            NUTSState(_new_state.position, _new_state),
            # accept does not make sense for nuts
            self._build_info(
                jnp.log(_new_info.acceptance_probability),
                jnp.log(_new_info.acceptance_probability)
            )
        )
        return ret


class NUTSKernelFactory(TunableMHKernelFactory[NUTSConfig, NUTSState, NUTSInfo]):
    kernel_cls: Type[NUTSKernel] = struct.field(pytree_node=False, default=NUTSKernel)

    def build_kernel(self, log_prob: LogDensity_T) -> NUTSKernel:
        return self.kernel_cls.create(log_prob, self.config)
