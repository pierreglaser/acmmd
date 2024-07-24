import abc
from typing_extensions import Self
from typing import ClassVar, Generator, Generic, Tuple, Type, TypeVar, Union, Any, cast

import jax
from jax._src.tree_util import tree_map
from jax.lax import fori_loop
import jax.numpy as jnp
from flax import struct
from jax import random
from jax.random import fold_in
from typing_extensions import TypeAlias

from jax_samplers.pytypes import LogDensity_T, Numeric, PRNGKeyArray

Array_T: TypeAlias = Any


# class ABCPyTreeNodeMeta(abc.ABCMeta, type(struct.PyTreeNode)):
#     # type(struct.PyTreeNode) reduces to the standard ``type```
#     # metaclass at runtime, but takes a different value for static
#     # type-checkers to signal them that struct.PyTreeNode subclass
#     # have dataclass-like semantics. This wrapper metaclass is thus
#     # just a convenience metaclass to be used instead of abc.ABEMeta
#     # to comply with static type-checkers.
#     pass


class Info(struct.PyTreeNode):
    pass


class State(struct.PyTreeNode):
    x: Array_T

State_T_co = TypeVar("State_T_co", bound=State, covariant=True)
State_T = TypeVar("State_T", bound=State)

Info_T = TypeVar("Info_T", bound=Info)
Info_T_co = TypeVar("Info_T_co", bound=Info, covariant=True)


class KernelConfig(struct.PyTreeNode):
    pass


Config_T = TypeVar("Config_T", bound=KernelConfig)
Config_T_co = TypeVar("Config_T_co", bound=KernelConfig, covariant=True)



class Result(Generic[State_T, Info_T], struct.PyTreeNode):
    state: State_T
    info: Info_T


class Kernel(
    Generic[Config_T, State_T, Info_T], struct.PyTreeNode, metaclass=abc.ABCMeta
):
    target_log_prob: LogDensity_T
    config: Config_T

    @classmethod
    @abc.abstractmethod
    def create(cls: Type[Self], target_log_prob: LogDensity_T, config: Config_T) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def init_state(self, x: Array_T) -> State_T:
        raise NotADirectoryError

    @abc.abstractmethod
    def one_step(self, x: State_T, key: PRNGKeyArray) -> Result[State_T, Info_T]:
        raise NotImplementedError

    def n_steps(self, x: State_T, n: int, key: PRNGKeyArray) -> Result[State_T, Info_T]:
        result = self.one_step(x, fold_in(key, 0))

        def body_fun(i, result: Result[State_T, Info_T]) -> Result[State_T, Info_T]:
            new_result = self.one_step(result.state, fold_in(key, i))
            # mean_info =  tree_map(lambda x, y: (i/i+1) * x + (1/i+1) * y, result.info, new_result.info)
            # return new_result.replace(info=mean_info)
            return new_result

        return cast(Result[State_T, Info_T], fori_loop(1, n, body_fun, result))


class MHKernel(
    Kernel[Config_T, State_T, Info_T], Generic[Config_T, State_T, Info_T], metaclass=abc.ABCMeta
):
    @classmethod
    @abc.abstractmethod
    def create(cls: Type[Self], target_log_prob: LogDensity_T, config: Config_T) -> Self:
        return super().create(target_log_prob, config)

    @abc.abstractmethod
    def _sample_from_proposal(self, key: PRNGKeyArray, x: State_T) -> State_T:
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_accept_prob(self, proposal: State_T, x: State_T) -> Numeric:
        raise NotImplementedError

    @abc.abstractmethod
    def _build_info(self, accept: Numeric, log_alpha: Numeric) -> Info_T:
        raise NotImplementedError

    def maybe_accept_proposal(
        self,
        proposal: State_T,
        x: State_T,
        key: PRNGKeyArray,
    ) -> Result[State_T, Info_T]:
        log_alpha = self._compute_accept_prob(proposal=proposal, x=x)
        accept = jnp.log(random.uniform(key=key)) < log_alpha
        accept = cast(int, accept)

        new_state = cast(State_T, jax.lax.cond(accept, lambda: proposal, lambda: x))
        info = self._build_info(accept=accept, log_alpha=log_alpha)
        return Result(state=new_state, info=info)


    def one_step(
        self, x: State_T, key: PRNGKeyArray
    ) -> Result[State_T, Info_T]:
        key_proposal, key_accept, key = random.split(key, num=3)
        proposal = self._sample_from_proposal(key_proposal, x)
        return self.maybe_accept_proposal(proposal, x, key=key_accept)


class TunableKernel(MHKernel[Config_T, State_T, Info_T], Generic[Config_T, State_T, Info_T]):
    supports_diagonal_mass: bool = struct.field(pytree_node=False, default=False)

    @abc.abstractmethod
    def get_step_size(self) -> Numeric:
        raise NotImplementedError

    @abc.abstractmethod
    def get_inverse_mass_matrix(self) -> Array_T:
        raise NotImplementedError

    @abc.abstractmethod
    def set_step_size(self, step_size) -> Self:
        raise NotImplementedError

    def set_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        assert self.get_inverse_mass_matrix().shape == inverse_mass_matrix.shape
        is_mass_matrix_diagonal = len(inverse_mass_matrix.shape) == 1
        if is_mass_matrix_diagonal:
            if self.supports_diagonal_mass:
                return self._set_diag_inverse_mass_matrix(inverse_mass_matrix)
            else:
                raise NotImplementedError
        else:
            return self._set_dense_inverse_mass_matrix(inverse_mass_matrix)

    def _set_dense_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        raise NotImplementedError

    def _set_diag_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        raise NotImplementedError


class KernelFactory(Generic[Config_T, State_T, Info_T_co], struct.PyTreeNode):
    config: Config_T
    kernel_cls: Type[Kernel[Config_T, State_T, Info_T_co]] = struct.field(pytree_node=False)

    def build_kernel(self, log_prob: LogDensity_T) -> Kernel[Config_T, State_T, Info_T_co]:
        return self.kernel_cls(log_prob, self.config)


class MHKernelFactory(KernelFactory[Config_T, State_T, Info_T_co], Generic[Config_T, State_T, Info_T_co], struct.PyTreeNode):
    config: Config_T
    kernel_cls: Type[MHKernel[Config_T, State_T, Info_T_co]] = struct.field(pytree_node=False)

    def build_kernel(self, log_prob: LogDensity_T) -> MHKernel[Config_T, State_T, Info_T_co]:
        return self.kernel_cls.create(log_prob, self.config)


class TunableMHKernelFactory(KernelFactory[Config_T, State_T, Info_T_co], Generic[Config_T, State_T, Info_T_co], struct.PyTreeNode):
    config: Config_T
    kernel_cls: Type[TunableKernel[Config_T, State_T, Info_T_co]] = struct.field(pytree_node=False)

    def build_kernel(self, log_prob: LogDensity_T) -> TunableKernel[Config_T, State_T, Info_T_co]:
        return self.kernel_cls.create(log_prob, self.config)
