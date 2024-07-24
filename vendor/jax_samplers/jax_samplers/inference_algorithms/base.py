import abc
from typing import Generic, Optional, Tuple, TypeVar

from flax import struct
from numpyro import distributions as np_distributions
from typing_extensions import Self

from jax_samplers.distributions import LogDensity_T
from jax_samplers.pytypes import PRNGKeyArray
from jax_samplers.kernels.base import Array_T

from ..particle_aproximation import ParticleApproximation


class InferenceAlgorithmConfig(struct.PyTreeNode):
    num_samples: int = struct.field(pytree_node=False)


IAC_T = TypeVar("IAC_T", bound=InferenceAlgorithmConfig)


class InferenceAlgorithmInfo(struct.PyTreeNode):
    pass


PA_T = TypeVar("PA_T", bound=ParticleApproximation)


PA_T_co = TypeVar("PA_T_co", bound=ParticleApproximation, covariant=True)


class InferenceAlgorithmResults(struct.PyTreeNode):
    samples: ParticleApproximation
    info: InferenceAlgorithmInfo


LD_T = TypeVar("LD_T", bound=LogDensity_T)


class InferenceAlgorithm(
    Generic[IAC_T], struct.PyTreeNode, metaclass=abc.ABCMeta
):
    config: IAC_T
    log_prob: LogDensity_T
    _init_state: Optional[ParticleApproximation] = None  # private API

    @abc.abstractmethod
    def init(self, key: PRNGKeyArray, dist: np_distributions.Distribution) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, key: PRNGKeyArray) -> Tuple[Self, InferenceAlgorithmResults]:
        raise NotImplementedError

    @abc.abstractmethod
    def set_log_prob(self, log_prob: LogDensity_T) -> Self:
        raise NotImplementedError

    def run_and_update_init(
        self, key: PRNGKeyArray
    ) -> Tuple[Self, InferenceAlgorithmResults]:
        self, results = self.run(key)
        self = self.replace(_init_state=results.samples)
        return self, results


class InferenceAlgorithmFactory(Generic[IAC_T], struct.PyTreeNode, metaclass=abc.ABCMeta):
    config: IAC_T

    @abc.abstractmethod
    def build_algorithm(self, log_prob: LogDensity_T) -> InferenceAlgorithm[IAC_T]:
        raise NotImplementedError
