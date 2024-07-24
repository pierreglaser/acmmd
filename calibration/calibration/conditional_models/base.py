from __future__ import annotations

import abc
from typing import Any, Callable, Generic, Optional, Type, TypeVar

import jax
import jax.numpy as jnp
import numpyro.distributions as np_distributions
from flax import struct
from jax import random
from jax_samplers.inference_algorithms.mcmc.base import MCMCAlgorithmFactory, MCMCConfig
from jax_samplers.kernels.mala import MALAConfig, MALAKernelFactory
from kwgflows.base import DiscreteProbability
from kwgflows.pytypes import Array, Array_or_PyTreeNode, PRNGKeyArray, Scalar
from typing_extensions import Self


class BaseConditionalModel(struct.PyTreeNode):
    pass


def _get_default_mcmc_factory() -> MCMCAlgorithmFactory[Any, Any, Any]:
    mcmc_config = MCMCConfig(
        num_samples=1000,
        kernel_factory=MALAKernelFactory(config=MALAConfig(step_size=0.01)),
        num_warmup_steps=100,
        adapt_step_size=True,
        target_accept_rate=0.5,
        init_using_log_l_mode=False,
    )
    return MCMCAlgorithmFactory(config=mcmc_config)


class LogDensityModel(BaseConditionalModel, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, y: Array) -> Scalar:
        raise NotImplementedError

    def score(self, y: Array) -> Array:
        return jax.grad(self.__call__)(y)

    def mcmc_discretize(
        self,
        base_dist: np_distributions.Distribution,
        key: PRNGKeyArray,
        mcmc_factory: Optional[MCMCAlgorithmFactory[Any, Any, Any]] = None,
        num_samples: int = 1000,
    ) -> DiscretizedModel[Self]:
        if mcmc_factory is None:
            mcmc_factory = _get_default_mcmc_factory()

        mcmc_factory = mcmc_factory.replace(
            config=mcmc_factory.config.replace(num_samples=num_samples)
        )
        alg = mcmc_factory.build_algorithm(self)

        key, subkey = random.split(key)
        alg = alg.init(subkey, base_dist)

        key, subkey = random.split(key)
        _, results = alg.run(subkey)
        assert len(results.samples.particles) == num_samples
        return DiscretizedModel(
            results.samples.particles,
            jnp.ones(num_samples) / num_samples,
            model=self,
        )


class ConcreteLogDensityModel(LogDensityModel):
    _call_fn: Callable[[Array], Scalar]
    dimension: int = struct.field(pytree_node=False)

    def __call__(self, y: Array) -> Scalar:
        return self._call_fn(y)


class SampleableModel(BaseConditionalModel, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample_from_conditional(self, key: PRNGKeyArray, num_samples: int) -> Array:
        raise NotImplementedError

    def non_mcmc_discretize(
        self, key: PRNGKeyArray, num_samples: int
    ) -> DiscretizedModel[Self]:
        particles = self.sample_from_conditional(key, num_samples)
        return DiscretizedModel(
            particles,
            jnp.ones(num_samples) / num_samples,
            model=self,
        )


class DistributionModel(SampleableModel, LogDensityModel, metaclass=abc.ABCMeta):
    """
    A wrapper around numpyro `Distribution` that is compatible with jax transformations.

    ...

    See: https://github.com/pyro-ppl/numpyro/issues/1317
    """

    data: Array_or_PyTreeNode
    aux: Any = struct.field(pytree_node=False)
    dist_cls: Type[np_distributions.Distribution] = struct.field(pytree_node=False)

    @classmethod
    def create(cls: Type[Self], d: np_distributions.Distribution):
        data, aux = d.tree_flatten()
        return cls(data, aux, type(d))

    @property
    def distribution(self) -> np_distributions.Distribution:
        return self.dist_cls.tree_unflatten(self.aux, self.data)

    def __call__(self, y: Array) -> Scalar:
        return self.distribution.log_prob(y)

    def sample_from_conditional(self, key: PRNGKeyArray, num_samples: int) -> Array:
        return self.distribution.sample(key, (num_samples,))

    def as_logdensity_model(self) -> ConcreteLogDensityModel:
        return ConcreteLogDensityModel(self, self.distribution.event_shape[0])


M = TypeVar("M", bound=BaseConditionalModel, covariant=True)


class DiscretizedModel(Generic[M], DiscreteProbability[Array], BaseConditionalModel):
    model: M
