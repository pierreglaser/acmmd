from typing import Generic, Optional, Tuple, Type

import jax.numpy as jnp
import jax
from flax import struct
from jax import random
from numpyro import distributions as np_distributions
from typing_extensions import Self

from kwgflows.base import DiscreteProbability
from kwgflows.divergences.base import KernelizedDivergence
from kwgflows.divergences.mmd import mmd, mmd_fixed_Q
from kwgflows.gradient_flow import (GradientFlowConfig, GradientFlowResult,
                                    gradient_flow)
from kwgflows.rkhs.kernels import base_kernel, energy_kernel
from kwgflows.pytypes import Array, T


class KWGFGenerativeModelConfig(Generic[T], struct.PyTreeNode):
    kernel: base_kernel[T]
    gradient_flow_config: GradientFlowConfig
    target: np_distributions.Distribution = struct.field(pytree_node=False)
    num_particles: int = struct.field(pytree_node=False)


class KWGFGenerativeModel(struct.PyTreeNode):
    config: KWGFGenerativeModelConfig[Array]
    kwgf_results: Optional[GradientFlowResult] = None

    @classmethod
    def create(
        cls: Type[Self],
        kernel: Optional[energy_kernel] = None,
        gradient_flow_config: Optional[GradientFlowConfig] = None,
        target: Optional[np_distributions.Distribution] = None,
        num_particles: int = 1000,
        X: Optional[Array] = None,
    ) -> Self: 
        if kernel is None:
            kernel = energy_kernel(1.0, 1.0)
        if gradient_flow_config is None:
            gradient_flow_config = GradientFlowConfig()
        if target is None:
            assert X is not None
            target = np_distributions.Independent(
                np_distributions.Normal(
                    jnp.zeros(X.shape[1]), jnp.ones(X.shape[1])  # type: ignore
                ),
                1,
            )
        else:
            assert X is None or X.shape[1] == target.event_shape[0]

        config = KWGFGenerativeModelConfig(
            kernel=kernel,
            gradient_flow_config=gradient_flow_config,
            target=target,
            num_particles=num_particles,
        )
        return cls(config=config)

    @property
    def _is_fitted(self):
        return self.kwgf_results is not None

    def fit(self, X: Array, key: random.PRNGKeyArray) -> Self:
        assert X.ndim == 2
        assert X.shape[1] == self.config.target.event_shape[0]

        target_samples = self.config.target.sample(key, (self.config.num_particles,))
        discretized_target = DiscreteProbability.from_samples(target_samples)
        divergence = mmd_fixed_Q(self.config.kernel, discretized_target)

        discretized_source = DiscreteProbability.from_samples(X)

        kwgf_results = gradient_flow(
            divergence, discretized_source, self.config.gradient_flow_config
        )
        return self.replace(kwgf_results=kwgf_results)


    def sample(self, key: random.PRNGKeyArray) -> Tuple[Array, Array]:
        assert self.kwgf_results is not None
        x0 = self.config.target.sample(key)

        def one_step(xt: Array, i: Array):
            assert self.kwgf_results is not None
            v = self.kwgf_results.get_velocity_field(-i)(xt)
            xt_plus_one = xt + self.config.gradient_flow_config.step_size * v
            return xt_plus_one, xt_plus_one

        xT, trajectory = jax.lax.scan(one_step, x0, jnp.arange(self.config.gradient_flow_config.num_steps))
        return xT, trajectory
