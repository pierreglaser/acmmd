from typing import Callable, Generic, Type, TypeVar

import jax
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as np_distributions
from flax import struct
from jax import vmap
from jax.nn import logsumexp
from jax.tree_util import tree_map
from typing_extensions import Self

from jax_samplers.pytypes import Array, Numeric, PRNGKeyArray
from jax_samplers.kernels.base import Array_T


class ParticleApproximation(struct.PyTreeNode):
    particles: Array_T
    log_ws: Array

    @classmethod
    def create(cls: Type[Self], xs: Array_T, log_ws: Array) -> Self:
        return cls(xs, log_ws)

    @property
    def xs(self) -> Array:
        return self.particles

    @property
    def normalized_ws(self) -> Array:
        # normalized log-weights
        return jax.nn.softmax(self.log_ws)

    @property
    def ws(self) -> Array:
        # Unnormalized weights. Unstable exponentiation, do not use unless necessary.
        return jnp.exp(self.log_ws)

    @property
    def num_samples(self) -> int:
        return self.xs.shape[0]

    @classmethod
    def from_npdistribution(
        cls: Type[Self],
        dist: np_distributions.Distribution,
        num_samples: int,
        key: PRNGKeyArray,
    ) -> Self:
        xs = dist.sample(key, (num_samples,))  # type: ignore
        log_ws = jax.nn.log_softmax(jnp.zeros(xs.shape[0]))
        return cls.create(xs=xs, log_ws=log_ws)

    def resample_and_reset_weights(self: Self, key: PRNGKeyArray) -> Self:
        # resampling implies resetting importance weights to 1
        # XXX: gumbel trick is memory intensive
        # indices = random.categorical(key, self.log_ws, shape=(self.num_samples,))
        mn = np_distributions.Categorical(probs=self.normalized_ws)
        indices = mn.sample(key, (self.num_samples,))  # type: ignore

        log_ws = -np.log(self.num_samples) * jnp.ones_like(self.log_ws)
        new_particles = jnp.take(self.particles, indices, axis=0)
        return self.replace(log_ws=log_ws, particles=new_particles)

    def log_effective_sample_size(self) -> Numeric:
        first_term: Numeric = 2.0 * logsumexp(self.log_ws)
        second_term: Numeric = logsumexp(2.0 * self.log_ws)
        return first_term - second_term

    def normalized_log_ess(self) -> Numeric:
        return self.log_effective_sample_size() - jnp.log(self.num_samples)

    def ensure_normalized_weights(self) -> Self:
        log_ws_normalized = jax.nn.log_softmax(self.log_ws)
        return self.replace(log_ws=log_ws_normalized)

    def average_of(self, func: Callable[[Array], Numeric]) -> Numeric:
        return jnp.average(vmap(func)(self.xs), weights=self.normalized_ws)


class BaseConfig(struct.PyTreeNode):
    pass


class BaseStats(struct.PyTreeNode):
    pass
