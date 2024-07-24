from abc import abstractmethod
from typing import cast

import jax
import jax.numpy as jnp
from flax import struct
from kwgflows.pytypes import Array, PRNGKeyArray, Scalar
from kwgflows.rkhs.kernels import base_kernel, gaussian_kernel
from numpyro import distributions as np_distributions
from typing_extensions import Self

from calibration.conditional_models.base import (
    DistributionModel,
    LogDensityModel,
    SampleableModel,
)


class AutoRegressiveModelBase(SampleableModel):
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def _sample_xi_given_xim1(self, key: PRNGKeyArray, xim1: Array) -> Array:
        raise NotImplementedError

    def sample_from_conditional(self, key: PRNGKeyArray, num_samples: int) -> Array:
        x = jnp.zeros((num_samples, self.dimension))

        def _get_one_sample(x, key) -> Array:
            for i in range(self.dimension):
                key, subkey = jax.random.split(key)
                x = self._sample_xi_given_xim1(subkey, x)
            return x

        subkeys = jax.random.split(key, num_samples)
        xs = jax.vmap(_get_one_sample, in_axes=(0, 0))(x, subkeys)
        return xs
