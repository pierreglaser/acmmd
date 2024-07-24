import abc
from typing import Callable, Generic, Type, TypeVar, cast

import jax.numpy as jnp
import numpyro.distributions as np_distributions
from flax import struct
from jax import vmap
from kwgflows.pytypes import Array, Numeric, PRNGKeyArray, Scalar, T
from kwgflows.rkhs.kernels import base_kernel
from kwgflows.rkhs.rkhs import rkhs_element
from kwgflows.utils import infer_num_samples_pytree
from typing_extensions import Self


class BaseProbability(Generic[T], struct.PyTreeNode, metaclass=abc.ABCMeta):
    pass


T_ret = TypeVar("T_ret", struct.PyTreeNode, Array)


class ProbabilityWithDensity(BaseProbability[Array]):
    log_prob_fn: Callable[[Array], Scalar]

    def __call__(self, x: Array) -> Numeric:
        return self.log_prob_fn(x)


class DiscreteProbability(BaseProbability[T], struct.PyTreeNode):
    X: T
    w: Array

    @property
    def num_atoms(self):
        return self.w.shape[0]

    @property
    def particles(self):
        return self.X

    @classmethod
    def create(cls: Type[Self], X: T, w: Array) -> "DiscreteProbability[T]":
        return cls(X=X, w=w)

    @classmethod
    def from_samples(cls: Type[Self], X: T) -> "DiscreteProbability[T]":
        num_samples = infer_num_samples_pytree(X)
        return cls(X, jnp.ones(num_samples) / num_samples)

    @classmethod
    def from_dist(
        cls, dist: np_distributions.Distribution, num_samples: int, key: PRNGKeyArray
    ) -> "DiscreteProbability[Array]":
        X = cast(Array, dist.sample(key, (num_samples,)))
        return cls.from_samples(X)

    def set_particles(self, X: Array) -> Self:
        return self.replace(X=X)

    def set_weights(self, w: Array) -> Self:
        return self.replace(w=w)

    def average_of(self, f: Callable[[T], Numeric]) -> Array:
        return jnp.average(vmap(f)(self.X), weights=self.w)

    def push_forward(self, f: Callable[[T], T_ret]) -> "DiscreteProbability[T_ret]":
        new_Xs = vmap(f)(self.X)
        return DiscreteProbability.create(new_Xs, self.w)

    def get_mean_embedding(self, kernel: base_kernel[T]) -> rkhs_element[T]:
        return rkhs_element(self.X, self.w, kernel)

    def analytical_expectation_of(self, f: Callable[[T, T], Numeric], y: T) -> Numeric:
        return self.average_of(lambda x: f(x, y))

    def bivariate_analytical_expectation_of(
        self, f: Callable[[T, T], Numeric], other: Self
    ) -> Numeric:
        f_evals = vmap(vmap(f, in_axes=(0, None)), in_axes=(None, 0))(self.X, other.X)
        return jnp.average(
            jnp.average(f_evals, weights=self.w, axis=1), weights=other.w, axis=0
        )
