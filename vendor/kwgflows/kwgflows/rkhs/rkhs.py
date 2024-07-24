from typing import Generic, TypeVar, cast

import jax
import jax.numpy as jnp
from flax import struct
from jax import grad, vmap
from jax.tree_util import tree_map
from typing_extensions import Self

from kwgflows.pytypes import Scalar, T
from kwgflows.rkhs.kernels import Array, base_kernel


class rkhs_element(Generic[T], struct.PyTreeNode):
    X: T
    w: Array
    kernel: base_kernel[T]

    def __add__(self, other: Self):
        assert self.kernel is other.kernel
        return type(self)(
            X=tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), self.X, other.X),
            w=jnp.concatenate([self.w, other.w], axis=0),
            kernel=self.kernel,
        )

    def __sub__(self, other: Self) -> Self:
        assert self.kernel is other.kernel
        return type(self)(
            X=tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), self.X, other.X),
            w=jnp.concatenate([self.w, -other.w], axis=0),
            kernel=self.kernel,
        )

    def __mul__(self, other: Scalar) -> Self:
        # other should be scalar
        assert isinstance(other, (float, int)) or other.ndim == 0
        return self.replace(w=self.w * other)

    def __neg__(self) -> Self:
        return self.replace(w=-self.w)

    def inner_product(self, other: Self) -> Scalar:
        K = self.kernel.make_gram_matrix(self.X, other.X)
        return jnp.dot(self.w, jnp.dot(K, other.w))

    def __call__(self, x: T) -> Scalar:
        self.kernel
        return jnp.dot(
            self.w,
            vmap(type(self.kernel).__call__, (None, None, 0))(self.kernel, x, self.X),
        )

    def rkhs_norm(self, squared: bool = False) -> Scalar:
        norm_squared = self.inner_product(self)
        return jax.lax.cond(
            squared, lambda _: norm_squared, lambda _: jnp.sqrt(norm_squared), None
        )

    def grad(self, x: T) -> T:
        # Small helper function to be able to access velocity fields at inference time by
        # reconstructing them from their respective witness functions
        # Storing velocity fields directly would not have been easy because grad(self.__call__)
        # would yield a non-pytree object. In theory, even more structure could be added
        # since this velocity field is actually RKHS element in H^d.
        z = grad(self.__call__)(x)
        return cast(T, z)
