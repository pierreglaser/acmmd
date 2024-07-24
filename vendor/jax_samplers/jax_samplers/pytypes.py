from flax import struct
from typing import Any, Callable, Protocol, Union
from typing_extensions import TypeAlias

import jax.numpy as jnp
from jax._src import prng

Scalar = Union[float, int]

# jax source codes aliases Array to Any - the reason is that
# jax functions seeks to be compatible with other array classes such as
# np.array, jax tracers (and possibly others), making it hard to assign Array
# to any other type than Any for now.
# In contrast, sbi_ebm source code could relax this constraint by promising to
# be compatible with jax arrays only.
# TODO:  assign Array to jnp.array.
Array: TypeAlias = Any
Numeric = Union[Array, Scalar]

PyTreeNode: TypeAlias = Any

PRNGKeyArray: TypeAlias = prng.PRNGKeyArray

LogLikelihood_T = Callable[[Array, Array], Numeric]

GradLogDensity = Callable[[Array], Numeric]

Simulator_T = Callable[[Array], Array]


class LogDensity_T(Protocol):
    def __call__(self, x: Array, /) -> Numeric:
        ...


# class LogLikelihood_T(Protocol):
#     def __call__(self, theta: Array, x: Array) -> Numeric:
#         ...


class LogJoint_T(Protocol):
    def __call__(self, theta: Array, x: Array) -> Numeric:
        ...


class DoublyIntractableLogDensity_T(Protocol):
    log_prior: LogDensity_T
    log_likelihood: LogLikelihood_T
    x_obs: Array

    def __call__(self, x: Array) -> Numeric:
        ...


class DoublyIntractableJointLogDensity_T(Protocol):
    log_prior: LogDensity_T
    log_likelihood: LogLikelihood_T

    def __call__(self, x: Array) -> Numeric:
        ...
