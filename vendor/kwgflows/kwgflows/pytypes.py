from typing import Any, Callable, Iterable, Mapping, Protocol, Tuple, TypeVar, Union

import jax.numpy as jnp
from flax import struct
from jax import Array as jax_Array
from jax._src import prng
from jax.random import KeyArray
from typing_extensions import TypeAlias

Scalar = Union[float, int]

# jax source codes aliases Array to Any - the reason is that
# jax functions seeks to be compatible with other array classes such as
# np.array, jax tracers (and possibly others), making it hard to assign Array
# to any other type than Any for now.
# In contrast, sbi_ebm source code could relax this constraint by promising to
# be compatible with jax arrays only.
# TODO:  assign Array to jnp.array.
# Array: TypeAlias = Any
Array: TypeAlias = jax_Array
Numeric = Union[Array, Scalar]

PyTreeNode: TypeAlias = Any

# PRNGKeyArray: TypeAlias = prng.PRNGKeyArray
PRNGKeyArray: TypeAlias = KeyArray

Simulator_T = Callable[[Array], Array]

# The traditional pytree type definition
# PyTree = Union[
#    Array, Iterable["PyTree"], Mapping[Any, "PyTree"]
# ]
# does not differentiate between the different PyTreeDefs which is ultimately what
# we'd like in order to write type-safe code. Unfortinately, generic PyTreeDefs
# typevariables would require recursive type variables, which is not yet implemeted
# in python.

# For now, I'm mostly favouring genericism over recursion, but when genericism
# is not needed, I use the recusive union-type definition.
Array_or_PyTreeNode: TypeAlias = Union[Array, PyTreeNode]
T = TypeVar(
    "T",
    bound=Union[Array_or_PyTreeNode, Tuple[Array_or_PyTreeNode, ...]],
)
# XXX: Is there a better way to use multiple type with identical bounds?
T1 = TypeVar(
    "T1",
    bound=Union[Array_or_PyTreeNode, Tuple[Array_or_PyTreeNode, ...]],
)

T2 = TypeVar(
    "T2",
    bound=Union[Array_or_PyTreeNode, Tuple[Array_or_PyTreeNode, ...]],
)


PyTree_T = Union[
    struct.PyTreeNode, Array, Iterable["PyTree_T"], Mapping[Any, "PyTree_T"]
]
