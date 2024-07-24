import abc
from typing import Callable, Generic

from flax import struct

from kwgflows.base import DiscreteProbability
from kwgflows.pytypes import Scalar
from kwgflows.rkhs.rkhs import rkhs_element
from kwgflows.pytypes import T


class Divergence(Generic[T], struct.PyTreeNode, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, p: DiscreteProbability[T]) -> Scalar:
        raise NotImplementedError

    @abc.abstractmethod
    def get_first_variation(self, p: DiscreteProbability[T]) -> Callable[[T], Scalar]:
        # XXX: in theory, T should be the type of euclidean vectors.
        raise NotImplementedError


class KernelizedDivergence(Divergence[T]):
    @abc.abstractmethod
    def get_first_variation(self, p: DiscreteProbability[T]) -> rkhs_element[T]:
        raise NotImplementedError
