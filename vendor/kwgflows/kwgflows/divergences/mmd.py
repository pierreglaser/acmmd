from flax import struct
from typing import Generic

from kwgflows.base import DiscreteProbability
from kwgflows.divergences.base import KernelizedDivergence
from kwgflows.rkhs.kernels import base_kernel
from kwgflows.rkhs.rkhs import rkhs_element
from kwgflows.pytypes import Scalar, T


class mmd(Generic[T], struct.PyTreeNode):
    kernel: base_kernel[T]
    squared: bool = True

    def get_witness_function(
        self, P: DiscreteProbability[T], Q: DiscreteProbability[T]
    ) -> rkhs_element[T]:
        return P.get_mean_embedding(self.kernel) - Q.get_mean_embedding(self.kernel)

    def __call__(self, P: DiscreteProbability[T], Q: DiscreteProbability[T]) -> Scalar:
        witness_function = self.get_witness_function(P, Q)
        return witness_function.rkhs_norm(squared=self.squared)


class mmd_fixed_Q(Generic[T], KernelizedDivergence[T]):
    kernel: base_kernel[T]
    Q: DiscreteProbability[T]

    def __call__(self, p: DiscreteProbability[T]) -> Scalar:
        return mmd(self.kernel)(p, self.Q)

    def get_first_variation(self, p: DiscreteProbability[T]) -> rkhs_element[T]:
        return mmd(self.kernel).get_witness_function(p, self.Q)
