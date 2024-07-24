from typing import Any, Dict, List, Optional, Tuple, Type, cast
from jax import vmap

import jax.numpy as jnp
from calibration.utils import median_median_heuristic_discrete
from flax import struct
from kwgflows.pytypes import Array, Scalar
from kwgflows.rkhs.kernels import MedianHeuristicKernel, base_kernel, gaussian_kernel, scalable_kernel
from typing_extensions import Self, TypeAlias

from pmpnn_extra.dtw import DTW

Sequence_T: TypeAlias = Array


class HammingIMQKernel(base_kernel[Sequence_T]):
    # XXX: currenlty on
    alpha: float
    beta: float
    lag: int = struct.field(pytree_node=False, default=1)

    @staticmethod
    def hamming_distance(x, y, lag=1) -> Scalar:
        # nan != nan, so we need to subtract the number of common nans
        # (however, padding value is set to 99 right now),
        sims = (x == y) + jnp.all(jnp.isnan(jnp.stack((x, y), axis=0)), axis=0)
        diffs = 1 - sims
        for l in range(1, lag):
            diffs = diffs.at[..., :-l].set(diffs[..., :-l] + diffs[..., l:])
        return jnp.sum(diffs > 0)

    @classmethod
    def create(
        cls: Type[Self],
        *args: List[Any],
        alpha: float = 1.0,
        beta: float = 1.0,
        lag: int = 1,
        **kwargs: Dict[Any, Any],
    ) -> Self:
        assert (args, kwargs) == (
            (),
            {},
        ), "No positional or unknown keyword arguments allowed"
        return cls(alpha, beta, lag)

    def __call__(self, x: Sequence_T, y: Sequence_T) -> Scalar:
        ret = (
            (1 + self.alpha) / (self.alpha + self.hamming_distance(x, y, lag=self.lag))
        ) ** self.beta
        return ret


class ModifiedGaussianKernel(MedianHeuristicKernel[Array]):
    gaussian_kernel: gaussian_kernel

    @classmethod
    def create(
        cls: Type[Self], *args: List[Any], sigma: float = 1, **kwargs: Dict[Any, Any]
    ) -> Self:
        assert (args, kwargs) == (
            (),
            {},
        ), "No positional or unknown keyword arguments allowed"
        kernel = gaussian_kernel.create(sigma=sigma)
        return cls(kernel)

    def set_bandwidth(self, bandwidth: float) -> Self:
        return self.replace(
            gaussian_kernel=self.gaussian_kernel.set_bandwidth(bandwidth)
        )

    @property
    def bandwidth(self) -> float:
        return self.gaussian_kernel.bandwidth

    def __call__(self, x: Array, y: Array):
        gkxy = self.gaussian_kernel(x, y)
        return 0.5 * gkxy / (1 - 0.5 * gkxy)

    def with_median_heuristic(
        self,
        X: Array,
        indices: Optional[Tuple[Array, Array]] = None,
        quantile: Optional[float] = None,
        *,
        allow_nans: bool = False,
    ) -> Self:
        return self.replace(
            gaussian_kernel=self.gaussian_kernel.with_median_heuristic(
                X, indices, quantile, allow_nans=allow_nans
            )
        )


class DTWKernel(MedianHeuristicKernel[Array], scalable_kernel[Array]):
    # XXX: the inner kernel allows for a median heuristic
    # but not the outer kernel. Thus, the `sigma` parameter
    # inherited from the MedianHeuristicKernel base class
    # is meaningless right now.
    # TODO: set `sigma` as a @property and have it return
    # the sigma of the inner kernel. This requires changing
    # the semantics of the base class.
    dtw: DTW = DTW()

    @property
    def bandwidth(self) -> float:
        assert isinstance(self.dtw.kernel, MedianHeuristicKernel)
        return self.dtw.kernel.bandwidth

    def set_bandwidth(self, bandwidth: float) -> Self:
        assert isinstance(self.dtw.kernel, MedianHeuristicKernel)
        return self.replace(
            dtw=self.dtw.replace(kernel=self.dtw.kernel.set_bandwidth(bandwidth))
        )

    def calibrate_scale(self, X: Array) -> Self:
        gm = vmap(vmap(lambda x, y: self.dtw(x, y, log_space=True), (None, 0)), (0, None))(
            X, X
        )
        log_scale = jnp.max(jnp.squeeze(gm))
        print("found log scale", log_scale)
        return self.replace(log_scale=log_scale)

    @classmethod
    def create(
        cls: Type[Self],
        *args: List[Any],
        gamma: float = 1,
        kernel: base_kernel[Array] = gaussian_kernel.create(sigma=1.0),
        **kwargs: Dict[Any, Any],
    ) -> Self:
        assert (args, kwargs) == (
            (),
            {},
        ), "No positional or unknown keyword arguments allowed"
        return cls(dtw=DTW(gamma=gamma, kernel=kernel))

    def __call__(self, x: Array, y: Array) -> Scalar:
        return cast(Scalar, jnp.exp(self.dtw(x, y, log_space=True) - self.log_scale))

    @staticmethod
    def pad_inputs(xs) -> Array:
        # xs is a list of arrays of possible different lengths
        # this function pads them to the same length (max length)
        # using nans
        max_len = max([x.shape[0] for x in xs])
        padded_xs = []
        for x in xs:
            pad_len = max_len - x.shape[0]
            padded_x = jnp.pad(x, (0, pad_len), constant_values=jnp.nan)
            padded_xs.append(padded_x)
        return jnp.array(padded_xs)

    def with_median_heuristic(
        self,
        X: Array,
        indices: Optional[Tuple[Array, Array]] = None,
        quantile: Optional[float] = None,
        inner_indices: Optional[Tuple[Array, Array]] = None,
        *,
        # true by default since DTW is typically used with
        # variable length sequences, which are padded with nans.
        allow_nans: bool = True,
    ) -> Self:
        # print(indices)
        kernel = self.dtw.kernel
        assert isinstance(kernel, MedianHeuristicKernel)
        if isinstance(kernel, MedianHeuristicKernel):
            sigma = median_median_heuristic_discrete(
                kernel, X, indices,
                quantile,
                inner_indices,
                allow_nans=allow_nans,
            )
            self = self.set_bandwidth(sigma)
        # dtw's parameter is a softmin parameter, not a bandwith one. Thus,
        # only perform ground-space median heuristic.
        return self
