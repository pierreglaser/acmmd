import abc
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, Type, cast

import jax
import jax.numpy as jnp
from flax import struct
from jax import vmap
from typing_extensions import Self

from kwgflows.pytypes import T1, T2, Array, Numeric, Scalar, T
from kwgflows.utils import infer_num_samples_pytree


def _rescale(x: Array, scale: Numeric) -> Array:
    return x / scale


def _l2_norm_squared(x: Array) -> Array:
    return jnp.sum(jnp.square(x))


def median_heuristic(
    metric: Callable[[T, T], Scalar],
    X: T,
    indices: Optional[Tuple[Array, Array]] = None,
    quantile: Optional[float] = None,
    allow_nans: bool = False,
) -> Scalar:
    """
    Return the median of the distances of elements in :param:`X` with indices :param:`indices[0]`
    and :param:`indices[1]` for the metric used by the :param:`kernel`.

    If :param:`indices` is `None` (the default), pairwise distances between all elements are computed.

    Edge cases:
    - If all distances are 0, returns 1.
    - If the true median is 0, the median is computed over the the set of distances
      where the 0 distances are ignored.
    """
    # Compute all relevant distances
    if indices is None:
        # All pairwise distances
        dists = jnp.asarray(
            jax.vmap(jax.vmap(metric, in_axes=(0, None)), in_axes=(None, 0))(X, X)
        )
        dists = dists[jnp.triu_indices_from(dists, k=1)]
    else:
        # Extract subset of samples between distances are computed
        indices1, indices2 = indices
        X1 = jax.tree_map(lambda x: x[indices1], X)
        X2 = jax.tree_map(lambda x: x[indices2], X)

        dists = jnp.asarray(jax.vmap(metric, in_axes=(0, 0))(X1, X2))

    if allow_nans:
        if quantile is not None:
            median_func = lambda x: jnp.nanquantile(x, quantile)
        else:
            median_func = jnp.nanmedian
    else:
        if quantile is not None:
            median_func = lambda x: jnp.quantile(x, quantile)
        else:
            median_func = jnp.median
    quantile = quantile or 0.5

    # Quick check that median is non-zero
    # XXX: jnp.median scales poorly: https://github.com/google/jax/issues/4379
    nonzero_dists = jnp.count_nonzero(dists)
    return jax.lax.switch(
        (nonzero_dists > 0).astype(int)
        + (nonzero_dists >= jnp.size(dists) * quantile).astype(int),
        [
            # all distances zero -> return 1.0
            lambda _: 1.0,
            # distances with zero median -> return median with zero distances ignored instead
            lambda dists: cast(
                Scalar,
                jnp.nanquantile(jnp.where(dists == 0.0, jnp.nan, dists), quantile),
            ),
            # distances with non-zero median -> return median
            lambda dists: cast(Scalar, median_func(dists)),
        ],
        dists,
    )


class base_kernel(Generic[T], struct.PyTreeNode, metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def create(cls: Type[Self], *args: List[Any], **kwargs: Dict[Any, Any]) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, x: T, y: T) -> Scalar:
        """Evaluate kernel :param:`self` with arguments :param:`x` and :param:`y`."""
        raise NotImplementedError

    def make_gram_matrix(self, X: T, Y: T) -> Array:
        # X has to be a vmapped instance of T (ditto for Y)
        gm = vmap(vmap(type(self).__call__, (None, None, 0)), (None, 0, None))(
            self, X, Y
        )
        # casting becauce vmap transformation in return type from numeric to array
        # is not impemented yet
        return cast(Array, gm)


class scalable_kernel(Generic[T], struct.PyTreeNode, metaclass=abc.ABCMeta):
    log_scale: Numeric = 1.0

    @abc.abstractmethod
    def calibrate_scale(self, X: T) -> Self:
        raise NotImplementedError


class BandwidthBasedKernel(base_kernel[T], metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def bandwidth(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def set_bandwidth(self, bandwidth: float) -> Self:
        raise NotImplementedError


class SigmaBandwidthKernel(BandwidthBasedKernel[T]):
    sigma: float

    @property
    def bandwidth(self) -> float:
        return self.sigma

    def set_bandwidth(self, bandwidth: float) -> Self:
        return self.replace(sigma=bandwidth)


class MedianHeuristicKernel(BandwidthBasedKernel[T], metaclass=abc.ABCMeta):
    """
    An abstract base class for bandwidth-based kernels that admit median
    heuristic tuning techniques.

    ...

    The bandwidth of the kernel will be named :attr:`sigma` in all subclasses.
    """

    @abc.abstractmethod
    def with_median_heuristic(
        self,
        X: T,
        indices: Optional[Tuple[Array, Array]] = None,
        quantile: Optional[float] = None,
        allow_nans: bool = False,
    ) -> Self:
        r"""Compute the median distance of elements in :param:`X` with
        indices :param:`indices[0]` and :param:`indices[1]`, using the
        distance measure of the kernel :param:`self`.

        If :param:`indices` is `None` (the default), pairwise distances
        between all elements are computed.

        ...

        The median pairwise distance of samples :math:`\{x_1, \ldots, x_n\}`
        using distance measure :math:`d(\cdot, \cdot)` is defined as

        .. math:: \operatorname{median}\{d(x_i, x_j) \colon 1 \leq i < j \leq n \}.
        """
        raise NotImplementedError


class RBFKernel(
    MedianHeuristicKernel[T], SigmaBandwidthKernel[T], metaclass=abc.ABCMeta
):
    r"""
    Class that represents radial-basis function kernels.

    ..

    Such kernels are of the form

    ..math::
      k(x, y) = \phi(d(x, y) / \sigma)

    for suitable arguments :math:`x` and :math:`y`,
    where :math:`d` is a base metric, :math:`\sigma` is a length-scale parameter,
    and :math:`\phi \colon \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}` is a function.
    """

    def __call__(self, x: T, y: T) -> Scalar:
        return self.phi(self.distance(x, y))

    def with_median_heuristic(
        self,
        X: T,
        indices: Optional[Tuple[Array, Array]] = None,
        quantile: Optional[float] = None,
        *,
        allow_nans: bool = False,
    ) -> Self:
        sigma = median_heuristic(
            self.distance, X, indices, quantile, allow_nans=allow_nans
        )
        return self.set_bandwidth(sigma * self.sigma)

    @abc.abstractmethod
    def distance(self, x: T, y: T) -> Scalar:
        r"""Compute the distance between :param:`x` and :param:`y` using the
        base metric of the radial-basis function kernel :param:`self`,
        scaled by the length-scale of the kernel.

        ..

        More precisely, the function returns

        ..math:: d(x, y) / \sigma

        where :math:`d` is the base metric of the kernel and :math:`\sigma` its length-scale
        parameter.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def phi(self, t: Scalar) -> Scalar:
        """Evaluate the radial-basis function kernel :param:`self` given the
        evaluation :param:`t` of the distance between two inputs, scaled by the
        length-scale parameter.
        """
        raise NotImplementedError


class SquaredRBFKernel(RBFKernel[T], metaclass=abc.ABCMeta):
    r"""
    Class that represents radial-basis function kernels of the form

    ..math::
      k(x, y) = \phi(d(x, y)^2 / \sigma^2)

    for suitable arguments :math:`x` and :math:`y`,
    where :math:`d` is a base metric, :math:`\sigma` is a length-scale parameter,
    and :math:`\phi \colon \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}` is a function.
    """

    def __call__(self, x: T, y: T) -> Scalar:
        return self.phi_squared(self.squared_distance(x, y))

    def distance(self, x: T, y: T) -> Scalar:
        return cast(Scalar, jnp.sqrt(self.squared_distance(x, y)))

    def phi(self, t: Scalar) -> Scalar:
        return cast(Scalar, self.phi_squared(t**2))

    @abc.abstractmethod
    def squared_distance(self, x: T, y: T, /) -> Scalar:
        r"""Compute the squared distance between :param:`x` and :param:`y` using the
        base metric of the kernel :param:`self`, scaled by the length-scale
        of the kernel.

        ..

        More precisely, the function returns

        ..math:: d(x, y)^2 / \sigma^2

        where :math:`d` is the base metric of the kernel and :math:`\sigma` its
        length-scale parameter.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def phi_squared(self, t_squared: Scalar) -> Scalar:
        """Evaluate the radial-basis function kernel :param:`self` given the
        evaluation :param:`t_squared` of the squared distance between two inputs,
        scaled by the length-scale parameter.
        """
        raise NotImplementedError


class SquaredEuclideanRBFKernel(SquaredRBFKernel[Array]):
    def squared_distance(self, x: Array, y: Array, /) -> Scalar:
        return cast(Scalar, jnp.sum(jnp.square((x - y) / self.sigma)))


class GaussianRBFKernel(SquaredRBFKernel[T]):
    def phi_squared(self, t_squared: Scalar) -> Scalar:
        return cast(Scalar, jnp.exp(-0.5 * t_squared))


class gaussian_kernel(GaussianRBFKernel[Array], SquaredEuclideanRBFKernel):
    @classmethod
    def create(
        cls: Type[Self], *args: List[Any], sigma: float = 1, **kwargs: Dict[Any, Any]
    ) -> Self:
        assert (args, kwargs) == (
            (),
            {},
        ), "No positional or unknown keyword arguments allowed"
        return cls(sigma)


class laplace_kernel(RBFKernel[Array]):
    @classmethod
    def create(
        cls: Type[Self], *args: List[Any], sigma: float = 1.0, **kwargs: Dict[Any, Any]
    ) -> Self:
        assert (args, kwargs) == (
            (),
            {},
        ), "No positional or unknown keyword arguments allowed"
        return cls(sigma)

    def distance(self, x: Array, y: Array) -> Scalar:
        return cast(Scalar, jnp.sum(jnp.abs((x - y) / self.sigma)))

    def phi(self, t: Scalar) -> Scalar:
        return cast(Scalar, jnp.exp(-t))


class imq_kernel(SquaredEuclideanRBFKernel):
    c: float = 1.0
    beta: float = -0.5

    @classmethod
    def create(
        cls: Type[Self],
        *args: List[Any],
        sigma: float = 1,
        c: float = 1.0,
        beta: float = -0.5,
        **kwargs: Dict[Any, Any],
    ) -> Self:
        assert (args, kwargs) == (
            (),
            {},
        ), "No positional or unknown keyword arguments allowed"
        return cls(sigma, c, beta)

    def phi_squared(self, t_squared: Scalar) -> Scalar:
        return cast(
            Scalar,
            jnp.power(self.c**2 + t_squared, self.beta),
        )


class negative_distance_kernel(SquaredEuclideanRBFKernel):
    @classmethod
    def create(
        cls: Type[Self], *args: List[Any], sigma: float = 1, **kwargs: Dict[Any, Any]
    ) -> Self:
        assert (args, kwargs) == (
            (),
            {},
        ), "No positional or unknown keyword arguments allowed"
        return cls(sigma)

    def phi_squared(self, t_squared: Scalar) -> Scalar:
        return -t_squared


class energy_kernel(base_kernel[Array]):
    # x0: Array
    beta: float
    sigma: float
    eps: float = 1e-8

    @classmethod
    def create(
        cls: Type[Self],
        *args: List[Any],
        beta: float = 1,
        sigma: float = 1,
        eps: float = 1e-8,
        **kwargs: Dict[Any, Any],
    ) -> Self:
        assert (args, kwargs) == (
            (),
            {},
        ), "No positional or unknown keyword arguments allowed"
        return cls(beta, sigma, eps)

    def __call__(self, x: Array, y: Array) -> Scalar:
        x0 = jnp.zeros_like(x)

        pxx0 = jnp.power(
            _l2_norm_squared(_rescale(x - x0, self.sigma)) + self.eps, self.beta / 2
        )
        pyx0 = jnp.power(
            _l2_norm_squared(_rescale(y - x0, self.sigma)) + self.eps, self.beta / 2
        )
        pxy = jnp.power(
            _l2_norm_squared(_rescale(x - y, self.sigma)) + self.eps, self.beta / 2
        )

        ret = 0.5 * (pxx0 + pyx0 - pxy)
        return cast(Scalar, ret)


class TensorProductKernel(MedianHeuristicKernel[Tuple[T1, T2]]):
    k_x: base_kernel[T1]
    k_y: base_kernel[T2]


    @property
    def bandwidth(self) -> float:
        assert isinstance(self.k_x, BandwidthBasedKernel)
        return self.k_x.bandwidth

    @classmethod
    def create(
        cls: Type[Self],
        *args: List[Any],
        k_x: Optional[base_kernel[T1]] = None,
        k_y: Optional[base_kernel[T2]] = None,
        **kwargs: Dict[Any, Any],
    ) -> Self:
        assert (args, kwargs) == (
            (),
            {},
        ), "No positional or unknown keyword arguments allowed"
        assert k_x is not None, "k_x must be specified"
        assert k_y is not None, "k_y must be specified"
        return cls(k_x, k_y)

    def __call__(self, x: Tuple[T1, T2], y: Tuple[T1, T2]) -> Scalar:
        xx, xy = x
        yx, yy = y
        return self.k_x(xx, yx) * self.k_y(xy, yy)

    def with_median_heuristic(
        self,
        X: Tuple[T1, T2],
        indices: Optional[Tuple[Array, Array]] = None,
        quantile: Optional[float] = None,
        allow_nans: bool = False,
    ):
        X1, X2 = X
        assert isinstance(self.k_x, MedianHeuristicKernel)
        assert isinstance(self.k_y, MedianHeuristicKernel)

        k_x = self.k_x.with_median_heuristic(
            X1, indices=indices, quantile=quantile, allow_nans=allow_nans
        )
        print(f"[TensorProductKernel]: median-based bandwidth for k_x: {k_x.bandwidth}")

        k_y = self.k_y.with_median_heuristic(
            X2, indices=indices, quantile=quantile, allow_nans=allow_nans
        )
        print(f"[TensorProductKernel]: median-based bandwidth for k_y: {k_y.bandwidth}")

        return self.replace(k_x=k_x, k_y=k_y)

    def set_bandwidth(self, bandwidth: float) -> Self:
        # XXX: does not comply with bandwidth protocol
        raise NotImplementedError
