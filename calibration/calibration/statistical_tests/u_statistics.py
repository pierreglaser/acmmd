import abc
from abc import abstractmethod
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar, cast

import jax
import jax.numpy as jnp
from flax import struct
from jax import tree_map, vmap
from jax.scipy.stats.norm import ppf
from jax_samplers.inference_algorithms.mcmc.base import MCMCConfig
from kwgflows.pytypes import (
    Array,
    Array_or_PyTreeNode,
    Numeric,
    PRNGKeyArray,
    Scalar,
    T,
)
from kwgflows.utils import infer_num_samples_pytree
from typing_extensions import TypeGuard

from calibration.logging import get_logger

from ..utils import fill_diagonal, superdiags_indices


class UStatFn(Generic[T], struct.PyTreeNode, metaclass=abc.ABCMeta):
    @abstractmethod
    def __call__(self, z: T, z_p: T) -> Scalar:
        raise NotImplementedError


class BaseApproximationState(struct.PyTreeNode):
    pass


ApproxState_T = TypeVar("ApproxState_T", bound=BaseApproximationState)


class ApproximableUStatFn(UStatFn[T], Generic[T, ApproxState_T], metaclass=abc.ABCMeta):
    @abstractmethod
    def make_approximation_state(
        self,
        z: T,
        num_particles: int,
        key: PRNGKeyArray,
        mcmc_config: Optional[MCMCConfig[Any, Any, Any]] = None,
    ) -> ApproxState_T:
        raise NotImplementedError

    @abstractmethod
    def maybe_approximate_internals(
        self,
        num_particles: int,
        key: PRNGKeyArray,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def call_approximate(
        self,
        z_and_approx: Tuple[T, ApproxState_T],
        z_p_and_approx: Tuple[T, ApproxState_T],
        approx_internals: Any,
    ) -> Scalar:
        raise NotImplementedError


UStatFn_T_co = TypeVar("UStatFn_T_co", bound=UStatFn[Any], covariant=True)


class BaseUStat(Generic[UStatFn_T_co], struct.PyTreeNode, metaclass=abc.ABCMeta):
    stat_fn: UStatFn_T_co

    @abstractmethod
    def _map(self, f: Callable[[T, T], Scalar], X: T) -> Array:
        raise NotImplementedError

    def _map_exact_stat_fn(self: "BaseUStat[UStatFn[T]]", X: T) -> Array:
        return self._map(self.stat_fn.__call__, X)

    def _map_approx_stat_fn(
        self: "BaseUStat[ApproximableUStatFn[T, ApproxState_T]]",
        X: T,
        num_particles: int,
        key: PRNGKeyArray,
        mcmc_config: Optional[MCMCConfig[Any, Any, Any]] = None,
    ) -> Array:
        logger = get_logger("calibration.u_statistics")
        logger.info(
            f"mapping approx stat fn over eval indices (num_particles={num_particles})"
        )
        n = infer_num_samples_pytree(X)
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, n)
        approx_state = vmap(
            self.stat_fn.make_approximation_state, in_axes=(0, None, 0, None)
        )(X, num_particles, subkeys, mcmc_config)
        key, subkey = jax.random.split(key)
        approx_internals = self.stat_fn.maybe_approximate_internals(
            num_particles, subkey
        )

        def call_approximate(
            z_and_approx: Tuple[T, ApproxState_T],
            z_p_and_approx: Tuple[T, ApproxState_T],
        ) -> Scalar:
            return self.stat_fn.call_approximate(
                z_and_approx, z_p_and_approx, approx_internals
            )

        return self._map(call_approximate, (X, approx_state))

    @abstractmethod
    def _compute_from_evals(
        self: "BaseUStat[UStatFn[T]]", func_evals: Array, X: T
    ) -> Scalar:
        raise NotImplementedError

    def compute_approx(
        self: "BaseUStat[ApproximableUStatFn[T, ApproxState_T]]",
        X: T,
        num_particles: int,
        key: PRNGKeyArray,
        mcmc_config: Optional[MCMCConfig[Any, Any, Any]] = None,
    ) -> Tuple[Scalar, Array]:
        func_evals = self._map_approx_stat_fn(X, num_particles, key, mcmc_config)
        return self._compute_from_evals(func_evals, X), func_evals

    def compute(self: "BaseUStat[UStatFn[T]]", X: T) -> Tuple[Scalar, Array]:
        func_evals = self._map_exact_stat_fn(X)
        return self._compute_from_evals(func_evals, X), func_evals

    @abstractmethod
    def get_null_quantile_wild_bootstrap(
        self: "BaseUStat[UStatFn[T]]",
        X: T,
        alpha: Numeric,
        num_permutations: int,
        key: PRNGKeyArray,
        cache: Optional[Array] = None,
    ) -> Tuple[Scalar, Array]:
        raise NotImplementedError


def is_approximable(
    ustat: BaseUStat[UStatFn[T]],
) -> TypeGuard["BaseUStat[ApproximableUStatFn[T, BaseApproximationState]]"]:
    return isinstance(ustat.stat_fn, ApproximableUStatFn)


class AnalyticalAsymptoticNullUStat(BaseUStat[UStatFn_T_co]):
    @abstractmethod
    def get_analytical_asymptotic_null_quantile(
        self: "AnalyticalAsymptoticNullUStat[UStatFn[T]]",
        X: T,
        alpha: Numeric,
        cache: Optional[Array] = None,
    ) -> Tuple[Scalar, Any]:
        raise NotImplementedError


class CompleteUStat(BaseUStat[UStatFn_T_co]):
    def _map(self, f: Callable[[T, T], Scalar], X: T) -> Array:
        backend = "scan"
        if backend == "vmap":
            vmapped_func = vmap(vmap(f, in_axes=(None, 0)), in_axes=(0, None))
            func_evals = cast(Array, vmapped_func(X, X))
            func_evals = fill_diagonal(func_evals, 0.0)
        elif backend == "scan":
            def step_fn(_: Array, x: T) -> Tuple[Any, Scalar]:
                def inner_step_fn(_: Array, y: T) -> Tuple[Any, Scalar]:
                    return None, f(x, y)

                return jax.lax.scan(inner_step_fn, None, X)

            _, func_evals = cast(Array, jax.lax.scan(step_fn, None, X))
            func_evals = fill_diagonal(func_evals, 0.0)
        else:
            raise ValueError

        return func_evals

    def _compute_from_evals(
        self: "CompleteUStat[UStatFn[T]]", func_evals: Array, X: T
    ) -> Scalar:
        logger = get_logger("calibration.u_statistics")
        logger.info("computing from evals (exact)")
        n = infer_num_samples_pytree(X)
        ret = jnp.sum(func_evals) / (n * (n - 1))
        return cast(Scalar, ret)

    def get_null_quantile_wild_bootstrap(
        self: "CompleteUStat[UStatFn[T]]",
        X: T,
        alpha: Numeric,
        num_permutations: int,
        key: PRNGKeyArray,
        cache: Optional[Array] = None,
    ) -> Tuple[Scalar, Array]:
        """
        Estimate the U-statistic using as input permutations of the sample.

        Instead of naively vmapping the `estimate` method over permutations of X,
        the implementation avoids recomputation of the kernel matrix using
        the rademacher variables [2, Section 4], and uses Blas-2 level operations
        in order to be as efficient as possible.

        [2] Efficient Aggregated Kernel Tests using Incomplete U-statistics,
        Schrab et al. (2022)
        """
        n = infer_num_samples_pytree(X)
        if cache is None:
            _, stat_fn_evals = self.compute(X)
        else:
            stat_fn_evals = cache

        def ustat_1_perm(key: PRNGKeyArray) -> Scalar:
            perms = jax.random.rademacher(key, shape=(n,))
            return cast(Scalar, 1 / (n * (n - 1)) * perms.T @ stat_fn_evals @ perms)

        keys = jax.random.split(key, num_permutations)
        h0_vals = cast(Array, vmap(ustat_1_perm)(keys))
        quantile_ = cast(Scalar, jnp.quantile(h0_vals, 1 - alpha))
        return quantile_, h0_vals


class IncompleteUStat(BaseUStat[UStatFn_T_co]):
    @abstractmethod
    def generate_indices(self, X: Array_or_PyTreeNode) -> Tuple[Array, Array]:
        raise NotImplementedError

    def _map(self, f: Callable[[T, T], Scalar], X: T) -> Array:
        x_indices, y_indices = self.generate_indices(X)

        def f_slice(i: Numeric, j: Numeric) -> Scalar:
            x_i = cast(T, tree_map(lambda z: z[i], X))
            x_j = cast(T, tree_map(lambda z: z[j], X))
            return f(x_i, x_j)

        vmapped_stat_fn = vmap(f_slice, in_axes=(0, 0))
        stat_fn_evals = cast(Array, vmapped_stat_fn(x_indices, y_indices))
        return stat_fn_evals

    def _compute_from_evals(
        self: "IncompleteUStat[UStatFn[T]]", func_evals: Array, X: T
    ) -> Scalar:
        return cast(Scalar, jnp.average(func_evals))

    def get_null_quantile_wild_bootstrap(
        self: "IncompleteUStat[UStatFn[T]]",
        X: T,
        alpha: Numeric,
        num_permutations: int,
        key: PRNGKeyArray,
        cache: Optional[Array] = None,
    ) -> Tuple[Scalar, Array]:
        """
        Incompletely Estimate the U-statistic using as input permutations of the sample.

        Restricting the set of permutations to those for which the kernel values have
        already been computed for the original incomplete U-statistic corresponds
        exactly to using a wild bootstrap. See [2, Section 4]

        Also performs a low-level optimization that avoids using a N x N matrix of
        U-statistic evaluations.

        [2] Efficient Aggregated Kernel Tests using Incomplete U -statistics,
        Schrab et al. (2021)
        """
        n = infer_num_samples_pytree(X)
        indices = self.generate_indices(X)

        if cache is None:
            _, stat_fn_evals = self.compute(X)
        else:
            stat_fn_evals = cache

        x_indices, y_indices = indices
        n_vals = x_indices.shape[0]

        def ustat_2_perm(key: PRNGKeyArray) -> Scalar:
            perms = jax.random.rademacher(key, shape=(n,))
            e_i = perms[x_indices]
            e_j = perms[y_indices]
            return cast(Scalar, 1 / n_vals * ((e_i * e_j) @ stat_fn_evals))

        keys = jax.random.split(key, num_permutations)

        h0_vals = cast(Array, vmap(ustat_2_perm)(keys))
        quantile_ = cast(Scalar, jnp.quantile(h0_vals, 1 - alpha))
        return quantile_, h0_vals


class SuperDiagonalIncompleteUStat(IncompleteUStat[UStatFn_T_co]):
    R: int = struct.field(pytree_node=False, default=1)

    def generate_indices(self, X: Array_or_PyTreeNode) -> Tuple[Array, Array]:
        N = infer_num_samples_pytree(X)
        return superdiags_indices(N, self.R)


class FirstSuperDiagonalIncompleteUStat(
    SuperDiagonalIncompleteUStat[UStatFn_T_co],
    AnalyticalAsymptoticNullUStat[UStatFn_T_co],
):
    R: int = struct.field(pytree_node=False, default=1)

    def generate_indices(self, X: Array_or_PyTreeNode) -> Tuple[Array, Array]:
        n = infer_num_samples_pytree(X)
        assert n % 2 == 0
        return (jnp.arange(0, n, step=2), jnp.arange(1, n, step=2))

    def _map(self, f: Callable[[T, T], Scalar], X: T) -> Array:
        n = infer_num_samples_pytree(X)
        n_pair = int(2 * (n // 2))

        X1 = tree_map(lambda x: x[:n_pair:2], X)
        X2 = tree_map(lambda x: x[1:n_pair:2], X)

        stat_evals = cast(Array, vmap(f, in_axes=(0, 0))(X1, X2))
        return stat_evals

    def compute(
        self: "FirstSuperDiagonalIncompleteUStat[UStatFn[T]]", X: T
    ) -> Tuple[Scalar, Array]:
        func_evals = self._map_exact_stat_fn(X)
        return cast(Scalar, jnp.average(func_evals)), func_evals

    def get_analytical_asymptotic_null_quantile(
        self: "FirstSuperDiagonalIncompleteUStat[UStatFn[T]]",
        X: T,
        alpha: Numeric,
        cache: Optional[Array] = None,
    ) -> Tuple[Scalar, Any]:
        n = infer_num_samples_pytree(X)
        n_pair = int(2 * (n // 2))

        if cache is None:
            _, stat_fn_evals = self.compute(X)
        else:
            stat_fn_evals = cache

        std = jnp.sqrt(jnp.mean(stat_fn_evals**2))  # mean is 0.
        quantile = cast(Scalar, ppf(1 - alpha, loc=0, scale=std / jnp.sqrt(n_pair / 2)))
        return quantile, None
