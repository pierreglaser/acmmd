import abc
from abc import abstractmethod
from typing import Any, Generic, Literal, Optional, Tuple, TypeVar, Union, cast

import jax
from flax import struct
from jax import numpy as jnp
from jax_samplers.inference_algorithms.mcmc.base import MCMCConfig
from jax_samplers.kernels.mala import MALAConfig, MALAKernelFactory
from jax_samplers.kernels.ula import ULAConfig, ULAKernelFactory
from kwgflows.pytypes import Array, Array_or_PyTreeNode, PRNGKeyArray, Scalar, T
from kwgflows.rkhs.kernels import base_kernel
from kwgflows.utils import infer_num_samples_pytree
from typing_extensions import TypeVarTuple, Unpack

from calibration.statistical_tests.u_statistics import (
    AnalyticalAsymptoticNullUStat,
    BaseUStat,
    CompleteUStat,
    FirstSuperDiagonalIncompleteUStat,
    SuperDiagonalIncompleteUStat,
    UStatFn,
    is_approximable,
)
from calibration.utils import superdiags_indices


class TestResults(struct.PyTreeNode):
    result: bool
    val: Scalar
    h0_vals: Optional[Array] = None

    def to_dict(self):
        assert self.h0_vals is not None
        if isinstance(self.result, jax.Array):
            result = cast(jax.Array, self.result).item()
        else:
            result = self.result

        if isinstance(self.val, jax.Array):
            val = cast(jax.Array, self.val).item()
        else:
            val = self.val

        return {
            "result": result,
            "val": val,
            "h0_vals": self.h0_vals.tolist(),
            "p_value": jnp.sum(self.h0_vals > self.val) / len(self.h0_vals),
            "num_permutations": len(self.h0_vals),
        }


TestInput_T = TypeVarTuple("TestInput_T")

OneSample_T = TypeVar(
    "OneSample_T",
    bound=Union[Array_or_PyTreeNode, Tuple[Array_or_PyTreeNode, ...]],
)


class OneSampleTest(
    Generic[OneSample_T, Unpack[TestInput_T]], struct.PyTreeNode, metaclass=abc.ABCMeta
):
    num_permutations: int = struct.field(pytree_node=False, default=100)
    alpha: float = struct.field(pytree_node=True, default=0.05)
    indices_subset: Literal["complete", "superdiagonals", "auto"] = struct.field(
        pytree_node=False, default="auto"
    )
    R: Optional[int] = struct.field(pytree_node=False, default=None)
    prefer_analytical_quantiles: bool = struct.field(pytree_node=False, default=True)
    median_heuristic: bool = struct.field(pytree_node=False, default=False)
    # 5_000 distance evaluations corresponds to pairwise evaluation of ~100 samples
    max_num_distances: int = struct.field(pytree_node=False, default=5_000)

    def generate_indices(
        self, key: jax.random.KeyArray, X: OneSample_T, max_num: Optional[int] = None
    ) -> Tuple[Array, Array]:
        if max_num is None:
            max_num = self.max_num_distances

        n = infer_num_samples_pytree(X)
        if self.R is None:
            assert self.indices_subset in ("auto", "complete")
            indices = jnp.triu_indices(n=n, k=1)
        else:
            assert self.indices_subset != "complete"
            if self.indices_subset in ("auto", "superdiagonals"):
                if self.R == 1:
                    indices = (
                        jnp.arange(0, stop=n - 1, step=2),
                        jnp.arange(0, stop=n, step=2),
                    )
                else:
                    indices = superdiags_indices(n, self.R)
            else:
                raise ValueError(f"Unknown indices_subset: {self.indices_subset}")

        # Use a random subset of indices if number of indices exceeds `max_num`
        num = jnp.size(indices[0])
        if num > max_num:
            subset = jax.random.choice(key, num, (max_num,))
            indices = (indices[0][subset], indices[1][subset])

        return indices

    @abstractmethod
    def _get_u_stat_fn(
        self, *args: Unpack[TestInput_T], key: jax.random.KeyArray
    ) -> UStatFn[OneSample_T]:
        raise NotImplementedError

    @abstractmethod
    def _build_one_sample(self, *args: Unpack[TestInput_T]) -> OneSample_T:
        raise NotImplementedError

    # Arguments of a generic signature not part of the variadic args tuple
    # (like `key`) are keyword-only.
    def __call__(self, *args: Unpack[TestInput_T], key: PRNGKeyArray) -> TestResults:
        u_stat_fn = self._get_u_stat_fn(*args, key=key)
        u_stat = self._get_u_stat(u_stat_fn)
        one_sample = self._build_one_sample(*args)
        return self._perform_test(one_sample, u_stat, key)

    def _get_u_stat(
        self, u_stat_fn: UStatFn[OneSample_T]
    ) -> BaseUStat[UStatFn[OneSample_T]]:
        if self.R is None:
            assert self.indices_subset in ("auto", "complete")
            return CompleteUStat(stat_fn=u_stat_fn)
        else:
            assert self.indices_subset != "complete"
            if self.indices_subset in ("auto", "superdiagonals"):
                if self.R == 1:
                    return FirstSuperDiagonalIncompleteUStat(stat_fn=u_stat_fn)
                else:
                    return SuperDiagonalIncompleteUStat(stat_fn=u_stat_fn, R=self.R)
            else:
                raise ValueError(f"Unknown indices_subset: {self.indices_subset}")

    def compute_quantile(
        self, X: T, u_stat: BaseUStat[UStatFn[T]], func_evals: Array, key: PRNGKeyArray
    ) -> Tuple[Scalar, Array]:
        if (
            isinstance(u_stat, AnalyticalAsymptoticNullUStat)
            and self.prefer_analytical_quantiles
        ):
            quantile_, h0_vals = u_stat.get_analytical_asymptotic_null_quantile(
                X, self.alpha, func_evals
            )
        else:
            quantile_, h0_vals = u_stat.get_null_quantile_wild_bootstrap(
                X, self.alpha, self.num_permutations, key, func_evals
            )
        return quantile_, h0_vals

    def _perform_test(
        self, X: T, u_stat: BaseUStat[UStatFn[T]], key: PRNGKeyArray
    ) -> TestResults:
        ustat_val, func_evals = u_stat.compute(X)
        quantile_, h0_vals = self.compute_quantile(X, u_stat, func_evals, key)
        return TestResults(
            result=cast(bool, ustat_val > quantile_),
            val=ustat_val,
            h0_vals=h0_vals,
        )

    def unbiased_estimate(
        self, *args: Unpack[TestInput_T], key: jax.random.KeyArray
    ) -> Scalar:
        # Semi-public - undocumented, unadvertised.
        u_stat_fn = self._get_u_stat_fn(*args, key=key)
        u_stat = self._get_u_stat(u_stat_fn)
        one_sample = self._build_one_sample(*args)
        return u_stat.compute(one_sample)[0]


class ApproximableTest(OneSampleTest[OneSample_T, Unpack[TestInput_T]]):
    approximate: bool = struct.field(pytree_node=False, default=False)
    approximation_num_particles: int = struct.field(pytree_node=False, default=10)
    approximation_mcmc_config: Optional[MCMCConfig[Any, Any, Any]] = None
    approximation_mcmc_num_warmup_steps: int = struct.field(
        pytree_node=False, default=10
    )

    def get_mcmc_config(self) -> MCMCConfig[Any, Any, Any]:
        if self.approximation_mcmc_config is None:
            return MCMCConfig(
                num_samples=10,  # not used
                kernel_factory=MALAKernelFactory(config=MALAConfig(step_size=0.1)),
                num_warmup_steps=self.approximation_mcmc_num_warmup_steps,
                adapt_step_size=True,
                target_accept_rate=0.5,
                init_using_log_l_mode=False,
                num_chains=100,  # automatically adapted
            )
        else:
            return self.approximation_mcmc_config

    def _perform_test(
        self, X: T, u_stat: BaseUStat[UStatFn[T]], key: PRNGKeyArray
    ) -> TestResults:
        if self.approximate:
            assert is_approximable(u_stat)
            key, subkey = jax.random.split(key)
            mmd_val, func_evals = u_stat.compute_approx(
                X,
                self.approximation_num_particles,
                subkey,
                self.get_mcmc_config(),
            )
        else:
            mmd_val, func_evals = u_stat.compute(X)

        quantile_, h0_vals = self.compute_quantile(X, u_stat, func_evals, key)
        return TestResults(
            result=cast(bool, mmd_val > quantile_),
            val=mmd_val,
            h0_vals=h0_vals,
        )

    def unbiased_estimate(
        self, *args: Unpack[TestInput_T], key: jax.random.KeyArray
    ) -> Scalar:
        # Semi-public - undocumented, unadvertised.
        u_stat_fn = self._get_u_stat_fn(*args, key=key)
        u_stat = self._get_u_stat(u_stat_fn)
        one_sample = self._build_one_sample(*args)
        if self.approximate:
            assert is_approximable(u_stat)
            key, subkey = jax.random.split(key)
            return u_stat.compute_approx(
                one_sample,
                self.approximation_num_particles,
                subkey,
                mcmc_config=self.get_mcmc_config(),
            )[0]
        else:
            return u_stat.compute(one_sample)[0]


K_T = TypeVar("K_T", bound=base_kernel[Any])


class KernelBasedStruct(Generic[K_T], struct.PyTreeNode):
    kernel: K_T
