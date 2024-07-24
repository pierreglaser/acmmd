from typing import Any, Callable, Generic, Literal, Optional, Tuple, cast
from typing_extensions import Self, Type

import jax
import jax.numpy as jnp
import numpyro.distributions as np_distributions
from flax import struct
from jax import random, vmap
from jax.lax import scan  # type: ignore
from jax.tree_util import tree_flatten, tree_leaves, tree_map
from numpyro.infer.hmc_util import HMCAdaptState
from jax_samplers.distributions import DoublyIntractableLogDensity, ThetaConditionalLogDensity, maybe_wrap

from jax_samplers.pytypes import Array, LogDensity_T, Numeric, PRNGKeyArray, PyTreeNode
from jax_samplers.inference_algorithms.base import (
    InferenceAlgorithm, InferenceAlgorithmConfig, InferenceAlgorithmFactory, InferenceAlgorithmInfo,
    InferenceAlgorithmResults)
from jax_samplers.kernels.adaptive_mala import AdaptiveMALAState
from jax_samplers.kernels.hmc import HMCInfo, HMCKernel, HMCKernelFactory
from jax_samplers.kernels.mala import MALAConfig, MALAKernelFactory
from jax_samplers.kernels.nuts import NUTSInfo, NUTSKernelFactory
from jax_samplers.kernels.savm import SAVMState

from ...kernels.base import (Array_T, Config_T, Config_T_co, Info_T, Info_T,
                             Kernel, KernelFactory, MHKernelFactory, Result, State, State_T, TunableKernel, TunableMHKernelFactory)
from ...particle_aproximation import ParticleApproximation
from .util import progress_bar_factory

from tqdm.auto import tqdm as tqdm_auto
from jax.tree_util import tree_map
from jax._src.flatten_util import ravel_pytree


def tree_any(function: Callable[[PyTreeNode], Numeric], tree: PyTreeNode) -> Numeric:
    mapped_tree = tree_map(function, tree)
    return jnp.any(ravel_pytree(mapped_tree)[0])


def adam_initialize_doubly_intractable(theta: Array, target_log_prob_fn: DoublyIntractableLogDensity, key: PRNGKeyArray, num_steps=50, learning_rate=0.05, num_likelihood_sampler_steps: int =100):
    """Use Adam optimizer to get a reasonable initialization for HMC algorithms.

    Args:
      x: Where to initialize Adam.
      target_log_prob_fn: Unnormalized target log-density.
      num_steps: How many steps of Adam to run.
      learning_rate: What learning rate to pass to Adam.

    Returns:
      Optimized version of x.
    """
    import optax
    import jax

    init_mcmc_chain = _MCMCChain(_MCMCChainConfig(
        MALAKernelFactory(MALAConfig(1.0, None)), num_steps=num_likelihood_sampler_steps // 2, num_warmup_steps=num_likelihood_sampler_steps // 2, 
        adapt_mass_matrix=False, adapt_step_size=True, target_accept_rate=0.5, record_trajectory=True
    ), ThetaConditionalLogDensity(target_log_prob_fn.log_likelihood, theta), )
    init_mcmc_chain = init_mcmc_chain.init(target_log_prob_fn.x_obs)
    init_mcmc_chain, _ = init_mcmc_chain.run(key=random.fold_in(key, 0))

    def update_step(input_, i):
        theta, adam_state, mcmc_chain, lr  = input_

        _, g_log_prior = jax.tree_map(lambda x: -x, jax.value_and_grad(target_log_prob_fn.log_prior)(theta))
        _, g_log_lik_unnormalized = jax.tree_map(lambda x: -x, jax.value_and_grad(target_log_prob_fn.log_likelihood)(theta, target_log_prob_fn.x_obs))

        assert isinstance(mcmc_chain, _MCMCChain)
        assert isinstance(mcmc_chain.log_prob, ThetaConditionalLogDensity)
        mcmc_chain = cast(_MCMCChain, mcmc_chain.replace(log_prob=mcmc_chain.log_prob.replace(theta=theta)))
        new_mcmc_chain, results = mcmc_chain.run(key=random.fold_in(key, i))

        g_log_normalizer = target_log_prob_fn.log_likelihood(theta, results.final_state.x)
        g =  g_log_prior + g_log_lik_unnormalized + g_log_normalizer

        # updates, new_adam_state = optax.adam(0.001).update(g, adam_state)
        updates, new_adam_state = optax.adam(lr).update(g, adam_state)
        new_theta = optax.apply_updates(theta, updates)

        # has_nan = tree_any(lambda x: jnp.isfinite(x), (new_theta, target_log_prob_fn(new_theta), updates, new_adam_state, g, mcmc_chain))
        has_nan = tree_any(lambda x: jnp.isnan(x), (target_log_prob_fn(new_theta), new_theta))
        has_inf = tree_any(lambda x: jnp.isinf(x), (target_log_prob_fn(new_theta), new_theta))

        new_ret = jax.lax.cond(
            has_nan,
            lambda _: (theta, optax.adam(lr/1.5).init(theta), new_mcmc_chain, lr/1.5),
            lambda _: (new_theta, new_adam_state, new_mcmc_chain, lr),
            None
        )
        return new_ret, (new_ret, has_nan, has_inf)

    init_state = optax.adam(learning_rate).init(theta)
    # theta, _, _, final_lr, final_arr  = jax.lax.fori_loop(1, num_steps, update_step, (theta, init_state, init_mcmc_chain, learning_rate, arr))
    (theta, _, _, final_lr), traj   = jax.lax.scan(
        update_step,
        (theta, init_state, init_mcmc_chain, learning_rate),
        jnp.arange(1, num_steps+1),
    )
    # print(traj[-1])
    # print(final_lr)
    # print(final_arr)

    return theta


def adam_initialize(x: Array, target_log_prob_fn: LogDensity_T, num_steps=50, learning_rate=0.05):
    """Use Adam optimizer to get a reasonable initialization for HMC algorithms.

    Args:
      x: Where to initialize Adam.
      target_log_prob_fn: Unnormalized target log-density.
      num_steps: How many steps of Adam to run.
      learning_rate: What learning rate to pass to Adam.

    Returns:
      Optimized version of x.
    """
    import optax
    import jax
    optimizer = optax.adam(learning_rate)

    def update_step(i, input_):
        x, adam_state = input_
        def g_fn(x):
            return jax.tree_map(lambda x: -x, jax.value_and_grad(target_log_prob_fn)(x))
        _, g = g_fn(x)
        updates, adam_state = optimizer.update(g, adam_state)
        return optax.apply_updates(x, updates), adam_state

    init_state = optimizer.init(x)
    x, _ = jax.lax.fori_loop(1, num_steps, update_step, (x, init_state))
    return x


class _MCMCChainConfig(Generic[Config_T, State_T, Info_T], struct.PyTreeNode):
    kernel_factory: KernelFactory[Config_T, State_T, Info_T]
    num_steps: int = struct.field(pytree_node=False)
    record_trajectory: bool = struct.field(pytree_node=False)
    num_warmup_steps: int = struct.field(pytree_node=False)
    adapt_step_size: bool = struct.field(pytree_node=False)
    adapt_mass_matrix: bool = struct.field(pytree_node=False)
    target_accept_rate: float = 0.2
    warmup_method: Literal['numpyro', 'sbi_ebm'] = struct.field(pytree_node=False, default='sbi_ebm')
    init_using_log_l_mode: bool = struct.field(pytree_node=False, default=True)
    init_using_log_l_mode_num_opt_steps: int = struct.field(pytree_node=False, default=500)



class _SingleChainResults(Generic[State_T, Info_T], struct.PyTreeNode):
    final_state: State_T
    chain: State_T
    info: Info_T
    warmup_info: Optional[Info_T] = None


class _MCMCChain(Generic[Config_T, State_T, Info_T], struct.PyTreeNode):
    config: _MCMCChainConfig[Config_T, State_T, Info_T]
    log_prob: LogDensity_T
    _init_state: Optional[State_T] = None
    _chain_id: int = 0
    _p_bar_update_fn: Optional[Callable[[int,  int], int]] = struct.field(pytree_node=False, default=None)

    def init(self, x0: Array) -> Self:
        init_state = self.config.kernel_factory.build_kernel(self.log_prob).init_state(x0)
        return self.replace(_init_state=init_state)

    def _init_from_log_l_mode(self, key: PRNGKeyArray) -> Self:
        assert self._init_state is not None
        init_state = self._init_state
        if not isinstance(self.log_prob, DoublyIntractableLogDensity):
            print('finding good initial position')
            good_first_position = adam_initialize(init_state.x, self.log_prob)
            init_state = init_state.replace(x=good_first_position)
        else:
            print('finding good initial position (doubly intractable)')
            key, subkey = random.split(key)
            # print("initial position: ", init_state.x)
            good_first_position = adam_initialize_doubly_intractable(
                init_state.x, self.log_prob, subkey,
                learning_rate=0.05, num_steps=self.config.init_using_log_l_mode_num_opt_steps
            )
            print("good initial position found at: ", good_first_position)
            init_state = init_state.replace(x=good_first_position)

        return self.replace(_init_state=init_state)

    def run(self, key: PRNGKeyArray) -> Tuple[Self, _SingleChainResults[State_T, Info_T]]:
        key, subkey = random.split(key)
        if self.config.init_using_log_l_mode:
            self = self._init_from_log_l_mode(subkey)

        if self.config.num_warmup_steps > 0:
            key, subkey = random.split(key)
            self, warmup_info = self._warmup(subkey)
        else:
            warmup_info = None

        kernel = self.config.kernel_factory.build_kernel(self.log_prob)

        def step_fn(
            x: State_T, iter_no: int
        ) -> Tuple[State_T, Optional[Result[State_T, Info_T]]]:
            mala_result = kernel.one_step(x, random.fold_in(subkey, iter_no))
            self._maybe_update_pbar(iter_no, self._chain_id)
            if not self.config.record_trajectory:
                output = None
            else:
                output = mala_result
            return mala_result.state, output

        assert self._init_state is not None
        init_state = self._init_state

        key, subkey = random.split(key)
        final_state, outputs = scan(step_fn, init_state, xs=jnp.arange(self.config.num_warmup_steps, self.config.num_warmup_steps + self.config.num_steps))  # type: ignore
        if self.config.record_trajectory:
            assert outputs is not None
            stats, chain = outputs.info, outputs.state
        else:
            _smoke_res = kernel.one_step(init_state, key)
            stats = _smoke_res.info
            chain = tree_map(lambda x: x[None, ...], final_state)
        return self, _SingleChainResults(final_state, chain, stats, warmup_info)

    def _maybe_update_pbar(self, iter_no, _chain_id) -> int:
        if self._p_bar_update_fn is not None:
            return self._p_bar_update_fn(iter_no, _chain_id)
        else:
            return iter_no


    def _warmup(self, key: PRNGKeyArray) -> Tuple[Self, Info_T]:
        if self.config.warmup_method == "sbi_ebm":
            return self._warmup_sbi_ebm(key)
        elif self.config.warmup_method == "numpyro":
            return self._warmup_numpyro(key)
        else:
            raise ValueError(f"Unknown warmup method {self.config.warmup_method}")

    def _warmup_sbi_ebm(self, key: PRNGKeyArray) -> Tuple[Self, Info_T]:
        if self.config.adapt_mass_matrix or self.config.adapt_step_size:
            assert isinstance(self.config.kernel_factory, TunableMHKernelFactory)
        kernel = self.config.kernel_factory.build_kernel(self.log_prob)

        # record_trajectory = self.config.record_trajectory
        record_trajectory = False

        def step_fn(carry: Tuple[State_T, AdaptiveMALAState], iter_no: int) -> Tuple[Tuple[State_T, AdaptiveMALAState], Optional[Result[State_T, Info_T]]]:
            this_kernel = kernel
            x, adaptation_state = carry
            if self.config.adapt_step_size:
                assert isinstance(this_kernel, TunableKernel)
                this_kernel = this_kernel.set_step_size(adaptation_state.sigma)

            if self.config.adapt_mass_matrix:
                assert isinstance(this_kernel, TunableKernel)
                this_kernel = this_kernel.set_inverse_mass_matrix(adaptation_state.C)

            mala_result = this_kernel.one_step(x, random.fold_in(subkey, iter_no))

            next_adaptation_state = adaptation_state

            if self.config.adapt_step_size:
                next_adaptation_state = next_adaptation_state.update_sigma(log_alpha=getattr(mala_result.info, "log_alpha", 0), gamma_n=1/(next_adaptation_state.iter_no+1)**0.5)
            if self.config.adapt_mass_matrix:
                next_adaptation_state = next_adaptation_state.update_cov(x=mala_result.state.x, gamma_n=1/(next_adaptation_state.iter_no+1)**0.5)

            next_adaptation_state = next_adaptation_state.replace(iter_no=next_adaptation_state.iter_no + 1, x=mala_result.state.x)

            if not record_trajectory:
                output = None
            else:
                output = mala_result

            self._maybe_update_pbar(iter_no, self._chain_id)
            return (mala_result.state, next_adaptation_state), output

        assert self._init_state is not None
        init_state = self._init_state

        target_accept_rate = self.config.target_accept_rate
        if self.config.adapt_mass_matrix:
            assert isinstance(kernel, TunableKernel)
            uses_diagonal_mass_matrix = len(kernel.get_inverse_mass_matrix().shape) == 1
            if uses_diagonal_mass_matrix:
                init_adaptation_state = AdaptiveMALAState(init_state.x, 1, init_state.x, jnp.zeros((init_state.x.shape[0],)), 1., target_accept_rate)
            else:
                init_adaptation_state = AdaptiveMALAState(init_state.x, 1, init_state.x, jnp.zeros((init_state.x.shape[0], init_state.x.shape[0])), 1., target_accept_rate)
        else:
            init_adaptation_state = AdaptiveMALAState(init_state.x, 1, None, None, 1., target_accept_rate)

        key, subkey = random.split(key)
        final_state, outputs = scan(step_fn, (init_state, init_adaptation_state), xs=jnp.arange(1, self.config.num_warmup_steps+1))  # type: ignore
        if record_trajectory:
            assert outputs is not None
            stats, chain = outputs.info, outputs.state
        else:
            _smoke_res = kernel.one_step(init_state, key)
            stats = _smoke_res.info
            chain = tree_map(lambda x: x[None, ...], final_state)


        # new_init_state = self.config.kernel_factory.build_kernel(self.log_prob).init_state(final_state[0].x)
        new_init_state = final_state[0]


        final_kernel = kernel

        if self.config.adapt_step_size:
            assert isinstance(final_kernel, TunableKernel)
            final_kernel = final_kernel.set_step_size(step_size=final_state[1].sigma)
        if self.config.adapt_mass_matrix:
            assert isinstance(final_kernel, TunableKernel)
            final_kernel = final_kernel.set_inverse_mass_matrix(final_state[1].C)

        return self.replace(
            config=self.config.replace(kernel_factory=self.config.kernel_factory.replace(config=final_kernel.config)),
            _init_state=new_init_state
        ), stats

    def _warmup_numpyro(self, key: PRNGKeyArray) -> Tuple[Self, Info_T]:
        kernel = self.config.kernel_factory.build_kernel(self.log_prob)
        assert isinstance(kernel, TunableKernel)

        if self.config.adapt_mass_matrix:
            init_mass_matrix = kernel.get_inverse_mass_matrix()
        else:
            assert self._init_state is not None
            init_mass_matrix = jnp.ones((self._init_state.x.shape[0],))


        from numpyro.infer.hmc_util import warmup_adapter
        wa_init, _wa_update = warmup_adapter(
            self.config.num_warmup_steps,
            adapt_step_size=self.config.adapt_step_size,
            adapt_mass_matrix=self.config.adapt_mass_matrix,
            dense_mass=init_mass_matrix is not None and len(init_mass_matrix.shape) == 2,
            target_accept_prob=self.config.target_accept_rate,
        )

        assert self._init_state is not None
        init_state = self._init_state

        key, subkey = random.split(key)
        init_adaptation_state = wa_init(
            (init_state.x,), subkey, kernel.get_step_size(), mass_matrix_size=init_state.x.shape[0]
        )
        init_adaptation_state = init_adaptation_state._replace(rng_key=None)
        record_trajectory = False
        def step_fn(carry: Tuple[State_T, HMCAdaptState], iter_no: int) -> Tuple[Tuple[State_T, HMCAdaptState], Optional[Result[State_T, Info_T]]]:
            this_kernel = kernel
            x, adaptation_state = carry
            if self.config.adapt_step_size:
                assert isinstance(this_kernel, TunableKernel)
                this_kernel = this_kernel.set_step_size(adaptation_state.step_size)

            if self.config.adapt_mass_matrix:
                assert isinstance(this_kernel, TunableKernel)
                this_kernel = this_kernel.set_inverse_mass_matrix(adaptation_state.inverse_mass_matrix)

            mala_result = this_kernel.one_step(x, random.fold_in(subkey, iter_no))

            next_adaptation_state = _wa_update(
                iter_no, jnp.exp(jnp.clip(mala_result.info.log_alpha, a_max=0)), (mala_result.state.x,), adaptation_state
            )

            if not record_trajectory:
                output = None
            else:
                output = mala_result

            self._maybe_update_pbar(iter_no, self._chain_id)
            return (mala_result.state, next_adaptation_state), output

        final_state, outputs = scan(step_fn, (init_state, init_adaptation_state), xs=jnp.arange(1, self.config.num_warmup_steps+1))  # type: ignore
        final_kernel = kernel
        new_init_state = final_state[0]
        if self.config.adapt_step_size:
            assert isinstance(final_kernel, TunableKernel)
            final_kernel = final_kernel.set_step_size(step_size=final_state[1].step_size)
        if self.config.adapt_mass_matrix:
            assert isinstance(final_kernel, TunableKernel)
            final_kernel = final_kernel.set_inverse_mass_matrix(final_state[1].inverse_mass_matrix)

        if record_trajectory:
            assert outputs is not None
            stats, chain = outputs.info, outputs.state
        else:
            _smoke_res = kernel.one_step(init_state, key)
            stats = _smoke_res.info
            chain = tree_map(lambda x: x[None, ...], final_state)

        return self.replace(
            config=self.config.replace(kernel_factory=self.config.kernel_factory.replace(config=final_kernel.config)),
            _init_state=new_init_state
        ), stats


class MCMCConfig(
    Generic[Config_T, State_T, Info_T], InferenceAlgorithmConfig, struct.PyTreeNode
):
    kernel_factory: KernelFactory[Config_T, State_T, Info_T]
    num_samples: int = struct.field(pytree_node=False)
    num_chains: int = struct.field(pytree_node=False, default=100)
    thinning_factor: int = struct.field(pytree_node=False, default=10)
    record_trajectory: bool = struct.field(pytree_node=False, default=True)
    num_warmup_steps: int = struct.field(pytree_node=False, default=0)
    adapt_step_size: bool = struct.field(pytree_node=False, default=False)
    adapt_mass_matrix: bool = struct.field(pytree_node=False, default=False)
    resample_stuck_chain_at_warmup: bool = struct.field(pytree_node=False, default=False)
    target_accept_rate: float = struct.field(pytree_node=False, default=0.2)
    progress_bar: bool = struct.field(pytree_node=False, default=False)
    warmup_method: Literal["numpyro", "sbi_ebm"] = struct.field(pytree_node=False, default="sbi_ebm")
    init_using_log_l_mode: bool = struct.field(pytree_node=False, default=True)
    init_using_log_l_mode_num_opt_steps: int = struct.field(pytree_node=False, default=50)


class MCMCInfo(Generic[State_T, Info_T], InferenceAlgorithmInfo):
    single_chain_results: _SingleChainResults[State_T, Info_T]


class MCMCResults(Generic[State_T, Info_T], InferenceAlgorithmResults):
    samples: ParticleApproximation
    info: MCMCInfo[State_T, Info_T]


class MCMCAlgorithm(
    InferenceAlgorithm[MCMCConfig[Config_T, State_T, Info_T]]
):
    _single_chains: Optional[_MCMCChain[Config_T, State_T, Info_T]] = None

    @property
    def _uninitialized_chain_vmap_axes(self) -> _MCMCChain:
        from jax.tree_util import tree_map
        assert self._single_chains is not None
        return cast(_MCMCChain, tree_map(lambda x: 0, self._single_chains)).replace(log_prob=None, _init_state=None)

    @property
    def _initialized_chain_vmap_axes(self) -> _MCMCChain:
        from jax.tree_util import tree_map
        assert self._single_chains is not None
        return cast(_MCMCChain, tree_map(lambda x: 0, self._single_chains)).replace(log_prob=None, _init_state=0)

    @classmethod
    def create(cls, config: MCMCConfig[Config_T, State_T, Info_T], log_prob: LogDensity_T) -> Self:
        # build single chain MCMC configs
        num_total_steps = (config.num_samples * config.thinning_factor) / config.num_chains
        assert num_total_steps == int(num_total_steps)
        _single_chain_configs = vmap(lambda _: _MCMCChainConfig(
            config.kernel_factory, int(num_total_steps), True, config.num_warmup_steps, config.adapt_step_size, config.adapt_mass_matrix, config.target_accept_rate, warmup_method=config.warmup_method,
            init_using_log_l_mode=config.init_using_log_l_mode, init_using_log_l_mode_num_opt_steps=config.init_using_log_l_mode_num_opt_steps
        ))(jnp.arange(config.num_chains))
        _single_chains = vmap(_MCMCChain, in_axes=(0, None, None, 0), out_axes=_MCMCChain(0, None, None, 0))(_single_chain_configs, log_prob, None, jnp.arange(config.num_chains))  # type: ignore
        return cls(config, log_prob, _init_state=None, _single_chains=_single_chains)

    def init(self, key: PRNGKeyArray, dist: np_distributions.Distribution, reweight_and_resample: bool = False) -> Self:
        xs = dist.sample(key, (self.config.num_chains,))
        init_state = ParticleApproximation(xs, jnp.zeros(self.config.num_chains))
        if reweight_and_resample:
            key, subkey = random.split(key)
            log_ratio = vmap(self.log_prob)(init_state.xs) - vmap(dist.log_prob)(init_state.xs)
            init_state = init_state.replace(log_ws=log_ratio).resample_and_reset_weights(subkey)

        # single_chains = vmap(lambda c, x0: cast(_MCMCChain[Config_T, State_T, Info_T], c).init(x0))(self._single_chains, init_state.particles)
        single_chains = vmap(lambda c, x0: cast(_MCMCChain[Config_T, State_T, Info_T], c).init(x0), in_axes=(_MCMCChain(0, None, None, 0), 0), out_axes=_MCMCChain(0, None, 0, 0))(self._single_chains, init_state.particles)  # type: ignore

        return self.replace(_init_state=init_state, _single_chains=single_chains)

    def init_from_particles(self, xs: Array) -> Self:
        assert len(xs.shape) == 2
        assert len(xs) == self.config.num_chains
        init_state = ParticleApproximation(xs, jnp.zeros(self.config.num_samples))

        # single_chains = vmap(lambda c, x0: cast(_MCMCChain[Config_T, State_T, Info_T], c).init(x0))(self._single_chains, init_state.particles)
        single_chains = vmap(lambda c, x0: cast(_MCMCChain[Config_T, State_T, Info_T], c).init(x0), in_axes=(_MCMCChain(0, None, None, 0), 0), out_axes=_MCMCChain(0, None, 0, 0))(self._single_chains, init_state.particles)  # type: ignore

        return self.replace(_init_state=init_state, _single_chains=single_chains)

    def set_log_prob(self, log_prob: LogDensity_T) -> Self:
        self = self.replace(log_prob=log_prob)
        if self._single_chains is not None:
            self = self.replace(_single_chains=self._single_chains.replace(log_prob=log_prob))
        return self

    def set_num_warmup_steps(self, num_warmup_steps) -> Self:
        self = cast(Self, self.replace(config=self.config.replace(num_warmup_steps=num_warmup_steps)))
        if self._single_chains is not None:
            self = self.replace(_single_chains=self._single_chains.config.replace(num_warmup_steps=num_warmup_steps))
        return self

    def _maybe_set_progress_bar(self) -> Self:
        assert self._single_chains is not None
        if self.config.progress_bar:
            pbar = tqdm_auto(range((self._single_chains.config.num_steps + self._single_chains.config.num_warmup_steps) * self.config.num_chains), miniters=100, mininterval=100)
            pbar.set_description("Compiling.. ", refresh=True)

            new_single_chains = self._single_chains.replace(_p_bar_update_fn=progress_bar_factory(pbar, self._single_chains.config.num_steps + self._single_chains.config.num_warmup_steps))
            return self.replace(_single_chains=new_single_chains)
        else:
            return self

    def _maybe_remove_progress_bar(self) -> Self:
        assert self._single_chains is not None
        return self.replace(_single_chains=self._single_chains.replace(_p_bar_update_fn=None))

    def _aggregate_single_chain_results(self, single_chain_results: _SingleChainResults[State_T, Info_T]) -> Array_T:
        final_samples = single_chain_results.chain.x[:, ::-self.config.thinning_factor, :].reshape(-1, single_chain_results.chain.x.shape[-1])
        assert len(final_samples) == self.config.num_samples
        return final_samples

    def run(self, key: PRNGKeyArray) -> Tuple[Self, MCMCResults[State_T, Info_T]]:
        self = self._maybe_set_progress_bar()
        assert self._single_chains is not None

        key, subkey = random.split(key)
        new_single_chains, single_chain_results = vmap(
            lambda c, k: cast(_MCMCChain[Config_T, State_T, Info_T], c).run(k),
            # in_axes=(0, 0), out_axes=(0, 0)
            in_axes=(_MCMCChain(0, None, 0, 0, self._single_chains._p_bar_update_fn), 0), out_axes=(_MCMCChain(0, None, 0, 0, self._single_chains._p_bar_update_fn), 0)  # type: ignore
        )(self._single_chains, random.split(subkey, self.config.num_chains))

        final_samples = self._aggregate_single_chain_results(single_chain_results)

        self = self.replace(_single_chains=new_single_chains)._maybe_remove_progress_bar()
        return self, MCMCResults(
                ParticleApproximation(final_samples, jnp.zeros((final_samples.shape[0],))),
                info=MCMCInfo(single_chain_results)
            )

class MCMCAlgorithmFactory(InferenceAlgorithmFactory[MCMCConfig[Config_T, State_T, Info_T]]):
    def build_algorithm(self, log_prob: LogDensity_T) -> MCMCAlgorithm[Config_T, State_T, Info_T]:
        return MCMCAlgorithm.create(log_prob=log_prob, config=self.config)
