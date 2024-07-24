from typing import Final, Generic, Optional, Protocol, Tuple, Type, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as np_distributions
from flax import struct
from jax import random, vmap
from jax.lax import cond, scan  # type: ignore
from jax.nn import logsumexp
from jax.tree_util import tree_map
from numpyro.distributions import Distribution as NPDist_Inst
from typing_extensions import Self

from jax_samplers.distributions import maybe_wrap
from jax_samplers.pytypes import Array, LogDensity_T, Numeric, PRNGKeyArray
from jax_samplers.inference_algorithms.base import (
    InferenceAlgorithm, InferenceAlgorithmConfig, InferenceAlgorithmFactory, InferenceAlgorithmInfo,
    InferenceAlgorithmResults)
from jax_samplers.kernels.base import (Array_T, Config_T, Info, Info_T,
                                           KernelFactory, State_T, TunableKernel, TunableMHKernelFactory)
from jax_samplers.kernels.ula import ULAConfig, ULAKernel
from jax_samplers.particle_aproximation import ParticleApproximation


class AnnealedLogProb(struct.PyTreeNode):
    init_log_prob: LogDensity_T = struct.field(pytree_node=True)
    final_log_prob: LogDensity_T = struct.field(pytree_node=True)
    beta: Numeric = struct.field(pytree_node=True)

    def __call__(self, x: Array) -> Numeric:
        return (1 - self.beta) * self.init_log_prob(x) + (
            self.beta
        ) * self.final_log_prob(x)


class AnnealingSchedule(struct.PyTreeNode):
    init_log_prob: LogDensity_T
    final_log_prob: LogDensity_T
    betas: Array = struct.field(pytree_node=True)

    @property
    def num_steps(self) -> Numeric:
        return self.betas.shape[0]

    @classmethod
    def create(
        cls: Type["AnnealingSchedule"],
        init_log_prob: LogDensity_T,
        final_log_prob: LogDensity_T,
        num_steps: int,
        betas: Optional[Array] = None,
    ):
        if betas is None:
            # betas = jnp.linspace(0, 1, num_steps + 1) ** 2  # type: ignore
            betas = jnp.linspace(0, 1, num_steps + 1)  # type: ignore

        return cls(
            init_log_prob=init_log_prob,
            final_log_prob=final_log_prob,
            betas=betas,
        )

    def get_dist(self, i: Numeric) -> AnnealedLogProb:
        return AnnealedLogProb(
            init_log_prob=self.init_log_prob,
            final_log_prob=self.final_log_prob,
            beta=self.betas[i],
        )


class SMCParticleApproximation(ParticleApproximation):
    _Z_log_space: Numeric
    log_Z: Numeric
    log_prob: LogDensity_T

    @classmethod
    def from_npdistribution(
        cls: Type[Self],
        dist: NPDist_Inst,
        num_samples: int,
        key: PRNGKeyArray,
    ) -> Self:
        particle_approx = ParticleApproximation.from_npdistribution(
            dist, num_samples, key
        )
        return cls(
            particles=particle_approx.particles,
            log_ws=particle_approx.log_ws,
            _Z_log_space=0.0,
            log_Z=0.0,
            # log_prob=maybe_wrap(lambda x: dist.log_prob(x)),
            log_prob=maybe_wrap(dist.log_prob),
        )

    def reweight_and_update_Z(self: Self, new_log_prob: LogDensity_T) -> Self:
        new_log_prob_vals = vmap(new_log_prob)(self.xs)
        prev_log_prob_vals = vmap(self.log_prob)(self.xs)
        # Due to numerical rounding approximations, unconstrained distributions
        # originating from distributions with bounded support can evaluate
        # (unconstrained) points that land exactly on the support boundary once
        # they are transformed back to the original bounded space using the
        # unconstrainining bijector's inverse, leading to (negative) infinite
        # log prob values, propagating nans further down the road.
        # As a hot fix, I currently manually replace possible -jnp.inf values
        # during probability evaluation operations. It would be better to
        # actually smooth very slightly the problematic unconstrained
        # distributions - which are ofen prior and not EBMs, which do not have
        # bounded support usually.
        # new_log_prob_vals = jnp.nan_to_num(new_log_prob_vals, neginf=-1e10)
        # prev_log_prob_vals = jnp.nan_to_num(prev_log_prob_vals, neginf=-1e6)

        log_ratio = new_log_prob_vals - prev_log_prob_vals
        log_ratio = jnp.nan_to_num(log_ratio, neginf=-1e80, posinf=-1e80, nan=-1e80)

        log_weights_unnormalized = self.log_ws + log_ratio
        log_weights = jax.nn.log_softmax(log_weights_unnormalized)

        # jit-compiled function can sometimes return nans
        # log_weights = jnp.nan_to_num(log_weights, -1e4)

        # NORMALIZING CONSTANT RECOMPUTATION STEP
        _Z_log_space = self._Z_log_space + logsumexp(log_weights_unnormalized)

        prev_log_prob = self.log_prob
        new_samples = self.replace(
            log_prob=new_log_prob, log_ws=log_weights, _Z_log_space=_Z_log_space
        )

        # LOG-NORMALIZING CONSTANT RECOMPUTATION STEP
        assert isinstance(new_log_prob, AnnealedLogProb)
        assert isinstance(prev_log_prob, AnnealedLogProb)

        # Log Z1 / Log Z0 = ∫₀¹ - P_ﬦ(V) dﬦ
        V = lambda x: (jnp.nan_to_num(new_log_prob.final_log_prob(x) - new_log_prob.init_log_prob(x), neginf=-1e10))

        average_potential_new = new_samples.average_of(V)
        average_potential_old = self.average_of(V)
        log_Z_inc = (new_log_prob.beta - prev_log_prob.beta) * 0.5 * (average_potential_new + average_potential_old)

        new_samples = new_samples.replace(log_Z=new_samples.log_Z + log_Z_inc)
        return new_samples

    @property
    def Z(self):
        # Unstable exponentiation
        return jnp.exp(self._Z_log_space)

    def normalized_log_prob(self, x: Array) -> Numeric:
        return self.log_prob(x) - self.log_Z


class AnnealingSchedule_P(Protocol):
    def get_dist(self, i: Numeric) -> AnnealedLogProb:
        raise NotADirectoryError


class SMCConfig(Generic[Config_T, State_T, Info_T], InferenceAlgorithmConfig):
    inner_kernel_factory: KernelFactory[Config_T, State_T, Info_T] = struct.field(
        pytree_node=True
    )
    inner_kernel_steps: int = struct.field(pytree_node=False)
    num_steps: int = struct.field(pytree_node=False, default=20)
    ess_threshold: float = struct.field(pytree_node=False, default=0.8)
    use_random_path: bool = struct.field(pytree_node=False, default=False)
    schedule: Optional[AnnealingSchedule_P] = struct.field(
        pytree_node=True, default=None
    )
    record_trajectory: bool = struct.field(pytree_node=False, default=True)


class SMCStepState(Generic[State_T], struct.PyTreeNode):
    particle_approximation: SMCParticleApproximation
    kernel_state: State_T


class SMCStepInfo(Generic[Info_T], Info):
    log_ess: Numeric
    inner_sampler_stats: Info_T
    did_resample: bool

    @property
    def ess(self):
        # (Unstable exponentiation)
        return jnp.exp(self.log_ess)


def _get_schedule(
    init_log_prob: LogDensity_T,
    final_log_prob: LogDensity_T,
    config: SMCConfig[Config_T, State_T, Info_T],
    key: PRNGKeyArray,
) -> AnnealingSchedule_P:
    if config.schedule is not None:
        return config.schedule

    if config.use_random_path:
        betas = jnp.sort(
            jnp.concatenate(
                [
                    jnp.array([0]),
                    random.uniform(key, (config.num_steps - 1,)),
                    jnp.array([1]),
                ]
            )
        )
    else:
        betas = None

    schedule = AnnealingSchedule.create(
        init_log_prob=init_log_prob,
        final_log_prob=final_log_prob,
        num_steps=config.num_steps,
        betas=betas,
    )
    return schedule


def _maybe_resample(
    x: SMCParticleApproximation,
    ess_threshold: Numeric,
    key: PRNGKeyArray,
    mcmc_state: State_T,
) -> Tuple[SMCParticleApproximation, State_T, bool]:

    def _resample() -> Tuple[SMCParticleApproximation, State_T]:
        import numpy as np
        import numpyro.distributions as np_distributions

        mn = np_distributions.Categorical(probs=x.normalized_ws)
        indices = mn.sample(key, (x.num_samples,))

        log_ws = -np.log(x.num_samples) * jnp.ones_like(x.log_ws)

        new_particles = jnp.take(x.particles, indices, axis=0)

        new_state: State_T = tree_map(
            lambda x: jnp.take(x, indices, axis=0), mcmc_state
        )
        return x.replace(log_ws=log_ws, particles=new_particles), new_state

    def _passthrough() -> Tuple[SMCParticleApproximation, State_T]:
        return x, mcmc_state

    # RESAMPLING STEP
    # skip resampling step as long as ess is above a certain threshold.
    log_ess = x.log_effective_sample_size()

    should_resample = log_ess < np.log(ess_threshold * x.num_samples)

    new_x, new_state = cond(should_resample, _resample, _passthrough)
    return new_x, new_state, should_resample


class SMCInfo(Generic[State_T, Info_T], InferenceAlgorithmInfo):
    smc_steps_states: SMCStepState[State_T]
    smc_steps_infos: SMCStepInfo[Info_T]


class SMCResults(Generic[State_T, Info_T], InferenceAlgorithmResults):
    samples: SMCParticleApproximation
    info: SMCInfo[State_T, Info_T]


T = TypeVar("T", bound=int, covariant=True)


def smc_sampler(
    unnormalized_log_prob: LogDensity_T,
    x0: SMCParticleApproximation,
    config: SMCConfig[Config_T, State_T, Info_T],
    key: PRNGKeyArray,
    init_state: Optional[SMCStepState[State_T]] = None,
) -> SMCResults[State_T, Info_T]:
    key, subkey = random.split(key)
    schedule = _get_schedule(x0.log_prob, unnormalized_log_prob, config, subkey)

    log_p0 = schedule.get_dist(i=0)
    x0 = x0.replace(log_prob=log_p0)

    CarryType = tuple[SMCStepState[State_T], PRNGKeyArray]
    OutputType = tuple[SMCStepState[State_T], SMCStepInfo[Info_T]]

    def smc_step(
        carry: CarryType, step_number: Array
    ) -> tuple[CarryType, Optional[OutputType]]:
        samples_i_minus_one, mcmc_state, key = (
            carry[0].particle_approximation,
            carry[0].kernel_state,
            carry[1],
        )

        log_p_i = schedule.get_dist(step_number + 1)

        # weights recomputation step
        samples_i = samples_i_minus_one.reweight_and_update_Z(log_p_i)

        # resampling step
        key, key_resampling = random.split(key)
        (
            samples_i_post_resampling,
            mcmc_state_post_resampling,
            did_resample,
        ) = _maybe_resample(samples_i, config.ess_threshold, key_resampling, mcmc_state)

        # mcmc step
        inner_kernel = config.inner_kernel_factory.build_kernel(log_p_i)
        mcmc_result = vmap(inner_kernel.n_steps, in_axes=(0, None, 0))(
            mcmc_state_post_resampling, config.inner_kernel_steps, random.split(key, num=samples_i.num_samples)
        )

        samples_i_post_mcmc = samples_i_post_resampling.replace(
            particles=mcmc_result.state.x
        )


        smc_stats = SMCStepInfo(
            log_ess=samples_i.log_effective_sample_size(),
            inner_sampler_stats=mcmc_result.info,
            did_resample=did_resample,
        )

        if not config.record_trajectory:
            return (SMCStepState(samples_i_post_mcmc, mcmc_result.state), key), None
        else:
            return (SMCStepState(samples_i_post_mcmc, mcmc_result.state), key), (
                SMCStepState(samples_i_post_mcmc, mcmc_result.state),
                smc_stats,
            )

    if init_state is None:
        init_kernel = config.inner_kernel_factory.build_kernel(log_p0)
        init_state = SMCStepState(x0, vmap(init_kernel.init_state)(x0.particles))
    else:
        init_state = init_state.replace(particle_approximation=init_state.particle_approximation.replace(log_prob=log_p0))


    final_carry, outputs = scan(
        smc_step, (init_state, key), jnp.arange(config.num_steps)
    )

    x_final = final_carry[0].particle_approximation
    x_final = x_final.replace(log_prob=unnormalized_log_prob)

    if config.record_trajectory:
        assert isinstance(outputs, tuple)
        all_xs, all_stats = outputs[0].particle_approximation, outputs[1]
        # XXX: this step is necessary to obtain a correct log_Z estimate,
        # consequently, if config.record_trajectory is False, the log_Z estimate
        # won't be correct. TODO: fix this limitation

        # postfix log_Z estimate
        x_final = x_final.replace(
            log_Z=x0.log_Z + (x_final.log_Z - x0.log_Z) / config.num_steps
        )

        all_xs = all_xs.replace(
            log_Z=x0.log_Z + (all_xs.log_Z - x0.log_Z) / config.num_steps
        )

        all_states = outputs[0]
    else:
        # make this branch vmap-compatible.
        all_xs = tree_map(lambda x: x[None, ...], x_final)
        kernel = config.inner_kernel_factory.build_kernel(log_p0)
        smoke_result = vmap(type(kernel).one_step, in_axes=(None, 0, None))(kernel, init_state.kernel_state, key)
        all_stats = SMCStepInfo(
            log_ess=jnp.array([-1.0]),
            inner_sampler_stats=smoke_result.info,
            did_resample=jnp.array([False]),  # type: ignore
        )
        all_states = final_carry[0]

    return SMCResults(x_final, SMCInfo(all_states, all_stats))


def _resample_step_sizes(step_sizes: Array, step_sizes_idxs: Array, init_particles: State_T, final_particles: State_T, key: PRNGKeyArray) -> Tuple[Array, Array]:
    # weights = jnp.sum(jnp.square(init_particles.x - final_particles.x), axis=-1)
    # weights /= jnp.sum(weights)
    dists = jnp.sum(jnp.square(init_particles.x - final_particles.x), axis=-1)
    dists_per_idx = vmap(lambda idx: jnp.sum(dists * (step_sizes_idxs == idx)) / jnp.sum(step_sizes_idxs == idx))(jnp.arange(len(step_sizes)))
    weights = dists_per_idx / jnp.sum(dists_per_idx)

    from numpyro import distributions as np_distributions

    _eps = 0.5
    indiv_dists = np_distributions.Uniform(low=step_sizes * (1 - _eps), high=step_sizes * (1 + _eps))
    mixing_dist = np_distributions.Categorical(probs=weights)

    step_size_dist = np_distributions.MixtureSameFamily(mixing_distribution=mixing_dist, component_distribution=indiv_dists)
    key, subkey = random.split(key)
    # return step_size_dist.sample(sample_shape=(init_particles.x.shape[0],), key=subkey)

    new_step_sizes = step_size_dist.sample(sample_shape=(len(step_sizes),), key=subkey)

    key, subkey = random.split(key)
    new_step_size_idxs = random.randint(minval=0, maxval=len(new_step_sizes), shape=(init_particles.x.shape[0],), key=subkey)
    return new_step_sizes, new_step_size_idxs



class AdaptiveSMCStepState(SMCStepState[State_T], struct.PyTreeNode):
    step_sizes: Array_T
    step_sizes_idxs: Array_T


class AdaptiveSMCStepInfo(SMCStepInfo[Info_T], struct.PyTreeNode):
    dists: Array_T


class AdaptiveSMCConfig(SMCConfig[Config_T, State_T, Info_T], InferenceAlgorithmConfig):
    num_step_sizes: int = struct.field(pytree_node=False, default=10)


class AdaptiveSMCInfo(SMCInfo[State_T, Info_T], struct.PyTreeNode):
    smc_steps_states: AdaptiveSMCStepState[State_T]
    smc_steps_infos: AdaptiveSMCStepInfo[Info_T]


class AdaptiveSMCResults(SMCResults[State_T, Info_T], struct.PyTreeNode):
    samples: SMCParticleApproximation
    info: AdaptiveSMCInfo[State_T, Info_T]

def smc_sampler_adaptive(
    unnormalized_log_prob: LogDensity_T,
    x0: SMCParticleApproximation,
    config: AdaptiveSMCConfig[Config_T, State_T, Info_T],
    key: PRNGKeyArray,
    init_state: Optional[AdaptiveSMCStepState[State_T]] = None,
) -> AdaptiveSMCResults[State_T, Info_T]:

    key, subkey = random.split(key)
    schedule = _get_schedule(x0.log_prob, unnormalized_log_prob, config, subkey)

    log_p0 = schedule.get_dist(i=0)
    x0 = x0.replace(log_prob=log_p0)

    assert isinstance(config.inner_kernel_factory, TunableMHKernelFactory)
    kernel = config.inner_kernel_factory.build_kernel(log_p0)
    if jnp.atleast_1d(kernel.get_step_size()).shape[0] == 1:
        step_sizes = jnp.repeat(kernel.get_step_size(), config.num_step_sizes)

        key, subkey = random.split(key)
        step_sizes_idxs = random.randint(minval=0, maxval=config.num_step_sizes, key=subkey, shape=(x0.num_samples,))
    else:
        step_sizes = kernel.get_step_size()
        raise ValueError


    assert len(jnp.atleast_1d(step_sizes).shape) == 1
    assert jnp.atleast_1d(step_sizes).shape[0] == config.num_step_sizes

    CarryType = tuple[AdaptiveSMCStepState[State_T], PRNGKeyArray]
    OutputType = tuple[AdaptiveSMCStepState[State_T], AdaptiveSMCStepInfo[Info_T]]

    def smc_step(
        carry: CarryType, step_number: Array
    ) -> tuple[CarryType, Optional[OutputType]]:
        samples_i_minus_one, mcmc_state, key = (
            carry[0].particle_approximation,
            carry[0].kernel_state,
            carry[1],
        )


        log_p_i = schedule.get_dist(step_number + 1)

        # weights recomputation step
        samples_i = samples_i_minus_one.reweight_and_update_Z(log_p_i)

        # resampling step
        key, key_resampling = random.split(key)
        (
            samples_i_post_resampling,
            mcmc_state_post_resampling,
            did_resample,
        ) = _maybe_resample(samples_i, config.ess_threshold, key_resampling, mcmc_state)

        # mcmc step
        assert isinstance(config.inner_kernel_factory, TunableMHKernelFactory)
        inner_kernel = config.inner_kernel_factory.build_kernel(log_p_i)

        key, subkey = random.split(key)
        mcmc_result = vmap(lambda ss, state, k: inner_kernel.set_step_size(ss).n_steps(state, config.inner_kernel_steps, k), in_axes=(0, 0, 0))(
            carry[0].step_sizes[carry[0].step_sizes_idxs], mcmc_state_post_resampling, random.split(subkey, num=samples_i.num_samples)
        )

        key, subkey = random.split(key)
        new_step_sizes, new_step_sizes_idxs = _resample_step_sizes(carry[0].step_sizes, carry[0].step_sizes_idxs, mcmc_state_post_resampling, mcmc_result.state, subkey)

        samples_i_post_mcmc = samples_i_post_resampling.replace(
            particles=mcmc_result.state.x
        )

        smc_stats = AdaptiveSMCStepInfo(
            log_ess=samples_i.log_effective_sample_size(),
            inner_sampler_stats=mcmc_result.info,
            did_resample=did_resample,
            dists=jnp.sum(jnp.square(mcmc_state_post_resampling.x - mcmc_result.state.x), axis=-1)

        )

        if not config.record_trajectory:
            return (AdaptiveSMCStepState(samples_i_post_mcmc, mcmc_result.state, new_step_sizes, new_step_sizes_idxs), key), None
        else:
            return (AdaptiveSMCStepState(samples_i_post_mcmc, mcmc_result.state, new_step_sizes, new_step_sizes_idxs), key), (
                AdaptiveSMCStepState(samples_i_post_mcmc, mcmc_result.state, new_step_sizes, new_step_sizes_idxs),
                smc_stats,
            )

    if init_state is None:
        init_kernel = config.inner_kernel_factory.build_kernel(log_p0)
        init_state = AdaptiveSMCStepState(x0, vmap(init_kernel.init_state)(x0.particles), step_sizes, step_sizes_idxs)
    else:
        init_state = init_state.replace(particle_approximation=init_state.particle_approximation.replace(log_prob=log_p0))
    final_carry, outputs = scan(
        smc_step, (init_state, key), jnp.arange(config.num_steps)
    )

    x_final = final_carry[0].particle_approximation
    x_final = x_final.replace(log_prob=unnormalized_log_prob)

    if config.record_trajectory:
        assert isinstance(outputs, tuple)
        all_xs, all_stats = outputs[0].particle_approximation, outputs[1]
        # XXX: this step is necessary to obtain a correct log_Z estimate,
        # consequently, if config.record_trajectory is False, the log_Z estimate
        # won't be correct. TODO: fix this limitation

        # postfix log_Z estimate
        x_final = x_final.replace(
            log_Z=x0.log_Z + (x_final.log_Z - x0.log_Z) / config.num_steps
        )

        all_xs = all_xs.replace(
            log_Z=x0.log_Z + (all_xs.log_Z - x0.log_Z) / config.num_steps
        )

        all_states = outputs[0]
    else:
        # make this branch vmap-compatible.
        all_xs = tree_map(lambda x: x[None, ...], x_final)
        kernel = config.inner_kernel_factory.build_kernel(log_p0)
        smoke_result = kernel.one_step(init_state.kernel_state, key)
        all_stats = AdaptiveSMCStepInfo(
            log_ess=jnp.array([-1.0]),
            inner_sampler_stats=smoke_result.info,
            did_resample=False,
            dists=jnp.array([-1.0])
        )
        all_states = final_carry[0]

    return AdaptiveSMCResults(x_final, AdaptiveSMCInfo(all_states, all_stats))


class SMC(
    Generic[Config_T, State_T, Info_T],
    InferenceAlgorithm[SMCConfig],
):
    config: SMCConfig[Config_T, State_T, Info_T]
    _init_smc_state: Optional[SMCStepState[State_T]] = None

    @classmethod
    def create(cls: Type[Self], config: SMCConfig[Config_T, State_T, Info_T], log_prob: LogDensity_T) -> Self:
        return cls(config, log_prob)

    def init(self, key: PRNGKeyArray, dist: np_distributions.Distribution) -> Self:
        init_state = SMCParticleApproximation.from_npdistribution(
            dist, self.config.num_samples, key
        )
        return self.replace(_init_state=init_state)

    def set_log_prob(self, log_prob: LogDensity_T) -> Self:
        self = self.replace(log_prob=log_prob)
        return self

    def run(self, key: PRNGKeyArray, update_init_state: bool = False) -> Tuple[Self, SMCResults[State_T, Info_T]]:
        assert self._init_state is not None
        assert isinstance(self._init_state, SMCParticleApproximation)
        results = smc_sampler(self.log_prob, self._init_state, self.config, key, init_state=self._init_smc_state)
        return self, results

    def run_and_update_init(
        self, key: PRNGKeyArray
    ) -> Tuple[Self, SMCResults[State_T, Info_T]]:
        self, results = self.run(key)
        final_state = cast(SMCStepState[State_T], tree_map(lambda x: x[-1], results.info.smc_steps_states))

        # some pytree manipultation
        final_state = final_state.replace(particle_approximation=final_state.particle_approximation.replace(log_prob=results.samples.log_prob))

        self = self.replace(_init_state=results.samples, _init_smc_state=final_state)
        return self, results


class AdaptiveSMC(
    Generic[Config_T, State_T, Info_T],
    InferenceAlgorithm[SMCConfig],
):
    config: AdaptiveSMCConfig[Config_T, State_T, Info_T]
    _init_smc_state: Optional[AdaptiveSMCStepState[State_T]] = None

    @classmethod
    def create(cls: Type[Self], config: AdaptiveSMCConfig[Config_T, State_T, Info_T], log_prob: LogDensity_T) -> Self:
        return cls(config, log_prob)

    def init(self, key: PRNGKeyArray, dist: np_distributions.Distribution) -> Self:
        init_state = SMCParticleApproximation.from_npdistribution(
            dist, self.config.num_samples, key
        )
        return self.replace(_init_state=init_state)

    def set_log_prob(self, log_prob: LogDensity_T) -> Self:
        self = self.replace(log_prob=log_prob)
        return self

    def run(self, key: PRNGKeyArray, update_init_state: bool = False) -> Tuple[Self, AdaptiveSMCResults[State_T, Info_T]]:
        assert self._init_state is not None
        assert isinstance(self._init_state, SMCParticleApproximation)
        results = smc_sampler_adaptive(self.log_prob, self._init_state, self.config, key, init_state=self._init_smc_state)
        return self, results

    def run_and_update_init(
        self, key: PRNGKeyArray
    ) -> Tuple[Self, SMCResults[State_T, Info_T]]:
        self, results = self.run(key)
        final_state = cast(AdaptiveSMCStepState[State_T], tree_map(lambda x: x[-1], results.info.smc_steps_states))

        # some pytree manipultation
        final_state = final_state.replace(particle_approximation=final_state.particle_approximation.replace(log_prob=results.samples.log_prob))

        self = self.replace(_init_state=results.samples, _init_smc_state=final_state)
        return self, results


class SMCFactory(Generic[Config_T, State_T, Info_T], InferenceAlgorithmFactory[SMCConfig[Config_T, State_T, Info_T]]):
    def build_algorithm(self, log_prob: LogDensity_T) -> SMC[Config_T, State_T, Info_T]:
        return SMC.create(log_prob=log_prob, config=self.config)


class AdaptiveSMCFactory(Generic[Config_T, State_T, Info_T], InferenceAlgorithmFactory[AdaptiveSMCConfig[Config_T, State_T, Info_T]]):
    def build_algorithm(self, log_prob: LogDensity_T) -> AdaptiveSMC[Config_T, State_T, Info_T]:
        return AdaptiveSMC.create(log_prob=log_prob, config=self.config)
