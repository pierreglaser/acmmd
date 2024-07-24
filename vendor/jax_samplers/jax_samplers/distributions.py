from typing import Callable, Protocol, List
import jax.numpy as jnp

from flax import struct

from jax_samplers.pytypes import (Array, DoublyIntractableJointLogDensity_T,
                             DoublyIntractableLogDensity_T, LogDensity_T, LogJoint_T,
                             LogLikelihood_T, Numeric)
import numpyro.distributions as np_distributions


class LogDensityNode(struct.PyTreeNode):
    log_prob: Callable = struct.field(pytree_node=False)

    def __call__(self, x: Array) -> Numeric:
        return self.log_prob(x)


class LogJointNode(struct.PyTreeNode):
    log_prob: Callable = struct.field(pytree_node=False)

    def __call__(self, theta: Array, x: Array) -> Numeric:
        return self.log_prob(theta, x)


# def maybe_wrap(log_prob: LogDensity_T) -> LogDensity_T:
#     if isinstance(log_prob, struct.PyTreeNode):
#         return log_prob
#     else:
#         return LogDensityNode(log_prob)



def maybe_wrap(log_prob: LogDensity_T) -> LogDensity_T:
    return LogDensityNode(log_prob)


def maybe_wrap_joint(log_prob: LogJoint_T) -> LogJoint_T:
    if isinstance(log_prob, struct.PyTreeNode):
        return log_prob
    return LogJointNode(log_prob)


class LogLikelihoodNode(struct.PyTreeNode):
    log_likelihood: LogLikelihood_T = struct.field(pytree_node=False)

    def __call__(self, theta: Array, x: Array) -> Numeric:
        return self.log_likelihood(theta, x)


def maybe_wrap_log_l(log_prob: LogLikelihood_T) -> LogLikelihood_T:
    return LogLikelihoodNode(log_prob)


class DoublyIntractableJointLogDensity(struct.PyTreeNode):
    log_prior: LogDensity_T
    log_likelihood: LogLikelihood_T
    dim_param: int = struct.field(pytree_node=False)
    expose_tilted_log_joint: bool = struct.field(pytree_node=False, default=True)

    def tilted_log_joint(self, x: Array) -> Numeric:
        # because this function discards the likelihood log-normalizer, the
        # output of this function corresponds to a tilted model where the
        # likelihood corresponds to the true (self.)log-likelihood, but the
        # prior does not match sels.prior.
        theta, x = x[..., : self.dim_param], x[..., self.dim_param :]
        return self.log_prior(theta) + self.log_likelihood(theta, x)

    def __call__(self, x: Array) -> Numeric:
        return self.tilted_log_joint(x)


class DoublyIntractableLogDensity(struct.PyTreeNode):
    log_prior: LogDensity_T
    log_likelihood: LogLikelihood_T
    x_obs: Array

    def __call__(self, x: Array) -> Numeric:
        # XXX: this is confusing. Inputs to __call__ log density should have a
        # generic name like "val", and not x.
        theta = x
        return self.log_prior(theta) + self.log_likelihood(theta, self.x_obs)


# wrapper for conditionned likelihood objects
class ThetaConditionalLogDensity(struct.PyTreeNode, LogDensity_T):
    log_prob: LogLikelihood_T
    theta: Array

    def __call__(self, x: Array) -> Numeric:
        return self.log_prob(self.theta, x)
        # return self.log_prob(jnp.concatenate([self.theta, x]))


class BlockDistribution(np_distributions.Distribution):
    arg_constraints = {"distributions": None}
    def __init__(self, distributions: List[np_distributions.Distribution]):
        self.distributions = distributions
        for dist in distributions:
            assert dist.batch_shape == ()
            assert len(dist.event_shape) == 1

        individual_event_shapes = [dist.event_shape for dist in distributions]
        self._individual_event_shapes = individual_event_shapes
        self._event_shape_bounds = [0] + list(jnp.cumsum(jnp.array([es[0] for es in self._individual_event_shapes])))

        super(BlockDistribution, self).__init__(
            batch_shape=(), event_shape=(sum(es[0] for es in individual_event_shapes),)
        )

    def sample(self, key, sample_shape=()):
        samples = []
        for dist in self.distributions:
            samples.append(dist.sample(key, sample_shape))
        return jnp.concatenate(samples, axis=-1)


    def log_prob(self, value):
        log_probs = []
        for i, dist in enumerate(self.distributions):
            log_probs.append(dist.log_prob(value[..., self._event_shape_bounds[i]:self._event_shape_bounds[i+1]]))
        log_probs = jnp.concatenate(jnp.atleast_1d(*log_probs), axis=-1)
        return jnp.sum(log_probs, axis=-1)
