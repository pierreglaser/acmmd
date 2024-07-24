from typing import Type
from typing import Callable, Optional, Tuple, Union
from typing_extensions import Self

from flax import struct
from jax import random, vmap
from jax.nn import logsumexp, softmax
import jax.numpy as jnp
from numpyro import distributions as np_distributions

from jax_samplers.pytypes import Array, Numeric, PRNGKeyArray
from jax_samplers.kernels.base import Info, KernelConfig, MHKernel, MHKernelFactory, State
from jax_samplers.particle_aproximation import ParticleApproximation


class DiscreteGibbsState(State):
    dim_z: int = struct.field(default=1, pytree_node=False)

    @property
    def theta_particle(self):
        return self.x[..., : self.dim_z]

    @property
    def x_particle(self):
        return self.x[..., self.dim_z :]


class DiscreteGibbsInfo(Info):
    accept: Numeric
    log_alpha: Numeric


class DiscreteLogDensity(struct.PyTreeNode):
    """
    Discrete relaxation of a unnormalized log density of the form:

        p(x) = exp(-E(x)) g(x)


    Where g is a base measure with available IID samples x_i which are used to
    approximate p using:

        p(x) = \sum_i exp(-E(x_i)) \delta(x_i)
    """
    _thetas: jnp.ndarray = struct.field(pytree_node=True, default=None)
    _xs: jnp.ndarray = struct.field(pytree_node=True, default=None)

    # set to optional because this argument is not used for some subclasses (annealed log probs)
    _log_prob: Optional[Callable[[Array, Array], Numeric]] = struct.field(
        pytree_node=True, default=None
    )

    def log_prob(self, theta: Array, x: Array) -> Numeric:
        assert self._log_prob is not None
        return self._log_prob(theta, x)

    def __call__(self, x: Array) -> Numeric:
        assert len(x) == 2, len(x)
        if isinstance(x, tuple):
            i, j = x
        else:
            i, j = x[0], x[1]
        return self.log_prob(self._thetas[i, ...], self._xs[j, ...])

    @property
    def dim_z(self):
        # index size of thetas
        return 1

    def make_true_hist(self):
        # XXX: works only for 2D arrays.
        unnormalized_log_probs = jnp.vectorize(self.log_prob, signature="(n),(n)->()")(
            self._thetas[..., None, :], self._xs[None, ..., :]
        )

        log_probs = unnormalized_log_probs - logsumexp(unnormalized_log_probs)
        return jnp.exp(log_probs)


    def true_average_of(self, f, returns_scalar=False):
        if returns_scalar:
            f_vals = jnp.vectorize(f, signature="(n),(n)->()")(
                self._thetas[..., None, :], self._xs[None, ..., :]
            )
            return (f_vals * self.make_true_hist()).mean()
        else:
            f_vals = jnp.vectorize(f, signature="(n),(n)->(k)")(
                self._thetas[..., None, :], self._xs[None, ..., :]
            )
            return (f_vals * self.make_true_hist()[..., None]).sum(axis=(0, 1))

    def make_empirical_hist(self, pa: ParticleApproximation):
        empirical_hist = jnp.zeros((self._thetas.shape[0], self._xs.shape[0]))

        def scan_func(empirical_hist: Array, arg: Tuple[Array, Array]):
            particle, w = arg
            theta_idx, x_idx = particle[0], particle[1]
            empirical_hist = empirical_hist.at[theta_idx, x_idx].set(
                empirical_hist[theta_idx, x_idx] + w
            )
            return empirical_hist, None

        from jax.lax import scan  # type: ignore
        empirical_hist, _ = scan(scan_func, empirical_hist, (pa.particles, pa.normalized_ws))
        return empirical_hist



class DiscreteGibbsConfig(KernelConfig):
    _smoke_vmap_placeholder: float = 0.



class DiscreteGibbsKernel(MHKernel[DiscreteGibbsConfig, DiscreteGibbsState, DiscreteGibbsInfo]):
    target_log_prob: DiscreteLogDensity

    @classmethod
    def create(cls: Type[Self], target_log_prob: DiscreteLogDensity, config: DiscreteGibbsConfig) -> Self:
        return cls(target_log_prob=target_log_prob, config=config)

    def init_state(self, x: Array) -> DiscreteGibbsState:
        i, j = x

        i = jnp.array(i)
        j = jnp.array(j)

        x = jnp.concatenate(
            [self.target_log_prob._thetas[i, ...], self.target_log_prob._xs[j, ...]],
            axis=-1,
        )
        ij = jnp.concatenate([jnp.atleast_1d(i), jnp.atleast_1d(j)], axis=-1)
        return DiscreteGibbsState(x=ij, dim_z=1)

    def _sample_from_proposal(
        self, key: PRNGKeyArray, x: DiscreteGibbsState, 
    ) -> DiscreteGibbsState:
        # update x | theta
        all_xs_idxs = jnp.arange(self.target_log_prob._xs.shape[0])

        x_vmapped_log_density = vmap(
            self.target_log_prob.__call__, in_axes=((None, 0),)  # type: ignore
        )
        x_logits = x_vmapped_log_density((x.theta_particle[..., 0], all_xs_idxs))
        key, subkey = random.split(key)

        import numpyro.distributions as np_distributions
        mn_x = np_distributions.Categorical(probs=softmax(x_logits))
        new_x_idx = mn_x.sample(subkey)

        # update theta | x
        theta_vmapped_log_density = vmap(
            self.target_log_prob.__call__, in_axes=((0, None),)  # type: ignore
        )
        all_thetas_idxs = jnp.arange(self.target_log_prob._thetas.shape[0])
        theta_logits = theta_vmapped_log_density((all_thetas_idxs, new_x_idx))
        key, subkey = random.split(key)

        import numpyro.distributions as np_distributions
        mn_theta = np_distributions.Categorical(probs=softmax(theta_logits))
        new_theta_idx = mn_theta.sample(subkey)

        # return new state
        new_theta_and_x_idxs = jnp.concatenate([jnp.atleast_1d(new_theta_idx), jnp.atleast_1d(new_x_idx)], axis=-1)
        return x.replace(x=new_theta_and_x_idxs)

    def _compute_accept_prob(self, proposal: DiscreteGibbsState, x: DiscreteGibbsState) -> Numeric:
        return 1.

    def _build_info(self, accept: Numeric, log_alpha: Numeric) -> DiscreteGibbsInfo:
        return DiscreteGibbsInfo(accept, log_alpha)


class DiscreteGibbsKernelFactory(MHKernelFactory[DiscreteGibbsConfig, DiscreteGibbsState, DiscreteGibbsInfo]):
    kernel_cls: Type[DiscreteGibbsKernel] = struct.field(pytree_node=False, default=DiscreteGibbsKernel)

    def build_kernel(self, log_prob: DiscreteLogDensity) -> DiscreteGibbsKernel:
        return self.kernel_cls.create(log_prob, self.config)
