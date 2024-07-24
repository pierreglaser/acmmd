from typing import Optional

import jax
import jax.numpy as jnp
from flax import struct
from jax import grad, random
from jax.tree_util import tree_map

from kwgflows.base import DiscreteProbability
from kwgflows.divergences.base import Divergence, KernelizedDivergence
from kwgflows.pytypes import Array, Numeric


class GradientFlowConfig(struct.PyTreeNode):
    num_steps: int = struct.field(pytree_node=False, default=1000)
    step_size: float = struct.field(pytree_node=False, default=1e-3)


class GradientFlowResult(struct.PyTreeNode):
    divergence: KernelizedDivergence[Array]
    Ps: DiscreteProbability[Array]

    def get_Pt(self, t: Numeric):
        return tree_map(lambda x: x[t], self.Ps)

    def get_velocity_field(self, t: Numeric, reverse: bool = False):
        Pt = self.get_Pt(t)
        first_variation = self.divergence.get_first_variation(Pt)
        first_variation = jax.lax.cond(
            reverse, lambda _: -first_variation, lambda _: first_variation, None
        )
        return grad(first_variation)


def gradient_flow(
    divergence: KernelizedDivergence[Array],
    P0: DiscreteProbability[Array],
    config: GradientFlowConfig,
) -> GradientFlowResult:
    def one_step(Pt: DiscreteProbability[Array], i: Array):
        first_variation = divergence.get_first_variation(Pt)
        Pt_plus_one = Pt.push_forward(
            lambda x: x - config.step_size * first_variation.grad(x)
        )
        return Pt_plus_one, Pt_plus_one

    PT, trajectory = jax.lax.scan(one_step, P0, jnp.arange(config.num_steps))
    return GradientFlowResult(divergence=divergence, Ps=trajectory)
