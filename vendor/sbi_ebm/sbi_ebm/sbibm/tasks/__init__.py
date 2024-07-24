from typing import Optional, Type

import jax.numpy as jnp
import numpy as np
import pyro.distributions as pyro_distributions
import sbibm
import torch
from jax_samplers.pytypes import Array, Simulator_T
from numpyro import distributions as np_distributions
from numpyro.distributions import transforms as np_transforms
from sbi_ebm.sbibm.pyro_to_numpyro import convert_dist
from sbibm.tasks import Task
from typing_extensions import Self


def get_task(task_name: str) -> Task:
    return sbibm.get_task(task_name)


class JaxTask:
    def __init__(self, task: Task) -> None:
        self.task = task

    def get_prior(self):
        def prior(num_samples: int = 1):
            return jnp.array(self.task.get_prior()(num_samples).detach().numpy())

        return prior

    def get_prior_dist(self) -> np_distributions.Distribution:
        prior_dist = self.task.get_prior_dist()
        assert isinstance(prior_dist, pyro_distributions.Distribution), prior_dist

        converted_prior_dist = convert_dist(prior_dist, implementation="numpyro")
        assert isinstance(converted_prior_dist, np_distributions.Distribution)
        return converted_prior_dist

    @classmethod
    def from_task_name(cls: Type[Self], task_name: str) -> Self:
        from sbi_ebm.sbibm.tasks import get_task

        return cls(get_task(task_name))

    def get_simulator(self) -> Simulator_T:
        pyro_simulator = self.task.get_simulator()

        def simulator(thetas: Array) -> Array:
            torch_thetas = torch.from_numpy(np.array(thetas)).float()
            return jnp.array(pyro_simulator(torch_thetas))

        return simulator

    def get_observation(self, num_observation: int) -> Array:
        return jnp.array(self.task.get_observation(num_observation))

    def _parameter_event_space_bijector(self) -> np_transforms.Transform:
        prior_dist = self.get_prior_dist()
        return np_distributions.biject_to(prior_dist.support)

    def __reduce__(self):
        return JaxTask.from_task_name, (self.task.name,)
