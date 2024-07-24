"""
Distributions conversion from pyro to numpyro and tensorflow probability.

This is needed because the prior density is needed to train the model. This is unlike
for the simulators, from which we just need to draw samples.
"""
from typing import Literal, Union, overload

import jax.numpy as jnp
from numpyro import distributions as np_distributions
from numpyro.distributions import transforms as np_transforms
from pyro import distributions as pyro_distributions
from torch import distributions as tdist
from torch import distributions as torch_distributions

_DISTMODULE_TARGETS_T = Literal["numpyro"]

_CONVERTER_DISPATCH = {}

_TRANSFORMS_DISPATCH = {}


def _register_converter(converter, type_):
    _CONVERTER_DISPATCH[type_] = converter


def _register_transforms_converter(converter, type_):
    _TRANSFORMS_DISPATCH[type_] = converter


@overload
def convert_dist(
    d: Union[pyro_distributions.Distribution, tdist.Distribution],
    implementation: Literal["pyro"],
) -> Union[pyro_distributions.Distribution, tdist.Distribution]:
    ...


@overload
def convert_dist(
    d: Union[pyro_distributions.Distribution, tdist.Distribution],
    implementation: Literal["numpyro"],
) -> np_distributions.Distribution:
    ...


def convert_dist(
    d: Union[pyro_distributions.Distribution, tdist.Distribution],
    implementation: Literal["pyro", "numpyro"],
) -> Union[
    np_distributions.Distribution, pyro_distributions.Distribution, tdist.Distribution
]:
    assert isinstance(d, (pyro_distributions.Distribution, tdist.Distribution))

    if implementation == "pyro":
        assert isinstance(d, pyro_distributions.Distribution)
        return d

    assert implementation == "numpyro"
    dist = _CONVERTER_DISPATCH[type(d)](d, implementation)
    assert isinstance(dist, np_distributions.Distribution)
    return dist


def _get_tensor_converter(implementation: _DISTMODULE_TARGETS_T):
    if implementation in ("numpyro",):
        return jnp.array
    else:
        raise ValueError


def _convert_multivariate_normal(
    d: pyro_distributions.MultivariateNormal, implementation: _DISTMODULE_TARGETS_T
):
    converter = _get_tensor_converter(implementation=implementation)
    loc, precision_matrix = converter(d.loc), converter(d.precision_matrix)

    if implementation == "numpyro":
        from numpyro.distributions import continuous as np_continuous

        return np_continuous.MultivariateNormal(
            loc=loc, precision_matrix=precision_matrix  # type: ignore
        )
    else:
        raise ValueError


_register_converter(_convert_multivariate_normal, pyro_distributions.MultivariateNormal)


def _convert_uniform(
    d: pyro_distributions.Uniform, implementation: _DISTMODULE_TARGETS_T
):
    converter = _get_tensor_converter(implementation=implementation)
    low, high = converter(d.low), converter(d.high)

    if implementation == "numpyro":
        return np_distributions.Uniform(low, high, validate_args=True)  # type: ignore
    else:
        raise ValueError


_register_converter(_convert_uniform, pyro_distributions.Uniform)


def _convert_lognormal(
    d: pyro_distributions.LogNormal, implementation: _DISTMODULE_TARGETS_T
):
    # XXX: I don't support this case yet.
    assert not isinstance(d.base_dist, torch_distributions.Independent)

    converter = _get_tensor_converter(implementation=implementation)
    loc, scale = (
        converter(d.base_dist.loc),
        converter(d.base_dist.scale),
    )
    if implementation == "numpyro":
        return np_distributions.LogNormal(loc, scale)  # type: ignore
    else:
        raise ValueError


_register_converter(_convert_lognormal, pyro_distributions.LogNormal)


def _convert_independent(
    d: pyro_distributions.Independent, implementation: _DISTMODULE_TARGETS_T
):
    reinterpreted_batch_ndims = d.reinterpreted_batch_ndims
    if implementation == "numpyro":
        return np_distributions.Independent(
            convert_dist(d.base_dist, implementation),
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )
    else:
        raise ValueError


_register_converter(_convert_independent, pyro_distributions.Independent)


def _convert_transformed_dist(
    d: tdist.TransformedDistribution, implementation: _DISTMODULE_TARGETS_T
):
    if implementation == "numpyro":
        bd = d.base_dist
        assert isinstance(bd, (pyro_distributions.Distribution, tdist.Distribution))
        return np_distributions.TransformedDistribution(
            convert_dist(bd, implementation),
            [convert_transform(t, implementation) for t in d.transforms],
        )
    else:
        raise ValueError


_register_converter(_convert_transformed_dist, tdist.TransformedDistribution)


def convert_transform(
    t: tdist.transforms.Transform, implementation: _DISTMODULE_TARGETS_T
):
    if implementation == "numpyro":
        return _TRANSFORMS_DISPATCH[type(t)](t, implementation)
    else:
        raise ValueError


def _convert_affine_transform(
    t: tdist.transforms.AffineTransform, implementation: _DISTMODULE_TARGETS_T
) -> np_transforms.AffineTransform:
    converter = _get_tensor_converter(implementation=implementation)
    loc, scale = converter(t.loc), converter(t.scale)

    if implementation == "numpyro":
        from numpyro.distributions.transforms import AffineTransform

        return AffineTransform(loc, scale)
    else:
        raise ValueError


_register_transforms_converter(
    _convert_affine_transform, tdist.transforms.AffineTransform
)


def _convert_sigmoid_transform(
    t: tdist.transforms.SigmoidTransform, implementation: _DISTMODULE_TARGETS_T
) -> np_transforms.SigmoidTransform:
    if implementation == "numpyro":
        from numpyro.distributions.transforms import SigmoidTransform

        return SigmoidTransform()
    else:
        raise ValueError


_register_transforms_converter(
    _convert_sigmoid_transform, tdist.transforms.SigmoidTransform
)


def _convert_exp_transform(
    t: tdist.transforms.ExpTransform, implementation: _DISTMODULE_TARGETS_T
) -> np_transforms.ExpTransform:
    if implementation == "numpyro":
        from numpyro.distributions.transforms import ExpTransform

        return ExpTransform()
    else:
        raise ValueError


_register_transforms_converter(_convert_exp_transform, tdist.transforms.ExpTransform)


def _convert_inverse_transform(
    t: tdist.transforms._InverseTransform, implementation: _DISTMODULE_TARGETS_T
) -> np_transforms._InverseTransform:
    if implementation == "numpyro":
        return convert_transform(t.inv, implementation).inv
    else:
        raise ValueError


_register_transforms_converter(
    _convert_inverse_transform, tdist.transforms._InverseTransform
)


def _convert_independent_transform(
    t: tdist.transforms.IndependentTransform, implementation: _DISTMODULE_TARGETS_T
) -> np_transforms.IndependentTransform:
    if implementation == "numpyro":
        return np_transforms.IndependentTransform(
            convert_transform(t.base_transform, implementation),
            t.reinterpreted_batch_ndims,
        )
    else:
        raise ValueError


_register_transforms_converter(
    _convert_independent_transform, tdist.transforms.IndependentTransform
)


def _convert_compose_transforms(
    t: tdist.transforms.ComposeTransform, implementation: _DISTMODULE_TARGETS_T
) -> np_transforms.ComposeTransform:
    if implementation == "numpyro":
        if len(t.parts) == 0:
            return np_transforms.ComposeTransform([np_transforms.IdentityTransform()])
        else:
            return np_transforms.ComposeTransform(
                [convert_transform(part, implementation) for part in t.parts]
            )

    else:
        raise ValueError


_register_transforms_converter(
    _convert_compose_transforms, tdist.transforms.ComposeTransform
)
