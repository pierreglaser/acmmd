"Converting a PMPNN forward pass function into a jax function"
# pyright: basic, reportPrivateUsage=false, reportPrivateImportUsage=false

import jax
import jax.numpy as jnp
import torch
from jax.dlpack import from_dlpack as jax_from_dlpack
from jax.dlpack import to_dlpack as jax_to_dlpack
from torch.utils.dlpack import from_dlpack as pyt_from_dlpack
from torch.utils.dlpack import to_dlpack as pyt_to_dlpack


def as_torch_pytree(x):
    # If x is a JAX ndarray, convert it to a DLPack and then to a PyTorch tensor
    if isinstance(x, jnp.ndarray):
        x = jax_to_dlpack(x)
        x = pyt_from_dlpack(x)
    elif isinstance(x, (float, int)):
        x = torch.tensor(x)
    return x


def as_jax_pytree(x):
    # If x is a PyTorch tensor, convert it to a DLPack and then to a JAX ndarray
    if isinstance(x, torch.Tensor):
        try:
            dlpack_obj = pyt_to_dlpack(x)
            x = jax_from_dlpack(dlpack_obj)
        except Exception:
            x = jnp.array(x.detach().cpu().numpy())
    return x



def sequential_vmap(
    func,
    flat_in_dims,
    pytree_def,
    randomness="different",
):
    def new_func(*flat_args):
        assert (d in (None, 0) for d in flat_in_dims)
        assert 0 in flat_in_dims
        batched_idxs = [i for i, d in enumerate(flat_in_dims) if d == 0]
        assert len(batched_idxs) > 0
        first_arg = flat_args[batched_idxs[0]]
        retvals = []

        for i in range(len(first_arg)):
            sliced_flat_args = []
            for d, arg in zip(flat_in_dims, flat_args):
                if d is not None:
                    sliced_flat_args.append(arg[i])
                else:
                    sliced_flat_args.append(arg)
            sliced_args = jax.tree_util.tree_unflatten(pytree_def, sliced_flat_args)
            sliced_args = jax.tree_map(as_torch_pytree, sliced_args)
            # __import__('pdb').set_trace()
            retvals.append(func(*sliced_args))
        retvals = [jax.tree_map(as_jax_pytree, r) for r in retvals]
        retvals = jax.tree_map(lambda *x: jnp.stack(x, axis=0), *retvals)
        return jnp.array(retvals)
    return new_func


def batched_vmap(
    func,
    flat_in_dims,
    pytree_def,
    randomness="different",
):
    batch_size = 5

    def new_func(*flat_args):
        assert (d in (None, 0) for d in flat_in_dims)
        assert 0 in flat_in_dims
        batched_idxs = [i for i, d in enumerate(flat_in_dims) if d == 0]
        assert len(batched_idxs) > 0
        first_arg = flat_args[batched_idxs[0]]
        retvals = []

        num_batches = len(first_arg) // batch_size + 1 * ((len(first_arg) % batch_size) > 0)

        for i in range(num_batches):
            sliced_flat_args = []
            for d, arg in zip(flat_in_dims, flat_args):
                if d is not None:
                    sliced_flat_args.append(arg[i * batch_size: (i + 1) * batch_size])
                else:
                    sliced_flat_args.append(arg)
            sliced_args = jax.tree_util.tree_unflatten(pytree_def, sliced_flat_args)
            sliced_args = jax.tree_map(as_torch_pytree, sliced_args)
            ret = func(*sliced_args)
            retvals.append(ret)

        retvals = [jax.tree_map(as_jax_pytree, r) for r in retvals]
        retvals = jax.tree_map(lambda *x: jnp.concatenate(x, axis=0), *retvals)
        return jnp.array(retvals)
    return new_func


def as_jax_function(fun, mode="torch.func"):
    from jax._src.core import Primitive
    from jax.interpreters.batching import primitive_batchers

    f_p = Primitive("custom_func")

    if mode == "torch.func":
        # assumes purity
        def f_impl(*flattened_args, pytree_def, all_flat_batched_dims=()):
            # Unflatten the input data and the batched_dims
            args = jax.tree_util.tree_unflatten(
                pytree_def, flattened_args  # pyright: ignore[reportGeneralTypeIssues]
            )
            # Convert the input data from PyTorch to JAX representations
            args = jax.tree_map(as_torch_pytree, args)

            b_fun = fun
            for flat_batched_dims in all_flat_batched_dims:
                batched_dims = jax.tree_util.tree_unflatten(pytree_def, flat_batched_dims)
                b_fun = torch.vmap(b_fun, in_dims=batched_dims, randomness="different")
            out = b_fun(*args)
            # Convert the output data from JAX to PyTorch representations
            out = jax.tree_map(as_jax_pytree, out)
            return out

        f_p.def_impl(f_impl)
    elif mode == "sequential":
        # assumes purity
        def f_impl(*flattened_args, pytree_def, all_flat_batched_dims=()):
            # Unflatten the input data and the batched_dims
            args = jax.tree_util.tree_unflatten(
                pytree_def, flattened_args  # pyright: ignore[reportGeneralTypeIssues]
            )
            # Convert the input data from PyTorch to JAX representations
            args = jax.tree_map(as_torch_pytree, args)

            b_fun = fun
            assert len(all_flat_batched_dims) in (0, 1)
            if len(all_flat_batched_dims) == 0:
                out = b_fun(*args)
            else:
                for flat_batched_dims in all_flat_batched_dims:
                    # batched_dims = jax.tree_util.tree_unflatten(pytree_def, flat_batched_dims)
                    b_fun = sequential_vmap(b_fun, flat_batched_dims, pytree_def, randomness="different")
                out = b_fun(*flattened_args)
            # Convert the output data from JAX to PyTorch representations
            out = jax.tree_map(as_jax_pytree, out)
            return out

        f_p.def_impl(f_impl)

    elif mode == "passhtrough_batched":
        # assumes purity
        def f_impl(*flattened_args, pytree_def, all_flat_batched_dims=()):
            # Unflatten the input data and the batched_dims
            args = jax.tree_util.tree_unflatten(
                pytree_def, flattened_args  # pyright: ignore[reportGeneralTypeIssues]
            )
            # Convert the input data from PyTorch to JAX representations
            args = jax.tree_map(as_torch_pytree, args)

            b_fun = fun
            assert len(all_flat_batched_dims) in (0, 1)
            if len(all_flat_batched_dims) == 0:
                out = b_fun(*args)
            else:
                for flat_batched_dims in all_flat_batched_dims:
                    # batched_dims = jax.tree_util.tree_unflatten(pytree_def, flat_batched_dims)
                    b_fun = batched_vmap(b_fun, flat_batched_dims, pytree_def, randomness="different")
                out = b_fun(*flattened_args)
            # Convert the output data from JAX to PyTorch representations
            out = jax.tree_map(as_jax_pytree, out)
            return out

        f_p.def_impl(f_impl)

    def f_batched(flattened_args, batched_dims, pytree_def, all_flat_batched_dims=()):
        # XXX: limited output shape inference for now
        assert all(d in (None, 0) for d in batched_dims)
        assert any(d is not None for d in batched_dims)
        dim_x = 0

        return (
            f_p.bind(
                *flattened_args,
                pytree_def=pytree_def,
                all_flat_batched_dims=(*all_flat_batched_dims, batched_dims)
            ),
            dim_x,
        )

    primitive_batchers[f_p] = f_batched

    def wrapper(*args, **kwargs):
        flattened_args, pytree_def = jax.tree_util.tree_flatten(args)
        return f_p.bind(*flattened_args, pytree_def=pytree_def, **kwargs)

    return wrapper
