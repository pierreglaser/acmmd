from jax.experimental import host_callback
from tqdm.std import tqdm as tqdm_tp

import jax
from jax.experimental import host_callback
import jax.numpy as jnp


def progress_bar_factory(tqdm_bar: tqdm_tp, num_iters):
    """Factory that builds a progress bar decorator along
    with the `set_tqdm_description` and `close_tqdm` functions
    """
    if num_iters > 100:
        print_rate = int(num_iters / 100)
    else:
        print_rate = 1

    # remainder = num_iters % print_rate

    total_calls = 0

    def _update_tqdm(arg, transform):
        nonlocal total_calls
        total_calls += 1
        thinning, chain_ids = arg
        num_chains = len(chain_ids)
        # print(total_calls, thinning * num_chains, arg)
        tqdm_bar.set_description(f"Running chain", refresh=True)
        tqdm_bar.update(thinning * num_chains)

    def _close_tqdm(arg, transform, device):
        tqdm_bar.update(arg)
        tqdm_bar.close()

    def _update_progress_bar(iter_num, chain_id) -> int:
        """Updates tqdm progress bar of a JAX loop only if the iteration number is a multiple of the print_rate
        Usage: carry = progress_bar((iter_num, print_rate), carry)
        """
        from jax.tree_util import tree_all
        cond = jnp.sum((chain_id == 1) & (chain_id == 0)).astype(bool)
        ret = jax.lax.cond(
            (iter_num + 1) % print_rate == 0,
            lambda : host_callback.id_tap(_update_tqdm, (print_rate, chain_id), result=iter_num+1),
            lambda : iter_num,
        )
        return ret

    return _update_progress_bar


