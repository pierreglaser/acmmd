from kwgflows.pytypes import PyTree_T

import jax


def infer_num_samples_pytree(pytree: PyTree_T) -> int:
    # utility function that infers the number of elements in a vmapped PyTree
    # this function requires the input PyTree to be already vmapped.
    # the number of element is inferred to be the number of elements in the first
    # axis of the first leaf of the PyTree.
    shapes = jax.tree_map(lambda x: x.shape[0], pytree)
    # include shapes in list to handle the case where shapes is a scalar
    unique_shapes = set(jax.tree_util.tree_leaves([shapes]))
    assert (
        len(unique_shapes) == 1
    ), "PyTree must have the same number of samples for each elem"
    return list(unique_shapes)[0]
