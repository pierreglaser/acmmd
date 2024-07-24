import jax.numpy as jnp
from jax import random
from pmpnn_extra.jax_pmpnn import JaxProteinMPNN
from pmpnn_extra.kernels import DTWKernel, HammingIMQKernel
from pmpnn_extra.pmpnn_utils import get_pmpnn_model
from pmpnn_extra.protein_file_utils import protein_structure_from_files

from calibration.statistical_tests.cgof_mmd import cgof_mmd_test

if __name__ == "__main__":
    model = get_pmpnn_model()
    tfs = protein_structure_from_files(pdb_names=["6MRR.pdb", "5L33.pdb"])
    jax_model = JaxProteinMPNN.create(model, tfs, batching_method="vmap")

    all_Ss = []
    for i, l in enumerate(jax_model.tf.lengths):
        print(l[0])
        new_S = jax_model.tf.S[i, 0].astype(float).at[l[0] :].set(jnp.nan)
        all_Ss.append(new_S)

    all_Ss = jnp.stack(all_Ss)

    k_y = HammingIMQKernel.create()
    gm_y = k_y.make_gram_matrix(all_Ss, all_Ss)

    all_Xs = []
    for i, l in enumerate(jax_model.tf.lengths):
        print(l[0])
        new_X = jax_model.tf.X[i, 0].astype(float).at[l[0] :].set(jnp.nan)
        # flatten the 4 atom 3d positions in each amino acid to a 12d vector
        all_Xs.append(new_X.reshape(new_X.shape[0], -1))

    all_Xs = jnp.stack(all_Xs)

    # center position per sequence.
    # TODO: introduce a rotation step to align the sequences.
    all_Xs = all_Xs - jnp.nanmean(all_Xs, axis=1, keepdims=True)

    # small gamma yields close-to-hard-dtw kernel. Large gamma
    # will yield larger values. Diagonal dominance needs to be investigated
    # on a larger dataset
    k_x = DTWKernel.create(gamma=0.01).with_median_heuristic(all_Xs, quantile=0.9)
    k_x = k_x.calibrate_scale(all_Xs)

    print("bandwidth", k_x.bandwidth)
    print("log scale", k_x.log_scale)

    test = cgof_mmd_test(x_kernel=k_x, y_kernel=k_y, approximate=True)
    val = test.__call__(all_Xs, all_Ss, jax_model, key=random.PRNGKey(1234))
