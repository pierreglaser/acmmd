import pandas as pd
from get_gearnet_embeddings import get_profile
from jax import random
from kwgflows.rkhs.kernels import gaussian_kernel
from pmpnn_extra.jax_pmpnn import JaxProteinMPNN
from pmpnn_extra.kernels import HammingIMQKernel
from pmpnn_extra.preprocessing import Dataset

from calibration.statistical_tests.cgof_mmd import cgof_mmd_test

# TODO
# Generate data from ProteinMPNN and score it against proteinmpnn itself

if __name__ == "__main__":
    p = get_profile("anon")
    df = pd.read_csv(p["ref_fnm"])

    d = Dataset.from_metadata(df.iloc[:10], p["pdb_chain_repo"], p["ckpt_fnm"])

    jax_model = JaxProteinMPNN.create(
        d.get_pdbs(format="pmpnn"), batching_method="vmap"
    )

    k_x = gaussian_kernel.create()
    k_y = HammingIMQKernel.create()

    X = d.get_encodings()
    Y = d.get_sequences()

    k_x = k_x.with_median_heuristic(X)

    # gx = k_x.make_gram_matrix(X, X)
    # gy = k_y.make_gram_matrix(Y, Y)

    key = random.PRNGKey(0)
    cgof = cgof_mmd_test(
        k_x,
        k_y,
        median_heuristic=False,
        approximation_num_particles=1,
        approximate=True,
    )
    e = cgof.unbiased_estimate(
        X,
        Y,
        jax_model,
        key=key,
    )
