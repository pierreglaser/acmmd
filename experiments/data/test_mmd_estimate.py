from calibration.statistical_tests.cgof_mmd import cgof_mmd_test
from calibration.statistical_tests.mmd import mmd_test
from kwgflows.rkhs.kernels import TensorProductKernel, gaussian_kernel
import pandas as pd
from pmpnn_extra.jax_pmpnn import JaxProteinMPNN
from pmpnn_extra.preprocessing import Dataset, replace_chain_num_with_names
from exp_using_pmpnn_data import pmpnn_pdb_as_pytree
from jax import tree_map
from jax import random
import jax
import torch

from utils import get_profile  # type: ignore


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_device('cuda')


    p = get_profile("anon")
    df = pd.read_csv(p["ref_fnm"])

    num_pdbs = -1

    temperature = 0.1

    d = Dataset.from_cath_query(
        pdb_folder=p["pdb_chain_repo"],
        ckpt_fnm=p["ckpt_fnm"],
        esm2_rep_fnm=p["esm2_rep_fnm"],
        with_caching=True,
        dask_cluster_args=p["dask_cluster"],
        cath_query={"Architecture Number": 15},
        # cath_query={"Topology Number": 8},
        pmpnn_loading_method="pmpnn_notied",
        max_num_pdbs=20
        # max_num_pdbs=None
    )

    _ = d.get_pdbs(format="pmpnn_notied")
    X, Y, pdbs = d.get_XY()
    torch.cuda.empty_cache()


    temperature_2 = 1.0

    jax_models = JaxProteinMPNN.create(
        pdbs, batching_method="native", temperature=temperature,
        return_esm2_embeddings=True
    )

    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    Z = jax_models.sample_from_conditional(key=subkey, num_samples=1)
    torch.cuda.empty_cache()

    k_x = gaussian_kernel.create()
    k_y = gaussian_kernel.create()
    k_x = k_x.with_median_heuristic(X)
    k_y = k_y.with_median_heuristic(Y)

    tk = TensorProductKernel(k_x, k_y)
    mmd = mmd_test(tk)
    key, subkey = random.split(key)
    r = mmd((X, Y), (X, Z), key=subkey)
