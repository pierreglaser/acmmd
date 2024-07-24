# pyright: basic
from typing import Optional
from calibration.statistical_tests.mmd import mmd_test
from kwgflows.rkhs.kernels import TensorProductKernel, gaussian_kernel
import pandas as pd
from pmpnn_extra.jax_pmpnn import JaxProteinMPNN
from pmpnn_extra.preprocessing import Dataset
from jax import random
from pmpnn_extra.preprocessing import select_pdbs
import torch, os, sys
from pathlib import Path

from argparse import ArgumentParser
import json
import numpy as np


def get_profile(profile_name):
    with open(Path(__file__).parent / "config.json", "r") as f:
        profile = json.load(f)[profile_name]
    return profile

def mmd_pmpnn(d: Dataset,
              temperature,
              key,
              upsample_to: Optional[int] = None):
    
    X, Y_true, pdbs = d.get_XY()
    torch.cuda.empty_cache()

    if upsample_to is not None:
        num_unique_samples = X.shape[0]
        key, subkey = random.split(key)
        idxs = random.randint(subkey, (upsample_to,), 0, num_unique_samples)
        X = X[idxs]
        pdbs = select_pdbs(pdbs, idxs)

    # Y: true samples

    # in this exp, true model = PMPNN with temperature_true
    model = JaxProteinMPNN.create(
        pdbs, batching_method="native", temperature=temperature,
        return_esm2_embeddings=True
    )

    key, subkey = random.split(key)
    Y_model = model.sample_from_conditional(key=subkey, num_samples=1)
    torch.cuda.empty_cache()

    k_x = gaussian_kernel.create()
    k_y = gaussian_kernel.create()
    tk = TensorProductKernel(k_x, k_y)
    mmd = mmd_test(tk, median_heuristic=True)

    key, subkey = random.split(key)
    r = mmd((X, Y_true), (X, Y_model), key=subkey)
    return r, mmd



CATH_FIELDS = ('Class Number', 'Architecture Number', 'Topology Number', 'Superfamily Number')


def get_topologies_within_req(
    ref_df,
    length_tr=100,
    num_data_points_tr=10,
    fields=CATH_FIELDS
):
    #subset to topologies with enough sequences 
    df = ref_df.set_index(list(fields))
    sliced_df = df.groupby(
        axis=0, level=list(range(len(fields)))
    ).filter(
        lambda x: (x.L < length_tr).sum() > num_data_points_tr
    )

    by_query = sliced_df.groupby(axis=0, level=list(fields))
    num_valid_groups = len(by_query.groups)

    print(
        f"found {num_valid_groups} groups satisfying the sequence length "
        f"and dataset size threshold"
    )
    queries_vals = list(by_query.groups.keys())
    queries = [dict(zip(df.index.names, q)) for q in queries_vals]

    sizes = by_query.apply(len)
    return queries, sizes


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--profile', type=str)
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        torch.set_default_device('cuda')

    profile = args.profile
    print(profile)
    p = get_profile(profile)
    print(p)
    ref_df = pd.read_csv(p["ref_fnm_with_length"])
    
    # get the topologies to test
    queries, sizes = get_topologies_within_req(
        ref_df,
        length_tr=100,
        num_data_points_tr=250,
        fields=CATH_FIELDS[:3]
    )
    # for query, size in list(zip(queries, sizes))[:1]:
    #     
    #     print (f'{size} chains in topology {query}')
    #     
    #     d = Dataset.from_cath_query(
    #         pdb_folder=p["pdb_chain_repo"],
    #         ckpt_fnm=p["ckpt_fnm"],
    #         esm2_rep_fnm=p["esm2_rep_fnm"],
    #         ref_df_path=p["ref_fnm"],
    #         with_caching=True,
    #         dask_cluster_args=p["dask_cluster"],
    #         cath_query=query,
    #         pmpnn_loading_method="pmpnn_notied",
    #         max_num_pdbs=None
    #     )
    #     _ = d.get_pdbs(format="pmpnn_notied")
    #     
    #     
    #     key = random.PRNGKey(0)
    #     torch.cuda.empty_cache()
    #     key, subkey = random.split(key)
    #     r, mmd = mmd_pmpnn(d, 0.1, subkey)

