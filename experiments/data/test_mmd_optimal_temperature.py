# pyright: basic
from concurrent.futures import CancelledError
import json

import time
from pathlib import Path
import argparse
import pandas as pd
import torch
from jax import random
from kwgflows.rkhs.kernels import TensorProductKernel, gaussian_kernel
from pmpnn_extra.jax_pmpnn import JaxProteinMPNN
from pmpnn_extra.preprocessing import Dataset, get_cluster
from utils import append_to_raw_results, get_profile, preload_dataset
from distributed import as_completed as distributed_as_completed

from calibration.statistical_tests.mmd import mmd_test


def mmd_on_true_data(
    profile_name,
    temperature,
    num_samples,
    key,
    torch_random_seed,
):
    if torch.cuda.is_available():
        torch.set_default_device("cuda")

    p = get_profile(profile_name)

    d = Dataset.from_cath_query(
        pdb_folder=p["pdb_chain_repo"],
        ckpt_fnm=p["ckpt_fnm"],
        esm2_rep_fnm=p["esm2_rep_fnm"],
        ref_df_path=p["ref_fnm_limited_redundancy"],
        with_caching=True,
        dask_cluster_args=p["dask_cluster"],
        cath_query={},
        pmpnn_loading_method="pmpnn_notied",
        max_workers=p['max_workers'],
        max_num_pdbs=num_samples,
    )

    _ = d.get_pdbs(format="pmpnn_notied")
    X, Y, pdbs = d.get_XY()
    torch.cuda.empty_cache()
    batch_size = 20
    torch.manual_seed(torch_random_seed)

    # in this exp, true model = PMPNN with temperature_true
    model = JaxProteinMPNN.create(
        pdbs,
        batching_method="native",
        temperature=temperature,
        return_esm2_embeddings=True,
        force_batching=True,
        batch_size=batch_size,
    )

    key, subkey = random.split(key)
    Z = model.sample_from_conditional(key=subkey, num_samples=1)
    torch.cuda.empty_cache()

    k_x = gaussian_kernel.create()
    k_y = gaussian_kernel.create()
    tk = TensorProductKernel(k_x, k_y)
    mmd = mmd_test(tk, median_heuristic=True)

    key, subkey = random.split(key)
    r = mmd((X, Y), (X, Z), key=subkey)
    args = (temperature, torch_seed, num_samples)
    return r, mmd, args


parser = argparse.ArgumentParser()

parser.add_argument(
    "--profile",
    type=str,
    default="anon",
)

parser.add_argument(
    "--result_filename_stem",
    type=str,
    default="result_mmd_on_all_limited_redundancy_dataset",
)


if __name__ == "__main__":
    cmdline_args = parser.parse_args()
    assert cmdline_args.profile != "", "Please specify a profile to use"

    if torch.cuda.is_available():
        torch.set_default_device("cuda")

    profile_name = cmdline_args.profile
    profile = get_profile(profile_name)

    temperatures = [0.1, 1.0]
    torch_seeds = list(range(2))
    num_samples=1000

    use_cluster = True
    if use_cluster:
        print("Using cluster")
        from dask.distributed import Client
        cluster = get_cluster(profile['dask_cluster_gpu'])
        client = Client(cluster)
    else:
        client = None


    futures = []
    rs = []
    args = []
    key = random.PRNGKey(0)

    _ = preload_dataset(
        profile_name,
        {},
        ref_fnm=profile['ref_fnm_limited_redundancy']
    )

    for temperature in temperatures:
        for torch_seed in torch_seeds:
            torch.cuda.empty_cache()

            key, subkey = random.split(key)
            if use_cluster:
                assert client is not None
                f = client.submit(
                    mmd_on_true_data,
                    profile_name,
                    temperature,
                    num_samples,
                    subkey,
                    torch_seed,
                )
                futures.append(f)
            else:
                r, mmd, a = mmd_on_true_data(
                    profile_name,
                    temperature,
                    num_samples,
                    subkey,
                    torch_seed,
                )
                rs.append({
                    **r.to_dict(), "time": time.time(),
                })

                args.append(a)

    if use_cluster:
        num_completed = 0
        for fut, ret in distributed_as_completed(futures, with_results=True, raise_errors=False):
            num_completed += 1
            print(f"completed {num_completed}/{len(futures)}")
            if fut.status == "error":
                print("error")
            else:
                assert not isinstance(ret, CancelledError)
                r, mmd, a = ret
                rs.append({**r.to_dict(), "time": time.time()})
                args.append(a)


    results_path = Path("results")
    results_path.mkdir(exist_ok=True, parents=True)
    pkl_result_path = results_path / f'{cmdline_args.result_filename_stem}.pkl'
    append_to_raw_results(args, rs, pkl_result_path)


    rs_df = pd.DataFrame(rs)
    args_df_no_query = pd.DataFrame(
        [a[1:] for a in args],
        columns=["temperature", "torch_seed", "num_samples"],
    )
    args_df_query = pd.DataFrame([a[0] for a in args])
    df_all = pd.concat([args_df_query, args_df_no_query, rs_df], axis=1)
    df_all = df_all.drop("h0_vals", axis=1)

    df_all.to_csv(
        results_path / f'{cmdline_args.result_filename_stem}.csv',
        mode="a"
    )
