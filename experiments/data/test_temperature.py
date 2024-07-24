# pyright: basic
from concurrent.futures import CancelledError
import json
from typing import Optional

import time
from pathlib import Path
import argparse
import pandas as pd
import torch
from jax import random
from kwgflows.rkhs.kernels import TensorProductKernel, gaussian_kernel
from pmpnn_extra.jax_pmpnn import JaxProteinMPNN
from pmpnn_extra.preprocessing import Dataset, get_cluster, select_pdbs
from utils import append_to_raw_results, get_profile, preload_dataset
from distributed import as_completed as distributed_as_completed

from calibration.statistical_tests.mmd import mmd_test



def mmd_pmpnn_diff_temp(
    profile_name,
    cath_query,
    temperature_true_model,
    temperature_wrong_model,
    key,
    torch_random_seed,
    upsample_to: Optional[int] = None,
):
    if torch.cuda.is_available():
        torch.set_default_device("cuda")

    p = get_profile(profile_name)

    d = Dataset.from_cath_query(
        pdb_folder=p["pdb_chain_repo"],
        ckpt_fnm=p["ckpt_fnm"],
        esm2_rep_fnm=p["esm2_rep_fnm"],
        ref_df_path=p["ref_fnm"],
        with_caching=True,
        dask_cluster_args=p["dask_cluster"],
        # cath_query={"Architecture Number": 15},
        # cath_query={"Topology Number": 8},
        cath_query=cath_query,
        pmpnn_loading_method="pmpnn_notied",
        max_workers=p['max_workers']
        # max_num_pdbs=None
    )

    _ = d.get_pdbs(format="pmpnn_notied")
    X, _, pdbs = d.get_XY()
    torch.cuda.empty_cache()
    batch_size = 20
    torch.manual_seed(torch_random_seed)

    if upsample_to is not None:
        num_unique_samples = X.shape[0]
        key, subkey = random.split(key)
        idxs = random.randint(subkey, (upsample_to,), 0, num_unique_samples)
        X = X[idxs]
        pdbs = select_pdbs(pdbs, idxs)

    # Y: true samples

    # in this exp, true model = PMPNN with temperature_true
    ground_truth_model = JaxProteinMPNN.create(
        pdbs,
        batching_method="native",
        temperature=temperature_true_model,
        return_esm2_embeddings=True,
        force_batching=True,
        batch_size=batch_size,
    )

    key, subkey = random.split(key)
    Z_1 = ground_truth_model.sample_from_conditional(key=subkey, num_samples=1)
    torch.cuda.empty_cache()

    # in this exp, true model = PMPNN with temperature_true
    # temperature_wrong_model = 1.0
    wrong_temp_model = JaxProteinMPNN.create(
        pdbs,
        batching_method="native",
        temperature=temperature_wrong_model,
        return_esm2_embeddings=True,
        force_batching=True,
        batch_size=batch_size,
    )

    key, subkey = random.split(key)
    Z_2 = wrong_temp_model.sample_from_conditional(key=subkey, num_samples=1)
    torch.cuda.empty_cache()

    k_x = gaussian_kernel.create()
    k_y = gaussian_kernel.create()
    tk = TensorProductKernel(k_x, k_y)
    mmd = mmd_test(tk, median_heuristic=True)

    key, subkey = random.split(key)
    r = mmd((X, Z_1), (X, Z_2), key=subkey)
    args = (cath_query, temperature_true_model, upsample_to, delta_t, torch_seed)
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
    default="result_temperature",
)


if __name__ == "__main__":
    cmdline_args = parser.parse_args()
    assert cmdline_args.profile != "", "Please specify a profile to use"

    if torch.cuda.is_available():
        torch.set_default_device("cuda")

    profile_name = cmdline_args.profile
    profile = get_profile(profile_name)

    with open(profile["cath_queries_file"], "r") as f:
        all_cath_queries = json.load(f)["queries"]


    cath_queries = all_cath_queries[:1]
    upsample_tos = [100, 500, 1000, 3000]
    delta_ts = [0.1][:1]
    # delta_ts = [0, 0.001, 0.01, 0.05][-1:]
    temperature_true_models = [0.1, 1.0]
    torch_seeds = list(range(10, 40))

    use_cluster = True
    if use_cluster:
        print("Using cluster")
        from dask.distributed import Client
        cluster = get_cluster(profile['dask_cluster_gpu'])
        client = Client(cluster)
    else:
        client = None


    for cath_query in cath_queries:
        _ = preload_dataset(profile_name, cath_query)


    futures = []
    rs = []
    args = []
    key = random.PRNGKey(0)

    for cath_query in cath_queries:
        for temperature_true_model in temperature_true_models:
            for upsample_to in upsample_tos:
                for delta_t in delta_ts:
                    for torch_seed in torch_seeds:
                        torch.cuda.empty_cache()
                        key, subkey = random.split(key)

                        torch_seed_randomized = random.randint(
                            subkey, (), 0, 100000
                        )

                        key, subkey = random.split(key)
                        if use_cluster:
                            assert client is not None
                            f = client.submit(
                                mmd_pmpnn_diff_temp,
                                profile_name,
                                cath_query,
                                temperature_true_model,
                                temperature_true_model + delta_t,
                                subkey,
                                torch_seed_randomized,
                                upsample_to=upsample_to,
                            )
                            futures.append(f)
                        else:
                            r, mmd, a = mmd_pmpnn_diff_temp(
                                profile_name,
                                cath_query,
                                temperature_true_model,
                                temperature_true_model + delta_t,
                                subkey,
                                torch_seed_randomized,
                                upsample_to=upsample_to,
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
                print(f"error: {fut}")
                rs.append(None)
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
        columns=["temperature_true_model", "upsample_to", "delta_t", "torch_seed"]
    )
    args_df_query = pd.DataFrame([a[0] for a in args])
    df_all = pd.concat([args_df_query, args_df_no_query, rs_df], axis=1)
    df_all = df_all.drop("h0_vals", axis=1)

    df_all.to_csv(
        results_path / f'{cmdline_args.result_filename_stem}.csv',
        mode="a"
    )
