# pyright: basic
import argparse
import json
import time
from concurrent.futures import CancelledError
from pathlib import Path

import pandas as pd
import torch
from distributed import as_completed as distributed_as_completed
from jax import random
from kwgflows.rkhs.kernels import gaussian_kernel
from pmpnn_extra.jax_pmpnn import JaxProteinMPNN
from pmpnn_extra.preprocessing import Dataset, get_cluster
from utils import append_to_raw_results, get_profile, preload_dataset

from calibration.kernels import ExpMMDKernel
from calibration.statistical_tests.skce import skce_test


def skce_on_true_data(
    profile_name,
    cath_query,
    temperature_true_model,
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
        ref_df_path=p["ref_fnm"],
        with_caching=True,
        dask_cluster_args=p["dask_cluster"],
        # cath_query={"Architecture Number": 15},
        cath_query=cath_query,
        pmpnn_loading_method="pmpnn_notied",
        max_workers=p["max_workers"],
        # max_num_pdbs=None
    )
    _ = d.get_pdbs(format="pmpnn_notied")
    _, Y, pdbs = d.get_XY()
    torch.cuda.empty_cache()
    batch_size = 20
    torch.manual_seed(torch_random_seed)

    ground_truth_model = JaxProteinMPNN.create(
        pdbs,
        # batching_method="native",
        batching_method="passhtrough_batched",
        temperature=temperature_true_model,
        return_esm2_embeddings=True,
        batch_size=batch_size,
    )

    torch.cuda.empty_cache()

    k_p = ExpMMDKernel.create(ground_space_kernel=gaussian_kernel.create(sigma=1.0))
    k_y = gaussian_kernel.create()

    skce = skce_test(
        x_kernel=k_p,
        y_kernel=k_y,
        median_heuristic=True,
        approximate=True,
        approximation_num_particles=3,
    )

    key, subkey = random.split(key)
    r = skce(ground_truth_model, Y, key=subkey)
    args = (cath_query, temperature, torch_seed)
    return r, skce, args


parser = argparse.ArgumentParser()

parser.add_argument(
    "--profile",
    type=str,
    default="anon",
)

parser.add_argument(
    "--result_filename_stem",
    type=str,
    default="result_skce_on_true_data",
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

    cath_queries = all_cath_queries
    temperatures = [0.1, 1.0]
    torch_seeds = list(range(2))

    use_cluster = True
    if use_cluster:
        print("Using cluster")
        from dask.distributed import Client

        cluster = get_cluster(profile["dask_cluster_gpu"])
        client = Client(cluster)
    else:
        client = None

    futures = []
    rs = []
    args = []
    key = random.PRNGKey(0)

    for cath_query in cath_queries:
        _ = preload_dataset(profile_name, cath_query)
        for temperature in temperatures:
            for torch_seed in torch_seeds:
                torch.cuda.empty_cache()

                key, subkey = random.split(key)
                if use_cluster:
                    assert client is not None
                    f = client.submit(
                        skce_on_true_data,
                        profile_name,
                        cath_query,
                        temperature,
                        subkey,
                        torch_seed,
                    )
                    futures.append(f)
                else:
                    r, skce, a = skce_on_true_data(
                        skce_on_true_data,
                        cath_query,
                        temperature,
                        subkey,
                        torch_seed,
                    )
                    rs.append(
                        {
                            **r.to_dict(),
                            "time": time.time(),
                        }
                    )
                    args.append(a)

    if use_cluster:
        num_completed = 0
        for fut, ret in distributed_as_completed(
            futures, with_results=True, raise_errors=False
        ):
            num_completed += 1
            print(f"completed {num_completed}/{len(futures)}")
            if fut.status == "error":
                print("error")
            else:
                assert not isinstance(ret, CancelledError)
                r, skce, a = ret
                rs.append({**r.to_dict(), "time": time.time()})
                args.append(a)

    results_path = Path("results")
    results_path.mkdir(exist_ok=True, parents=True)
    pkl_result_path = results_path / f"{cmdline_args.result_filename_stem}.pkl"
    append_to_raw_results(args, rs, pkl_result_path)

    rs_df = pd.DataFrame(rs)
    args_df_no_query = pd.DataFrame(
        [a[1:] for a in args],
        columns=["temperature", "torch_seed"],
    )
    args_df_query = pd.DataFrame([a[0] for a in args])
    df_all = pd.concat([args_df_query, args_df_no_query, rs_df], axis=1)
    df_all = df_all.drop("h0_vals", axis=1)

    df_all.to_csv(results_path / f"{cmdline_args.result_filename_stem}.csv", mode="a")
