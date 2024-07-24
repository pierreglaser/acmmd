import copy
import json
import pickle
from pathlib import Path

import torch
from pmpnn_extra.preprocessing import Dataset


def get_profile(profile_name):
    with open(Path(__file__).parent / "config.json", "r") as f:
        profile = json.load(f)[profile_name]
    return profile


def resolve_args(args):
    args_dict = copy.copy(vars(args))
    profile_name = args_dict.pop("profile")
    if profile_name is not None:
        profile = get_profile(profile_name)
        for k, v in args_dict.items():
            if v is None:
                setattr(args, k, profile[k])

    else:
        setattr(
            args,
            "dask_cluster",
            {
                "module": "dask.distributed",
                "cls": "LocalCluster",
                "kwargs": {
                    "n_workers": 1,
                    "threads_per_worker": 1,
                    "memory_limit": "4GB",
                },
            },
        )
    return args


def preload_dataset(profile_name, cath_query, ref_fnm=None, max_num_pdbs=None):
    if torch.cuda.is_available():
        torch.set_default_device("cuda")

    p = get_profile(profile_name)
    d = Dataset.from_cath_query(
        pdb_folder=p["pdb_chain_repo"],
        ckpt_fnm=p["ckpt_fnm"],
        esm2_rep_fnm=p["esm2_rep_fnm"],
        ref_df_path=p["ref_fnm"] if ref_fnm is None else ref_fnm,
        with_caching=True,
        dask_cluster_args=p["dask_cluster"],
        # cath_query={"Architecture Number": 15},
        # cath_query={"Topology Number": 8},
        cath_query=cath_query,
        pmpnn_loading_method="pmpnn_notied",
        max_workers=p['max_workers'],
        # max_num_pdbs=100
        max_num_pdbs=None
    )
    _ = d.get_pdbs(format="pmpnn_notied")
    X, _, _ = d.get_XY()
    return len(X)


def append_to_raw_results(args, rs, pkl_result_path):
    if pkl_result_path.exists():
        with open(pkl_result_path, "rb") as f:
            prev_args, prev_rs = pickle.load(f)
        all_args_pkl = prev_args + args
        all_rs_pkl = prev_rs + rs
        with open(pkl_result_path, "wb") as f:
            pickle.dump((all_args_pkl, all_rs_pkl), f)

    else:
        with open(pkl_result_path, "wb") as f:
            pickle.dump((args, rs), f)
