# pyright: basic
"""
Script for retrieving gearnet embeddings en masse from a 
repo of pdb_chains. TorchDrug cannot easily handle importing a select chain
from within a pdb. So a pdb containing only the chain that we are operating on
needs to be passed. If the specific pdb_chain file is not found, the code
will read in the full complex pdb from a preset pdb repo, and then output
a new file only containing the chain we are interested in.


"""
import copy
import json
from argparse import ArgumentParser
from pathlib import Path

from pmpnn_extra.preprocessing import encode_pdbs, extract_and_save_single_chains

from utils import resolve_args  # type: ignore



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ref_fnm")
    parser.add_argument("--complex_pdb_repo")
    parser.add_argument("--pdb_chain_repo")
    parser.add_argument("--ckpt_fnm")
    parser.add_argument("--outdir")
    parser.add_argument("--max_num_pdbs", type=int, default=100000)
    parser.add_argument("--profile", default=None)
    # XXX: smoke argument. not bothering with exposing it formally
    # now as it is dict-like.
    parser.add_argument("--dask_cluster", default=None)
    args = parser.parse_args()

    resolved_args = resolve_args(args)

    # extract_and_save_single_chains(resolved_args)
    # list all pdb files in pdb_chain_repo
    pdbs_files = list(Path(resolved_args.pdb_chain_repo).glob("*.pdb"))[:args.max_num_pdbs]
    print(f"Found {len(pdbs_files)} pdb files")
    encode_pdbs(
        pdbs_files,
        resolved_args.ckpt_fnm,
        resolved_args.dask_cluster,
        save=True,
        # outdir=resolved_args.outdir,
    )
    # # encoder = GearNetEncoder(args.ckpt_fnm)
    # # encodings.to_csv("encodings.csv")
