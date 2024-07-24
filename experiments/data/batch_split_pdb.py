import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    "--profile",
    type=str,
    default="",
)

parser.add_argument(
    "--n_jobs",
    type=int,
    default=100,
)


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.profile != "", "Please specify a profile to use"

    with open("config.json", "r") as f:
        config = json.load(f)[args.profile]

    complex_pdbs_folder = Path(config["complex_pdb_repo"])
    split_pdbs_folder = Path(config["pdb_chain_repo"])

    assert complex_pdbs_folder.exists(), "Please download the complex pdbs first"
    print("Splitting pdbs to", split_pdbs_folder)

    all_pdbs_files = complex_pdbs_folder.glob("*.pdb")
    all_pdbs_files = list(all_pdbs_files)

    all_pdbs_files = [str(f.absolute()) for f in all_pdbs_files]


    def split(file):
        p = subprocess.Popen(
            [
                "pdb_splitchain",
                str(file),
            ],
            cwd=split_pdbs_folder,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
        )
        # out, _ = p.communicate()
        try:
            return p.wait(timeout=1)
        except subprocess.TimeoutExpired:
            p.kill()
            return 1

    e = ThreadPoolExecutor(max_workers=args.n_jobs)
    e.map(split, all_pdbs_files)
    e.shutdown()
