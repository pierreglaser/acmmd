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

    out_file = Path(config["complex_pdb_repo"])
    out_file.mkdir(exist_ok=True, parents=True)
    print("Downloading to", out_file)

    this_file_dir = Path(__file__).parent.absolute()
    all_pdbs_file = (
        this_file_dir
        / "structures/cath_reference_files/dwnfile_cath-single_chain_domains-topology_number_count_gte10.txt"
    )

    # split this file into parts of `batch_size` tokens each
    with open(all_pdbs_file, "r") as f:
        pdbs = f.readlines()[0].split(",")

    num_pdbs = len(pdbs)
    batch_size = 50
    num_splits = num_pdbs // batch_size + 1

    split_pdbs_dir = this_file_dir / "structures" / "cath_reference_files" / "split_pdbs"
    split_pdbs_dir.mkdir(exist_ok=True)

    print("Splitting into", num_splits, "parts")
    for i in range(num_splits):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_pdbs)
        split_pdbs = pdbs[start:end]
        split_pdbs = ",".join(split_pdbs)

        with open(split_pdbs_dir / f"split_pdbs_{i}.txt", "w") as f:
            f.write(split_pdbs)

    cmd = this_file_dir / "batch_download_pdbs.sh"

    def download_batch_pdbs(split_pdbs_file):
        p = subprocess.Popen(
            [
                cmd,
                "-f",
                str(split_pdbs_file),
                "-p",
                "-o",
                str(out_file),
            ],
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
        )
        # out, _ = p.communicate()
        return p.wait()

    e = ThreadPoolExecutor(max_workers=args.n_jobs)
    futures = []
    for i in range(num_splits):
        futures.append(
            e.submit(download_batch_pdbs, split_pdbs_dir / f"split_pdbs_{i}.txt")
        )

    num_batch_completed = 0
    for future in as_completed(futures):
        num_batch_completed += 1
        print("completed", num_batch_completed, "of", num_splits)
        assert future.result() == 0
    e.shutdown()

    # clear the split_pdbs_dir
    for i in range(num_splits):
        (split_pdbs_dir / f"split_pdbs_{i}.txt").unlink()

    # remove the split_pdbs_dir
    split_pdbs_dir.rmdir()

    all_pdbs_files = out_file.glob("*.pdb.gz")
    all_pdbs_files = list(all_pdbs_files)

    def gunzip(file):
        p = subprocess.Popen(
            [
                "gunzip",
                str(file),
            ],
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
        )
        # out, _ = p.communicate()
        return p.wait()

    e = ThreadPoolExecutor(max_workers=args.n_jobs)
    e.map(gunzip, all_pdbs_files)
    e.shutdown()
