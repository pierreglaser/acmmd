import json
import subprocess
from argparse import ArgumentParser

from pathlib import Path


parser = ArgumentParser()
parser.add_argument(
    '--profile',
    type=str,
    default='',
)

if __name__ == "__main__":
    args = parser.parse_args()

    assert args.profile != "", "Please specify a profile to use"

    with open("config.json", "r") as f:
        config = json.load(f)[args.profile]

    checkpoint_filename = config["ckpt_fnm"]
    Path(checkpoint_filename).parent.mkdir(parents=True, exist_ok=True)

    p = subprocess.Popen(
        [
            "curl",
            "https://zenodo.org/records/7593637/files/angle_gearnet_edge.pth",
            "-o",
            checkpoint_filename,
        ],
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
    )

    p.wait()
    print("Done")
