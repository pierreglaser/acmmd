# Accompaning Code for the ICML 2024 paper Kernel-Based Evaluation of Conditional Biological Sequence Models

## Setup instructions

Requirements: conda, an UNIX system (Linux or macOS)

If conda is not installed on your machine, do:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh -b -p ./mambaforge
bash -l
```

Then, to install the environment, do:

```bash
git clone git@github.com:acmmd-anonymous-authors/acmmd
cd seqmodels-calibration
conda env create -f environment.yml
conda activate seqmodels-calibration
```

At this point, it should be possible to launch the test suite:

```bash
python -m pytest -n auto ./tests
```

And to run the protein-mpnn scripts:

```bash
git submodule init
git submodule update
cd protein_mpnn/
python example_scoring.py
```

## Running the experiments

To produce the results of the paper, follow the instructions in the
experiments'folder [README.md](./experiments/data/README.md)
