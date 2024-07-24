#!/usr/bin/env bash
set -o errexit
set -o pipefail
set -o nounset

conda install -c conda-forge numpy scipy matplotlib
conda install -c conda-forge jax
pip install flax chex numpyro tensorflow-datasets

ln -s ~/.local/miniforge/ .lsp_symlink
