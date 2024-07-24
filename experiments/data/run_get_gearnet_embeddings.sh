#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-08:00                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=16G                         # Memory total in MiB (for all cores)
#SBATCH --mail-type=ALL
#SBATCH --job-name="gearnet_embed"

#Job array specific
#SBATCH --array=1-5       #Job array end is inclusive
#SBATCH -o logs/%x_%A_%3a_%j_%u.out
#SBATCH -e logs/%x_%A_%3a_%j_%u.err

#load the correct modules
mkdir -p logs

module load gcc/6.2.0 
source activate <path_to_conda_env>
RUNPY=<path_to_conda_env>/bin/python


PRE=<path_to_project_root>
$RUNPY ${PRE}/structure_kernels/gearnet/get_gearnet_embeddings.py \
    # XXX: my changes broke this code 
    --ref_fnm ${PRE}/structures/reference_files/split_files/split${SLURM_ARRAY_TASK_ID}_cath-single_chain_domains-topology_number_count_gte10.csv \
    --complex_pdb_repo ${PRE}/structures/pdb_repo \
    --pdb_chain_repo ${PRE}/structures/pdb_chain_repo \
    --ckpt_fnm ${PRE}/structure_kernels/gearnet/checkpoints/angle_gearnet_edge.pth \
    --outdir ${PRE}/structure_kernels/gearnet/gearnet_edge_graph_embeddings
