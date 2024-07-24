#!/bin/bash
#SBATCH -c 1                               # Required number of CPUs
#SBATCH -n 1
#SBATCH -t 0-6:00                         # Runtime in D-HH:MM format
#SBATCH -p short  # Adding more partitions for quick testing                         # Partition to run in
#SBATCH --mem=16G                         # Memory total in MiB (for all cores)
#SBATCH --mail-type=ALL
#SBATCH --job-name="dwn_pdbs"

#Job array specific
#SBATCH -o logs/%x_%j_%u.out
#SBATCH -e logs/%x_%j_%u.err


#load the correct modules
mkdir -p logs

module load gcc/6.2.0

set -e

#run command
RUNP=<path_to_project_root/structures>
INFILE="${RUNP}/reference_files/dwnfile_cath-single_chain_domains-topology_number_count_gte10.txt"
OUTDIR="${RUNP}/pdb_repo/"

#download all files
${RUNP}/batch_download_pdbs.sh -f ${INFILE} -p -o ${OUTDIR}

#unzip 
gunzip ${OUTDIR}/*.pdb.gz
