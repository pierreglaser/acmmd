# Experiments for the paper Kernel-Based Evaluation of Conditional Biological Sequence Models

## Data Downloading and Preprocessing

**Important**: If you just want to reproduce the results of our synthetic simulation study,
you can skip this section.

To run the experiments, first create an entry with the name or your choice,
(using "user" in the example below) in the `config.json` and specified the
various data folders in which the data will be downloaded, saved, and 
preprocessed. The entry should have the following format:

```jsonc
{
    "user": {
        "ref_fnm": "./structures/cath_reference_files/cath-single_chain_domains-topology_number_count_gte10.csv",
        "complex_pdb_repo": "<your/path/to/complex_pdb_repo>",
        "pdb_chain_repo": "<your/path/to/pdb_chain_repo>",
        "ckpt_fnm": "<your/path/to/gearnet/checkpoints>/angle_gearnet_edge.pth",
        "outdir": "<your/path/to/gearnet_edge_graph_embeddings>",
        "dask_cluster":
            {
                "module": "<your dask cluster module>",
                "cls": "<your cluster type>"
                "kwargs": {
                    "n_workers": 2, // or more
                    "threads_per_worker": 1,
                    "memory_limit": "4GB",
                }
            }
    },
```

Once this is done, first download the pdbs using the command:
```bash
python batch_download_pdbs.py --profile=user
```
This script will download all the pdbs whose name are present in the list
specified in the `"complex_pdb_repo"` entry of the `"user"` profile of
the `config.json` file.


Then, run the command:
```bash
python batch_split_pdb.py  --profile=user
```
This script will split all the pdbs into single chains pdbs into
the the `"complex_pdb_repo"` entry of the `"user"` profile of
the `config.json` file.

Then, download the gearnet checkpoint weights using the command:
```bash
python download_gearnet_checkpoint.py --profile=user
```
which will download the pretrained gearnet model.

Finally, run
```bash
python -i ./get_gearnet_embeddings.py -- --max_num_pdbs 20 --profile=user
```
Which will (1) extract and save a single chain from each pdb as specified in
the `"pdb_chain_repo"` entry of the `config.json` file, and (2) compute the
gearnet embeddings for each chain and save them in the `"outdir"` entry of the
the `"user"` profile present in the `config.json` file. 


## Running the Experiments

To run the experiments for the simple simulation study (which does not involve embeddings and proteins), run

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  ipython synthetic_exp_composite.py
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform ipython synthetic_exp_calibration_composite.py
```

To run the synthetic experiments using ProteinMPNN models, run the following command:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python synthetic_exp_figures_composite.py  --type=calibration
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python synthetic_exp_figures_composite.py  --type=calibration
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  ipython -i test_temperature.py  -- --profile=user
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  ipython -i test_calibration.py  -- --profile=user
```

To run the whole dataset experiments, run

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  ipython -i test_mmd_on_true_dataset.py
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  ipython -i test_skce_on_true_dataset.py
```

To run the temperature study of ProteinMPNN models on true data, run

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  ipython -i test_mmd_optimal_temperature.py
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  ipython -i test_skce_optimal_temperature.py
```

## Reproducing the figures

Once these experiments are run, to reproduce the figures, follow the
instructions provided in the [README](./figures/README.md) file in the
`figures` folder.
