# pyright: basic
"""
Utilities for retrieving gearnet embeddings en masse from a
repo of pdb_chains. TorchDrug cannot easily handle importing a select chain
from within a pdb. So a pdb containing only the chain that we are operating on
needs to be passed. If the specific pdb_chain file is not found, the code
will read in the full complex pdb from a preset pdb repo, and then output
a new file only containing the chain we are interested in.
"""

import jax
import copy
import importlib
import json
import os
import subprocess
import time as time
from typing_extensions import ParamSpec
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union, cast, overload
import math
import joblib

import jax.numpy as jnp
from joblib import Memory
import numpy as np
import pandas as pd
import torch
import tqdm
from Bio.PDB import PDBIO  # pyright: ignore[reportPrivateImportUsage]
from Bio.PDB import PDBParser  # pyright: ignore[reportPrivateImportUsage]
from dask.distributed import Client
from distributed import as_completed as distributed_as_completed
from torchdrug import data, layers, models
from torchdrug.data.protein import PackedProtein
from torchdrug.layers import geometry
from jax import tree_map

import torch
import esm

from pmpnn_extra.protein_file_utils import protein_structure_from_file, protein_structure_from_files, tfOutputTuple
from pmpnn_extra.pytorch2jax import as_jax_pytree

GEARNET_MODEL = None
CKPT_FMT = None
PMPNN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

ESM2_MAX_SEQ_LEN = 100
PMPNN_PDB_MAX_SEQ_LEN = 100

PDB_MAX_NUM_LINES = 3000

# chain_name_to_num = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ6f", range(28)))  # 6 and f somewhat is used as a chain name for 1 pdb
# chain_num_to_name = dict(zip(range(28), "ABCDEFGHIJKLMNOPQRSTUVWXYZ6f"))

chain_name_to_num = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", range(62)))
chain_num_to_name = dict(zip(range(62), "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"))



def replace_chain_names_with_nums(pdb):
    pdb = pdb._replace(
        chain_list_list=[
            [chain_name_to_num[c] for c in chain_list]
            for chain_list in pdb.chain_list_list
        ]
    )
    pdb = pdb._replace(
        masked_list_list=[
            [chain_name_to_num[c] for c in chain_list]
            for chain_list in pdb.masked_list_list
        ]
    )
    return pdb


def select_pdbs(pdbs, idxs):
    jax_pmpnn_pdbs = pmpnn_pdb_as_pytree(pdbs)
    jax_pmpnn_pdbs_subsampled = jax.tree_map(lambda x: x[idxs], jax_pmpnn_pdbs)
    selected_pdbs = jax_pmpnn_pdb_as_torch_pdb(jax_pmpnn_pdbs_subsampled)
    return selected_pdbs



def pmpnn_pdb_as_pytree(pdb):
    pdb = replace_chain_names_with_nums(pdb)
    pdb = tree_map(lambda x: jnp.array(x.detach().clone().cpu().numpy()) if isinstance(x, torch.Tensor) else jnp.array(x), pdb, is_leaf=lambda x: isinstance(x, list))
    pdb = tree_map(as_jax_pytree, pdb)
    return pdb

def jax_pmpnn_pdb_as_torch_pdb(pdb):
    import jax
    pdb = replace_chain_num_with_names(pdb)
    pdb = tree_map(
        lambda x: torch.from_numpy(np.array(x)) if isinstance(x, jax.Array) else x, pdb, is_leaf=lambda x: isinstance(x, list))
    return pdb

def replace_chain_num_with_names(pdb):
    pdb = pdb._replace(
        chain_list_list=[
            [chain_num_to_name[int(c)] for c in jnp.atleast_1d(chain_list)]
            for chain_list in pdb.chain_list_list
        ]
    )
    pdb = pdb._replace(
        masked_list_list=[
            [chain_num_to_name[int(c)] for c in jnp.atleast_1d(mask_list)]
            for mask_list in pdb.masked_list_list
        ]
    )
    return pdb


def get_num_lines(file):
    num_lines = subprocess.check_output(
        ["wc", "-l", file]
    ).decode("utf-8")
    num_lines = int(num_lines.split()[0])
    return num_lines


def import_pdb(pdb_file, max_num_lines=None):
    # import protein and format
    # don't load if num. lines greater than max_num_lines
    # (this is a hack to avoid loading in the full complex)
    if max_num_lines is not None:
        num_lines = get_num_lines(pdb_file)
        print("num_lines", num_lines)
        if num_lines > max_num_lines:
            raise ValueError("too many lines in pdb file")

    protein = data.Protein.from_pdb(
        str(pdb_file),
        atom_feature="position",
        bond_feature="length",
        residue_feature="symbol",
    )
    _protein = data.Protein.pack([protein])
    graph_construction_model = layers.GraphConstruction(
        node_layers=[geometry.AlphaCarbonNode()],
        edge_layers=[
            geometry.SpatialEdge(radius=10, min_distance=5),
            geometry.KNNEdge(k=10, min_distance=5),
            geometry.SequentialEdge(max_distance=2),
        ],
        edge_feature="gearnet",
    )
    protein_ = graph_construction_model(_protein)
    protein_.view = "residue"
    return protein_


def extract_and_save_single_chains(args):
    # arrange
    if not os.path.isdir(args.pdb_chain_repo):
        os.makedirs(args.pdb_chain_repo)

    # import the reference file
    ref_df = pd.read_csv(args.ref_fnm)

    for idx, pdb_name, chain_name in tqdm.tqdm(
        list(ref_df[["pdb", "chain"]].itertuples())[: args.max_num_pdbs]
    ):
        pdb_file = os.path.join(args.pdb_chain_repo, f"{pdb_name}_{chain_name}.pdb")
        with open(pdb_file, "w") as pdb_file:
            p = subprocess.Popen(
                [
                    "pdb_selchain",
                    f"-{chain_name}",
                    f"{os.path.join(args.complex_pdb_repo, pdb_name)}.pdb",
                ],
                stdout=pdb_file,
                stderr=subprocess.PIPE,
            )
            out, err = p.communicate()
            if err:
                print("an error occured for pdb", pdb_name)
                # print(err)
                
def AAseq_from_pmpnn_seq(S, pad_value=99):
    PMPNN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
    Ss = []
    for i,s in enumerate(S):
        #remove padding
        sp = s[s!=pad_value]
        AA_s = ''.join([PMPNN_ALPHABET[s] for s in sp])
        Ss.append((i, AA_s))
    return (Ss)


class GearNetEncoder:
    def __init__(self, ckpt_fnm, agg_func="mean"):
        # init model
        self.gearnet_edge = models.GearNet(
            input_dim=21,
            hidden_dims=[512, 512, 512, 512, 512, 512],
            num_relation=7,
            edge_input_dim=59,
            num_angle_bin=8,
            batch_norm=True,
            concat_hidden=True,
            short_cut=True,
            readout=agg_func,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device for structure encoding:")
        print(device)
        self.device = device
        net = torch.load(ckpt_fnm, map_location=device)
        self.gearnet_edge.load_state_dict(net)
        self.gearnet_edge.eval()
        self.gearnet_edge = self.gearnet_edge.to(self.device)

        print(f"Initialized GearNet edge encoder from {ckpt_fnm}")
        print(f"Using {agg_func} for graph representations")

    def encode_(self, protein_):
        with torch.no_grad():
            output = self.gearnet_edge(
                protein_.to(self.device), protein_.node_feature.float().to(self.device),
                all_loss=None, metric=None
            )

        return output["graph_feature"]


def get_gearnet_model(ckpt_fnm):
    """
    GearNet model loading function that avoids having to send/load the gearnet
    model at every new function call when from within a multiprocessing worker.
    """
    global GEARNET_MODEL
    global CKPT_FMT
    if GEARNET_MODEL is None:
        assert CKPT_FMT is None
        encoder = GearNetEncoder(ckpt_fnm)
        GEARNET_MODEL = encoder
        CKPT_FMT = ckpt_fnm
    else:
        assert CKPT_FMT == ckpt_fnm

    return GEARNET_MODEL


def get_encoding_df_path(ckpt_fnm, folder, return_filename=False):
    df_filename = Path(str(ckpt_fnm).lstrip("/.").replace("/", "_")).with_suffix(".csv")
    return folder / df_filename


def get_encoding_df(ckpt_fnm, folder):
    path = get_encoding_df_path(ckpt_fnm, folder)
    assert path.exists()
    print(f"Loading encoding dataframe at {path}")
    return pd.read_csv(path, index_col=[0, 1])


def get_cluster(dask_cluster_args=None):
    if dask_cluster_args is None:
        dask_cluster_args = {
            "module": "dask.distributed",
            "cls": "LocalCluster",
            "kwargs": {
                "n_workers": 1,
                "threads_per_worker": 1,
                "memory_limit": "4GB",
            },
        }
    cluster_config = copy.deepcopy(dask_cluster_args)
    n_workers = cluster_config["kwargs"].pop("n_workers")
    print (cluster_config)
    cluster = getattr(
        importlib.import_module(cluster_config["module"]), cluster_config["cls"]
    )(n_workers=0, **cluster_config["kwargs"])

    print(f"scaling cluster to {n_workers} workers")
    cluster.scale(n_workers)

    # quick test
    print(cluster)
    print("testing cluster")
    client = Client(cluster)
    _ret = client.submit(lambda x: x + 1, 1).result()
    print("done")
    client.close()
    return cluster


def encode_pdb(pdb_file, ckpt_fnm):
    encoder = get_gearnet_model(ckpt_fnm)
    pdb = import_pdb(pdb_file, max_num_lines=PDB_MAX_NUM_LINES)
    graph_encoding = encoder.encode_(pdb).cpu().numpy()
    return graph_encoding


def encode_pdbs_sequential(pdb_files, ckpt_fnm):
    encodings = []
    output_dim = cast(
        int,
        get_gearnet_model(ckpt_fnm).gearnet_edge.output_dim,
    )

    nan_placeholder = np.nan * np.ones((output_dim,))
    for pdb_file in pdb_files:
        try:
            encodings.append(encode_pdb(str(pdb_file), ckpt_fnm)[0])
        except Exception as e:
            encodings.append(nan_placeholder)

    names_and_chains = [p.with_suffix("").name for p in pdb_files]
    df = pd.DataFrame(encodings, index=names_and_chains)
    return df

def encode_sequences_sequential(sequences, model=None, alphabet=None, batch_converter=None):
    '''
    sequences -> list( tuple(seq_name, AA_sequence) ...)
                        seq_name = {pdb}_{chain}
    '''
    encodings = []
    #output_dim = 1280
    
    #init model
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        t0 = time.time()

        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval() 
        
        print("Using device for sequence encoding:")
        print(device)
        model = model.to(device)

        print("time loading model", time.time() - t0)
    else:
        assert alphabet is not None
        assert batch_converter is not None
    
    
    # Extract per-residue representations (on CPU)
    batch_size = 1
    assert batch_size == 1
    print("Encoding sequences using batch size", batch_size)
    num_batches = len(sequences) // batch_size + int(len(sequences) % batch_size != 0)


    with torch.no_grad():
        sequence_representations = []
        for batch_no in range(num_batches):
            print(f"Processing batch {batch_no + 1} / {num_batches}")
            seq_batch = sequences[batch_no * batch_size : (batch_no + 1) * batch_size]

            #prepare data
            batch_labels, batch_strs, batch_tokens = batch_converter(seq_batch)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            batch_tokens = batch_tokens.to(device)


            # print(batch_tokens.shape)
            num_tokens = len(batch_tokens[0])
            print(f"num tokens: {num_tokens}")
            if num_tokens > ESM2_MAX_SEQ_LEN:
                print('discarding this batch -- too big')
                assert batch_size == 1
                sequence_representations.extend([torch.nan * torch.ones((1280,))])
            else:
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                token_representations = results["representations"][33]
                # print(token_representations.shape)
                # Generate per-sequence representations via averaging
                # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
                sequence_representations_batch = []
                for i, tokens_len in enumerate(batch_lens):
                    print("encoding sequence no.", i)
                    sequence_representations_batch.append(token_representations[i, 1 : tokens_len - 1].mean(0))
                sequence_representations.extend(sequence_representations_batch)

    #unpack the representations
    seq_rep = [s.cpu().numpy() for s in sequence_representations]
    seq_names = [t[0] for t in sequences]
    df = pd.DataFrame(seq_rep, index=seq_names)
    del sequence_representations
    return df

def encode_pdbs(
    pdbs_files,
    ckpt_fnm,
    dask_cluster_args,
    save=False,
    outfnm=None,
    use_cluster=True
):
    # parallelism does not help because
    # workers are fighting for limited resources.
    if save:
        assert isinstance(outfnm, str)
        outdir = os.path.dirname(outfnm)
        Path(outdir).mkdir(exist_ok=True, parents=True)
        encoding_df_path = Path(outfnm)
        if encoding_df_path.exists() & (save != 'force'):
            print ('Opening gearnet encodings from file')
            encoding_df = pd.read_csv(encoding_df_path, index_col=0, header=None)
            return (encoding_df)
        elif encoding_df_path.exists() & (save == 'force'):
            #reset the file
            os.remove(encoding_df_path)
        #assert not encoding_df_path.exists()
    else:
        encoding_df_path = None
        
    if use_cluster:

        cluster = get_cluster(dask_cluster_args)
        client = Client(cluster)
        fs = []

        n_workers = int(dask_cluster_args['kwargs']['n_workers'])
        n_split = math.ceil(len(pdbs_files) / 50)
        pdbs_files_arr = np.array_split(pdbs_files, n_split)

        for pdb_files_b in tqdm.tqdm(pdbs_files_arr):
            f = client.submit(encode_pdbs_sequential, pdb_files_b, ckpt_fnm)
            fs.append(f)

        num_completed = 0
        all_encodings = []
        for _, encodings in distributed_as_completed(fs, with_results=True):
            num_completed += 1
            print(f"completed {num_completed}/{len(fs)}")
            if isinstance(encodings, pd.DataFrame):
                all_encodings.append(encodings)
                if save:
                    assert encoding_df_path is not None
                    encodings.to_csv(
                        encoding_df_path,
                        index=True,
                        mode="a", header=False
                    )
            else:
                print(f"encodings computation failed for one batch")

        client.close()
        cluster.close()
        all_encodings = pd.concat(all_encodings)
        return all_encodings
    
    else:
        print ('Encoding structures off cluster')
        all_encodings = encode_pdbs_sequential(pdbs_files, ckpt_fnm)
        all_encodings.to_csv(encoding_df_path)
        print ('Done')
        return (all_encodings)


def encode_sequences(
    sequences, #list(tuple(seqname, AAseq))
    dask_cluster_args,
    save=False,
    outfnm=None,
    use_cluster=True
):
    # parallelism does not help because
    # workers are fighting for limited resources.
    if save:
        assert isinstance(outfnm, str)
        outdir = os.path.dirname(outfnm)
        Path(outdir).mkdir(exist_ok=True, parents=True)
        encoding_df_path = Path(outfnm)
        if encoding_df_path.exists() & (save != 'force'):
            print ('Opening esm2 encodings from file')
            encoding_df = pd.read_csv(encoding_df_path, index_col=0, header=None)
            return (encoding_df)
        elif encoding_df_path.exists() & (save == 'force'):
            #reset the file
            os.remove(encoding_df_path)
        #assert not encoding_df_path.exists()
    else:
        encoding_df_path = None
        
    if use_cluster:

        cluster = get_cluster(dask_cluster_args)
        print (dask_cluster_args)
        client = Client(cluster)
        fs = []

        n_workers = int(dask_cluster_args['kwargs']['n_workers'])
        n_split = np.min([len(sequences) , (len(sequences) // n_workers) + 1])

        n_split = math.ceil(len(sequences) / 50)

        sequences_arr = np.array_split(sequences, n_split)

        for sequences in tqdm.tqdm(sequences_arr):
            f = client.submit(encode_sequences_sequential, sequences)
            fs.append(f)

        num_completed = 0
        all_encodings = []
        for _, encodings in distributed_as_completed(fs, with_results=True):
            num_completed += 1
            print(f"completed {num_completed}/{len(fs)}")
            if isinstance(encodings, pd.DataFrame):
                all_encodings.append(encodings)
                if save:
                    assert encoding_df_path is not None
                    encodings.to_csv(
                        encoding_df_path,
                        index=True,
                        mode="a", header=False
                    )
            else:
                print(f"encodings computation failed for one batch")

        client.close()
        cluster.close()
        all_encodings = pd.concat(all_encodings)
        return all_encodings
    
    else:
        print ('Encoding sequences off cluster')
        all_encodings = encode_sequences_sequential(sequences)
        all_encodings.to_csv(encoding_df_path)
        print ('Done')
        return (all_encodings)


this_folder = Path(__file__).parent

memory = Memory(location="cache", verbose=0)



def extract_sequences_from_pmpnn_pdbs(tfs, names):
    Ss = []
    for i,l in enumerate(tfs.lengths):
        S = tfs.S[i][:l]
        AA_S = ''.join([PMPNN_ALPHABET[s] for s in S])
        Ss.append((names[i], AA_S))
        
    return (Ss)

def _pdb_files_from_ref_df(ref_df, pdb_folder, return_queried_files=False):
    pdb_files = []
    queried_files = []
    for idx, pdb_name, chain_name in tqdm.tqdm(
        list(ref_df[["pdb", "chain"]].itertuples())
    ):
        pdb_file = Path(pdb_folder) / f"{pdb_name}_{chain_name}.pdb"
        if not pdb_file.exists():
            print(f"warning: pdb file {pdb_file} does not exist")
        else:
            pdb_files.append(pdb_file)
        queried_files.append(pdb_file)
    if return_queried_files:
        return pdb_files, queried_files
    else:
        return pdb_files

def _query_tuple_from_dict(d: Dict) -> Tuple:
    return tuple(d.items())

def _get_ref_df_slice_from_query(ref_df_path:str , query_tuple: tuple, max_num_pdbs: Optional[int] = None) -> pd.DataFrame:
    ref_df = pd.read_csv(ref_df_path)

    ref_df_slice = ref_df.copy()
    if len(query_tuple) > 0:
        for (k, v) in query_tuple:
            ref_df_slice = ref_df_slice[ref_df_slice[k] == v]

    ref_df_slice = cast(pd.DataFrame, ref_df_slice)
    if max_num_pdbs is not None:
        return ref_df_slice.head(max_num_pdbs)
    else:
        return ref_df_slice


P = ParamSpec("P")
T = TypeVar("T")


def make_cache_wrapper(memory: joblib.Memory):
    def cache(f: Callable[P, T], ignore=None) -> Callable[P, T]:
        return cast(Callable[P, T], memory.cache(f, ignore=ignore))

    return cache


cache = make_cache_wrapper(memory)


def cacheable_get_pmpnn_pdbs_from_query(ref_df_path: str, query_tuple: tuple, pdb_folder, max_num_pdbs: Optional[int] = None):
    ref_df = _get_ref_df_slice_from_query(ref_df_path, query_tuple, max_num_pdbs)
    pdbs_files = _pdb_files_from_ref_df(ref_df, pdb_folder)
    _pdbs_pmpnn_format = protein_structure_from_files(
        [str(f) for f in pdbs_files]
    )
    return _pdbs_pmpnn_format

def cacheable_get_pmpnn_pdbs_from_query_notied(ref_df_path: str, query_tuple, pdb_folder, max_workers=16, max_num_pdbs: Optional[int] = None):
    ref_df = _get_ref_df_slice_from_query(ref_df_path, query_tuple, max_num_pdbs)
    pdbs_files, queried_files = _pdb_files_from_ref_df(ref_df, pdb_folder, return_queried_files=True)
    pdbs =[]
    if max_workers == 1:
        print('loading pdbs sequentially')
        for f in pdbs_files:
            print(f"loading {f}")
            pdb = protein_structure_from_file(str(f))
            pdbs.append(pdb)
    else:
        from multiprocessing import get_context
        context = get_context('spawn')
        from concurrent.futures import ProcessPoolExecutor
        assert max_workers > 1
        print(f'loading pdbs in parallel using {max_workers} workers')
        e = ProcessPoolExecutor(max_workers=max_workers, mp_context=context)
        pdbs = list(e.map(protein_structure_from_file, list(map(str, pdbs_files))))
    ret, is_selected = concat_pdbs(pdbs)   # type: ignore
    # account for nonexisting files in is_selected
    broader_is_selected = []
    for f in queried_files:
        if f in pdbs_files:
            broader_is_selected.append(is_selected[pdbs_files.index(f)])
        else:
            broader_is_selected.append(False)
    return ret, broader_is_selected

def to_padded_tensor(list_of_tensors, padding_value):
    v = torch.nested.nested_tensor(list_of_tensors)
    return torch.nested.to_padded_tensor(v, padding=padding_value)


def concat_pdbs(pdbs: List[tfOutputTuple]):
    pmpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    padded_value=0

    X_padding_value = 0
    S_padding_value = 0
    residue_idx_padding_value = -100
    num_pdbs = len(pdbs)
    is_selected = [True if len(p.S[0]) < PMPNN_PDB_MAX_SEQ_LEN else False for p in pdbs]
    pdbs = [p for p in pdbs if len(p.S[0]) < PMPNN_PDB_MAX_SEQ_LEN]
    print(f"kept {len(pdbs)} out of {num_pdbs} -- the others exceeded the length thresdhold")

    concatenated_pdbs = tfOutputTuple(
        X=to_padded_tensor([pdb.X[0] for pdb in pdbs], padding_value=X_padding_value),
        S=to_padded_tensor([pdb.S[0] for pdb in pdbs], padding_value=S_padding_value),
        mask=to_padded_tensor([pdb.mask[0] for pdb in pdbs], padding_value=padded_value),
        lengths=np.array([pdb.lengths[0] for pdb in pdbs]),
        chain_M=to_padded_tensor([pdb.chain_M[0] for pdb in pdbs], padding_value=0),
        chain_encoding_all=to_padded_tensor([pdb.chain_encoding_all[0] for pdb in pdbs], padding_value=0),
        chain_list_list=[pdb.chain_list_list[0] for pdb in pdbs],
        visible_list_list=[pdb.visible_list_list[0] for pdb in pdbs],
        masked_list_list=[pdb.masked_list_list[0] for pdb in pdbs],
        masked_chain_length_list_list=[pdb.masked_chain_length_list_list[0] for pdb in pdbs],
        chain_M_pos=to_padded_tensor([pdb.chain_M_pos[0] for pdb in pdbs], padding_value=0),
        residue_idx=to_padded_tensor([pdb.residue_idx[0] for pdb in pdbs], padding_value=-100),
        omit_AA_mask=to_padded_tensor([pdb.omit_AA_mask[0] for pdb in pdbs], padding_value=0),
        dihedral_mask=to_padded_tensor([pdb.dihedral_mask[0] for pdb in pdbs], padding_value=0),
        tied_pos_list_of_lists_list=[pdb.tied_pos_list_of_lists_list[0] for pdb in pdbs],
        # size: (1, num_residues, len(pmpnn_alphabet))
        pssm_coef=to_padded_tensor([pdb.pssm_coef[0] for pdb in pdbs], padding_value=0.),
        # size: (1, num_residues, len(pmpnn_alphabet))
        pssm_bias=to_padded_tensor([pdb.pssm_bias[0] for pdb in pdbs], padding_value=0.),
        pssm_log_odds_all=to_padded_tensor([pdb.pssm_log_odds_all[0] for pdb in pdbs], padding_value=0.),
        # size: (1, num_residues, len(pmpnn_alphabet))
        bias_by_res_all=to_padded_tensor([pdb.bias_by_res_all[0] for pdb in pdbs], padding_value=0.),
        tied_beta=to_padded_tensor([pdb.tied_beta for pdb in pdbs], padding_value=0.),
    )
    return concatenated_pdbs, is_selected


def cacheable_get_gearnet_pdbs_from_query(ref_df_path: str, query_tuple:tuple, pdb_folder, max_num_pdbs: Optional[int] = None):
    ref_df = _get_ref_df_slice_from_query(ref_df_path, query_tuple, max_num_pdbs)
    pdbs_files = _pdb_files_from_ref_df(ref_df, pdb_folder)
    pdbs = []
    for pdb_file in pdbs_files:
        print(f"loading {pdb_file}")
        try:
            pdb = import_pdb(pdb_file, max_num_lines=PDB_MAX_NUM_LINES)
        except ValueError:
            pdb = None

        pdbs.append(pdb)
    return pdbs


def cacheable_get_gearnet_encodings_from_query(
    ref_df_path:str, query_tuple, pdb_folder, ckpt_fnm, dask_cluster_args, max_num_pdbs: Optional[int] = None
):
    ref_df = _get_ref_df_slice_from_query(ref_df_path, query_tuple, max_num_pdbs)
    pdbs_files = _pdb_files_from_ref_df(ref_df, pdb_folder)


    # XXX: take into account other "caching" mechanism
    # arising from using `save=True` in the following
    # function
    encodings = encode_pdbs(
        pdbs_files, ckpt_fnm, dask_cluster_args,
        save=False, outfnm=None, use_cluster=True
    )
    all_queried_pdbs = [f"{name}_{chain}" for name, chain in zip(ref_df['pdb'].values, ref_df['chain'].values)]
    encodings = encodings.reindex(pd.Index(all_queried_pdbs))
    return encodings



def cacheable_get_esm2_encodings_from_query(
    ref_df_path: str, query_tuple, pdb_folder, dask_cluster_args, max_num_pdbs: Optional[int] = None,
    max_workers: int = 1
):
    ref_df = _get_ref_df_slice_from_query(ref_df_path, query_tuple, max_num_pdbs)
    tfs, _ = cache(cacheable_get_pmpnn_pdbs_from_query_notied)(
        ref_df_path,
        query_tuple, pdb_folder, max_workers=max_workers, max_num_pdbs=max_num_pdbs
    )
    names = (ref_df['pdb'] + '_' + ref_df['chain']).values
    sequences = extract_sequences_from_pmpnn_pdbs(tfs, names)

    return encode_sequences(
        sequences,
        dask_cluster_args,
        save=False,
        outfnm=None,
        use_cluster=False
)


class Dataset:
    ref_df: Optional[pd.DataFrame]
    pdb_folder: Optional[Path]
    ckpt_fnm: Optional[Path]
    esm2_rep_fnm: Optional[Path]
    gearnet_rep_fnm: Optional[Path]
    dask_cluster_args: Optional[Dict]
    cath_query: Optional[dict[str, Any]]
    with_caching: bool
    pmpnn_loading_method: Literal["pmpnn", "pmpnn_notied"]
    max_num_pdbs: Optional[int]
    ref_df_path: Optional[str]
    max_workers: int
    _pdbs_gearnet_format: Optional[List[PackedProtein]]
    _pdbs_pmpnn_format: Optional[tfOutputTuple]
    _pdbs_pmpnn_format_notied: Optional[tfOutputTuple]
    _pmpnn_notied_encoding_mask: Optional[List]
    _encoding_df: Optional[pd.DataFrame]

    @classmethod
    def from_metadata(
        cls,
        ref_df,
        pdb_folder,
        ckpt_fnm,
        esm2_rep_fnm,
        gearnet_rep_fnm,
        dask_cluster_args: Optional[Dict] = None,
        cath_query: Optional[dict[str, Any]] = None,
        pmpnn_loading_method: Literal["pmpnn", "pmpnn_notied"] = "pmpnn",
        max_num_pdbs: Optional[int] = None,
    ):
        assert isinstance(ref_df, pd.DataFrame)
        self = cls()
        self.ref_df = ref_df
        self.pdb_folder = pdb_folder
        self.ckpt_fnm = ckpt_fnm
        self.esm2_rep_fnm = esm2_rep_fnm
        self.gearnet_rep_fnm = gearnet_rep_fnm
        self.dask_cluster_args = dask_cluster_args
        self.cath_query = None
        self.with_caching = False
        self.max_num_pdbs = max_num_pdbs
        self.pmpnn_loading_method = pmpnn_loading_method
        self.ref_df_path = None
        self._pdbs_gearnet_format = None
        self._pdbs_pmpnn_format = None
        self._pdbs_pmpnn_format_notied = None
        self._encoding_df = None
        self._AA_sequences = None
        self._esm2_encoding_df = None
        self.max_workers = 1
        return self

    @classmethod
    def from_cath_query(
        cls,
        pdb_folder,
        ckpt_fnm,
        esm2_rep_fnm,
        ref_df_path: str,
        dask_cluster_args: Optional[Dict] = None,
        cath_query: Optional[dict[str, Any]] = None,
        with_caching: bool = True,
        pmpnn_loading_method: Literal["pmpnn", "pmpnn_notied"] = "pmpnn",
        max_num_pdbs: Optional[int] = None,
        max_workers: int = 1
    ):
        self = cls()
        assert isinstance(cath_query, dict)
        self.ref_df = cast(pd.DataFrame, _get_ref_df_slice_from_query(
            ref_df_path, _query_tuple_from_dict(cath_query), max_num_pdbs=max_num_pdbs

        ))

        self.pdb_folder = pdb_folder

        self.ckpt_fnm = ckpt_fnm
        self.esm2_rep_fnm = esm2_rep_fnm
        self.dask_cluster_args = dask_cluster_args
        self.cath_query = cath_query
        self.gearnet_rep_fnm = None
        self.with_caching = with_caching
        self.pmpnn_loading_method = pmpnn_loading_method
        self.max_num_pdbs = max_num_pdbs
        self.ref_df_path = ref_df_path
        self.max_workers = max_workers
        self._pdbs_gearnet_format = None
        self._pdbs_pmpnn_format = None
        self._pdbs_pmpnn_format_notied = None
        self._encoding_df = None
        return self

    @classmethod
    def from_pdbs(
        cls,
        pdbs: List,
        ckpt_fnm: Path,
        esm2_rep_fnm: Path,
        gearnet_rep_fnm: Path,
        dask_cluster_args: Optional[Dict] = None,
    ):
        self = cls()
        self.ref_df = None
        self.pdb_folder = None
        self.cath_query = None
        self.with_caching = False
        self.ckpt_fnm = ckpt_fnm
        self.esm2_rep_fnm = esm2_rep_fnm
        self.gearnet_rep_fnm = gearnet_rep_fnm
        self.dask_cluster_args = dask_cluster_args
        self.max_num_pdbs = None
        self.ref_df_path = None
        self._pdbs_gearnet_format = pdbs
        self.pmpnn_loading_method = "pmpnn"
        self._pdbs_pmpnn_format = None
        self._pdbs_pmpnn_format_notied = None
        self._AA_sequences = None
        self._encoding_df = None
        self._esm2_encoding_df = None
        self.max_workers = 1
        return self

    @overload
    def get_pdbs(self, format: Literal["gearnet"] = "gearnet") -> List[PackedProtein]:
        ...

    @overload
    def get_pdbs(self, format: Literal["pmpnn"] = "pmpnn") -> tfOutputTuple:
        ...

    @overload
    def get_pdbs(self, format: Literal["pmpnn_notied"] = "pmpnn_notied") -> tfOutputTuple:
        ...

    @overload
    def get_pdbs(
        self, format: Literal["gearnet", "pmpnn", "pmpnn_notied"] = "gearnet"
    ) -> Union[List[PackedProtein], tfOutputTuple]:
        ...

    def get_pdbs(
        self, format: Literal["gearnet", "pmpnn", "pmpnn_notied"] = "gearnet"
    ) -> Union[List[PackedProtein], tfOutputTuple]:
        if format == "gearnet":
            return self._get_pdbs_gearnet_format()
        elif format == "pmpnn":
            return self._get_pdbs_pmpnn_format()
        elif format == "pmpnn_notied":
            return self._get_pdbs_pmpnn_format_notied()
        else:
            raise ValueError(f"Invalid format: {format}")

    def _get_pdbs_gearnet_format(self):
        if self.cath_query is not None:
            assert self.pdb_folder is not None
            assert self.ref_df_path is not None

            if self.with_caching:
                print("Getting gearnet-style pdbs from cath query -- cacheable call")
                func = cache(cacheable_get_gearnet_pdbs_from_query)
            else:
                func = cacheable_get_gearnet_pdbs_from_query
            self._pdbs_gearnet_format = func(
                self.ref_df_path,
                _query_tuple_from_dict(self.cath_query), self.pdb_folder,
                max_num_pdbs=self.max_num_pdbs
            )

        if self._pdbs_gearnet_format is None:
            assert self.ref_df is not None
            assert self.pdb_folder is not None

            pdbs = []
            for idx, pdb_name, chain_name in tqdm.tqdm(
                list(self.ref_df[["pdb", "chain"]].itertuples())
            ):
                pdb_file = os.path.join(self.pdb_folder, f"{pdb_name}_{chain_name}.pdb")
                pdb = import_pdb(pdb_file)
                pdbs.append(pdb)
            self._pdbs_gearnet_format = pdbs
        return self._pdbs_gearnet_format

    def _get_pdbs_pmpnn_format(self) -> tfOutputTuple:
        if self.cath_query is not None:
            assert self.pdb_folder is not None
            assert self.ref_df_path is not None
            if self.with_caching:
                print("Getting (pmpnn) pdbs from query -- cacheable call")
                func = cache(cacheable_get_pmpnn_pdbs_from_query)
            else:
                func = cacheable_get_pmpnn_pdbs_from_query

            self._pdbs_pmpnn_format = func(
                self.ref_df_path,
                _query_tuple_from_dict(self.cath_query), self.pdb_folder,
                max_num_pdbs=self.max_num_pdbs
            )
            return cast(tfOutputTuple, self._pdbs_pmpnn_format)
        if self._pdbs_pmpnn_format is None:
            self._pdbs_pmpnn_format = protein_structure_from_files(
                [str(f) for f in self.get_pdb_files()]
            )
        return self._pdbs_pmpnn_format

    def _get_pdbs_pmpnn_format_notied(self) -> tfOutputTuple:
        if self.cath_query is not None:
            assert self.pdb_folder is not None
            assert self.ref_df_path is not None
            if self.with_caching:
                print("Getting (pmpnn notied) pdbs from query -- cacheable call")
                func = cache(cacheable_get_pmpnn_pdbs_from_query_notied)
            else:
                func = cacheable_get_pmpnn_pdbs_from_query_notied

            self._pdbs_pmpnn_format_notied, self._pmpnn_notied_encoding_mask = func(
                self.ref_df_path,
                _query_tuple_from_dict(self.cath_query), self.pdb_folder,
                max_num_pdbs=self.max_num_pdbs,
                max_workers=self.max_workers,
            )
            return cast(tfOutputTuple, self._pdbs_pmpnn_format_notied)
        if self._pdbs_pmpnn_format_notied is None:
            pdbs = []
            for f in self.get_pdb_files():
                print(f"loading {f}")
                pdb = protein_structure_from_file(str(f))
                pdbs.append(pdb)
            self._pdbs_pmpnn_format_notied, self._pmpnn_notied_encoding_mask = concat_pdbs(pdbs)
        return self._pdbs_pmpnn_format_notied

    def _get_pdbs_from_query(self, query):
        pass

    def _get_XY_and_mask(self, return_pdbs=False):
        ref_df = self.ref_df

        X = self.get_encodings()
        has_valid_gearnet_encodings = (jnp.isnan(X).sum(axis=1) == 0)
        num_valid_gearnet_encodings = jnp.sum(has_valid_gearnet_encodings)
        print(f"{num_valid_gearnet_encodings} inputs with valid gearnet encodings")

        assert len(X) == len(ref_df)

        # make sure the sequence encodings have the right shape
        Y = self.get_sequence_encodings()
        Y_orig_len = jnp.zeros((len(ref_df), Y.shape[1]))

        has_valid_sequence_encodings = self._pmpnn_notied_encoding_mask
        assert has_valid_sequence_encodings is not None
        assert len(has_valid_sequence_encodings) == len(self.ref_df)

        num_valid_sequence_encodings = jnp.sum(jnp.array(has_valid_sequence_encodings))
        print(f"{num_valid_sequence_encodings} inputs that passed pdb screening encodings")

        non_nans_sequence_encodings = jnp.isnan(Y).sum(axis=1)
        num_non_nans_sequence_encodings = jnp.sum(jnp.array(has_valid_sequence_encodings))

        print(
            f"among these, {num_non_nans_sequence_encodings} with well-defined sequence encodings"
            f"others most likely exceeded the sequence size threshold"
        )

        idx = 0
        y_idx = 0
        for has_valid_encoding in has_valid_sequence_encodings:
            if has_valid_encoding:
                # if jnp.isnan(Y[y_idx]).sum() > 0:
                #     print(f"got nans for sequence encoding no. {y_idx},"
                #           f" most likely due to too large sequence")
                Y_orig_len = Y_orig_len.at[idx].set(Y[y_idx])
                y_idx += 1
            idx += 1

        # this one takes into account esm2 refusal to compute encodings for sequences > max len
        has_valid_esm2_sequence_encodings = jnp.isnan(Y_orig_len).sum(axis=1) == 0
        final_mask = (
            jnp.array(has_valid_sequence_encodings) * jnp.array(has_valid_gearnet_encodings) * has_valid_esm2_sequence_encodings
        ).astype(bool)
        final_X, final_Y =  X[final_mask], Y_orig_len[final_mask]
        assert len(final_X) == len(final_Y)
        print(f"dataset size {len(final_X)} (out of {len(ref_df)} initial pdbs)")

        # sequence encoding is based off pmpnn pdbs loading
        pmpnn_pdbs = self._get_pdbs_pmpnn_format_notied()
        assert len(pmpnn_pdbs.X) == len(Y)
        assert len(final_mask) == len(has_valid_sequence_encodings)
        final_mask_subsampled = jnp.array(
            [m for (m, a) in zip(final_mask, has_valid_sequence_encodings) if a]
        )

        jax_pmpnn_pdbs = pmpnn_pdb_as_pytree(pmpnn_pdbs)
        jax_pmpnn_pdbs_subsampled = jax.tree_map(lambda x: x[final_mask_subsampled], jax_pmpnn_pdbs)
        final_pdbs = jax_pmpnn_pdb_as_torch_pdb(jax_pmpnn_pdbs_subsampled)
        return (final_X, final_Y, final_pdbs), final_mask

    def get_XY(self):
        ret = self._get_XY_and_mask()
        return ret[0]


    def get_pdb_files(self):
        assert self.ref_df is not None
        assert self.pdb_folder is not None
        return _pdb_files_from_ref_df(self.ref_df, self.pdb_folder)

    def _compute_gearnet_encodings(self, save=False):
        files = self.get_pdb_files()
        return encode_pdbs(files, self.ckpt_fnm, self.dask_cluster_args,
                           save=save, outfnm=self.gearnet_rep_fnm)
    
    def get_sequence_encodings(self, save=False, use_cluster=True):
        if self.cath_query is not None:
            # pre-load the pdbs
            _ = self.get_pdbs('pmpnn_notied')
            print("Getting esm2 encodings from cath query -- cacheable call")
            assert self.pdb_folder is not None
            assert self.ref_df_path is not None
            if self.with_caching:
                func = cache(cacheable_get_esm2_encodings_from_query, ignore=["dask_cluster_args"])
            else:
                func = cacheable_get_esm2_encodings_from_query

            self._esm2_encoding_df = func(
                self.ref_df_path,
                _query_tuple_from_dict(self.cath_query),
                self.pdb_folder,
                self.dask_cluster_args,
                max_num_pdbs=self.max_num_pdbs,
                max_workers=self.max_workers,
            )

        if self._esm2_encoding_df is None:
            sequences = self.get_AA_sequences()
            self._esm2_encoding_df = (encode_sequences(sequences, self.dask_cluster_args,
                                     save=save, outfnm=self.esm2_rep_fnm,
                                    use_cluster=use_cluster))
        return jnp.array(self._esm2_encoding_df.values)


    def get_encodings(self, save=False):
        if self.cath_query is not None:
            assert self.ref_df_path is not None
            print("Getting gearnet encodings from cath query -- cacheable call")
            if self.with_caching:
                func = cache(
                    cacheable_get_gearnet_encodings_from_query,
                    ignore=["dask_cluster_args"]
                )
            else:
                func = cacheable_get_gearnet_encodings_from_query

            self._encoding_df = func(
                self.ref_df_path,
                _query_tuple_from_dict(self.cath_query),
                self.pdb_folder,
                self.ckpt_fnm,
                self.dask_cluster_args,  # type: ignore
                max_num_pdbs=self.max_num_pdbs,
            )

        if self._encoding_df is None:
            self._encoding_df = self._compute_gearnet_encodings(save=save)
        return jnp.array(self._encoding_df.values)

    def pad(self, tfs, pad_value=0):
        Ss = []
        for i, l in enumerate(tfs.lengths):
            new_S = jnp.array(tfs.S[i].detach().clone().cpu().numpy()).astype(float).at[l:].set(pad_value)
            Ss.append(new_S)

        Ss = jnp.stack(Ss)
        return Ss

    def get_sequences(self, pad_value=99):
        tfs = self.get_pdbs(self.pmpnn_loading_method)
        return self.pad(tfs, pad_value)
    
    def get_AA_sequences(self):
        if self._AA_sequences is not None:
            assert self.ref_df is not None
            tfs = self.get_pdbs(self.pmpnn_loading_method)
            names = (self.ref_df['pdb'] + '_' + self.ref_df['chain']).values
            self._AA_sequences = extract_sequences_from_pmpnn_pdbs(tfs, names)
        return self._AA_sequences
