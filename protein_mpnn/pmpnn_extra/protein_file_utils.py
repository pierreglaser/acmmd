#!/usr/bin/env python
# coding: utf-8

# Preparing simplified functions from the PMPNN script
# pyright: basic, reportPrivateUsage=false


from pathlib import Path
import torch
from collections import namedtuple

import pmpnn_extra
from ProteinMPNN.protein_mpnn_utils import tied_featurize, parse_PDB, parse_fasta
from ProteinMPNN.protein_mpnn_utils import StructureDatasetPDB


#data objects to initialize
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
alphabet_dict = dict(zip(alphabet, range(21)))
    
tfOutputTuple = namedtuple("tfOutputTuple", ["X", "S", "mask", "lengths", "chain_M",
                                             "chain_encoding_all", "chain_list_list",
                                             "visible_list_list", "masked_list_list",
                                             "masked_chain_length_list_list", "chain_M_pos",
                                             "omit_AA_mask", "residue_idx", "dihedral_mask",
                                             "tied_pos_list_of_lists_list", "pssm_coef",
                                             "pssm_bias", "pssm_log_odds_all", "bias_by_res_all",
                                             "tied_beta"])


def protein_structure_from_files(
    pdb_paths=None, pdb_names=None, ca_only=False, max_length=200000, fasta_paths=None, fasta_names=None,
):
    if pdb_paths is None:
        assert pdb_names is not None, "Must provide either pdb_paths or pdb_names"
        default_path = Path(list(pmpnn_extra.__path__)[0]).parent / "examples" / "pdbs"
        pdb_paths = []
        for name in pdb_names:
            pdb_paths.append(str(default_path / name))

    if fasta_names is not None:
        assert fasta_paths is None, "Must provide either fasta_path or fasta_name"
        default_path = Path(list(pmpnn_extra.__path__)[0]).parent / "examples" / "fastas"
        fasta_paths = []
        for name in fasta_names:
            fasta_paths.append(str(default_path / name))

    # prepare pdb input for tied_featurize: sequence, coordinates, metadata extracted from pdb and saved to dict
    pdb_dict_list = parse_PDB(pdb_paths, ca_only=ca_only)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)
    batch_clones = dataset_valid.data
    
    # load variables to pass to model to get log_probs
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    #device = torch.device("cpu") #fix to this as downstream processing leverages cpu
    tf_output = tied_featurize(batch=batch_clones,
                               device=device,
                               chain_dict=None,
                               fixed_position_dict=None,  # ?
                               omit_AA_dict=None,  #?
                               tied_positions_dict=None, #?
                               pssm_dict=None,  # ?
                               bias_by_res_dict=None,  # ?
                               ca_only=ca_only)

    ## save return (20 outputs) to named tuple
    tf = tfOutputTuple(*tf_output)

    # read in sequence from fasta if given
    if fasta_paths is not None:
        assert len(fasta_paths) == len(pdb_paths), "Must provide fasta for each pdb"
        for i, fasta_path in enumerate(fasta_paths):
            S_input = sequence_from_fasta(fasta_path, tf.X[i])
            tf.S[
                i : S_input.shape[1]
            ] = S_input
    return tf


def protein_structure_from_file(pdb_path=None, pdb_name=None, ca_only=False, max_length=200000, fasta_path=None, fasta_name=None):
    if pdb_path is None:
        assert pdb_name is not None, "Must provide either pdb_path or pdb_name"
        default_path = Path(list(pmpnn_extra.__path__)[0]).parent / "examples" / "pdbs"
        pdb_path = str(default_path / pdb_name)
    else:
        assert pdb_name is None, "Must provide either pdb_path or pdb_name"

    if fasta_name is not None:
        assert fasta_path is None, "Must provide either fasta_path or fasta_name"
        default_path = Path(list(pmpnn_extra.__path__)[0]).parent / "examples" / "fastas"
        fasta_path = str(default_path / fasta_name)


    # prepare pdb input for tied_featurize: sequence, coordinates, metadata extracted from pdb and saved to dict
    pdb_dict_list = parse_PDB(pdb_path, ca_only=ca_only)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)
    batch_clones = [dataset_valid[0]] # if only one pdb passed as input and batch size = 1
    
    # load variables to pass to model to get log_probs
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    #device = torch.device("cpu") #fix to this as downstream processing leverages cpu
    tf_output = tied_featurize(batch=batch_clones,
                               device=device,
                               chain_dict=None,
                               fixed_position_dict=None,  # ?
                               omit_AA_dict=None,  #?
                               tied_positions_dict=None, #?
                               pssm_dict=None,  # ?
                               bias_by_res_dict=None,  # ?
                               ca_only=ca_only)

    ## save return (20 outputs) to named tuple
    tf = tfOutputTuple(*tf_output)

    # read in sequence from fasta if given
    if fasta_path is not None:
        S_input = sequence_from_fasta(fasta_path, tf.X)
        tf.S[
            :, : S_input.shape[1]
        ] = S_input  # assumes that S and S_input are alphabetically sorted for masked_chains
    return tf


def sequence_from_fasta(path, X):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    fasta_names, fasta_seqs = parse_fasta(path, omit=["/"])
    assert len(fasta_seqs) == 1 ## currently only compatible with one pdb in, one pdb out
    fasta_seq = fasta_seqs[0]
    input_seq_length = len(fasta_seq)


    # assumes no insertions/delections
    assert input_seq_length == X.shape[1] #

    # update tf.S to be input sequence – otherwise is sequence read from pdb
    # XXX: dims of X??
    S_input = torch.tensor([alphabet_dict[AA] for AA in fasta_seq], device=device)[None,:].repeat(X.shape[0], 1)
    return S_input
