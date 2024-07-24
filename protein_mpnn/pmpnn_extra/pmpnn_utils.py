#!/usr/bin/env python
# coding: utf-8

# Preparing simplified functions from the PMPNN script
# pyright: basic, reportPrivateUsage=false
from pathlib import Path

import numpy as np
import torch

import ProteinMPNN as protein_mpnn
from ProteinMPNN.protein_mpnn_utils import ProteinMPNN, _scores

# data objects to initialize
alphabet = "ACDEFGHIKLMNPQRSTVWYX"
alphabet_dict = dict(zip(alphabet, range(21)))


def get_pmpnn_model(model_path=None, ca_only=False, backbone_noise=0.0):
    if model_path is None:
        model_path = (
            Path(protein_mpnn.__path__[0])  # type: ignore
            / "vanilla_model_weights/v_48_020.pt"
        )
    # initialize the model
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    ## load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    hidden_dim = 128
    num_layers = 3

    model = ProteinMPNN(
        ca_only=ca_only,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=backbone_noise,
        k_neighbors=checkpoint["num_edges"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def score(model, tf, S=None, decoding_order_type="random"):
    # TODO: compatability with scoring multiple sequences
    # score sequence for pdb (log probs)
    decoding_order = None
    randn_1 = None

    if S is None:
        S = tf.S

    if decoding_order_type == "random":
        randn_1 = torch.randn(tf.chain_M.shape, device=tf.X.device)
        use_input_decoding_order = False
    elif decoding_order_type == "sequential":
        use_input_decoding_order = True
        decoding_order = torch.arange(
            tf.chain_M.shape[1], device=tf.X.device
        ).unsqueeze(0)
    else:
        raise ValueError(
            f"decoding_order must be 'random' or 'sequential', got "
            f"{decoding_order_type}"
        )

    log_probs = model(
        tf.X,
        S,
        tf.mask,
        tf.chain_M * tf.chain_M_pos,
        tf.residue_idx,
        tf.chain_encoding_all,
        randn_1,
        use_input_decoding_order=use_input_decoding_order,
        decoding_order=decoding_order,
    )
    mask_for_loss = tf.mask * tf.chain_M * tf.chain_M_pos
    scores = _scores(S, log_probs, mask_for_loss)
    return scores


def sample(
    model: ProteinMPNN,
    tf,
    num_samples,
    temperature,
    X=None,
    return_seqs=True,
    decoding_order_type="random",
    pad_value=99,
):
    if X is None:
        X = tf.X
    # set some default inputs according to the pmpnn_run code
    # which amino acids not to sample
    omit_AAs_list = ["X"]
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    # don't bias the amino acid generation
    bias_AAs_np = np.zeros(len(alphabet))
    # don't bias the sampling by a pssm
    pssm_multi = 0.0

    # sample sequences from a pdb
    all_seqs = []
    all_S = []
    for ii in range(num_samples):
        if decoding_order_type == "random":
            randn_2 = torch.randn(tf.chain_M.shape, device=X.device)
        elif decoding_order_type == "sequential":
            randn_2 = torch.arange(tf.chain_M.shape[1], device=X.device).unsqueeze(0)
        else:
            raise ValueError(
                f"decoding_order must be 'random' or 'sequential', got "
                f"{decoding_order_type}"
            )

        sample_dict = model.sample(
            X,
            randn_2,
            tf.S,
            tf.chain_M,
            tf.chain_encoding_all,
            tf.residue_idx,
            mask=tf.mask,
            temperature=temperature,
            omit_AAs_np=omit_AAs_np,
            bias_AAs_np=bias_AAs_np,
            chain_M_pos=tf.chain_M_pos,
            bias_by_res=tf.bias_by_res_all,
        )
        S_sample = sample_dict[
            "S"
        ]  # a tensor where each element is the idx in "alphabet" for which AA is at that position
        if return_seqs:
            S_seq = "".join([alphabet[s] for s in S_sample[0]])
        else:
            S_seq = None

        S_sample = S_sample + (1 - tf.chain_M).long() * pad_value
        # uncomment if we need the log prob of these samples
        # log_probs = model(tf.X, S_sample, tf.mask, tf.chain_M*tf.chain_M_pos, tf.residue_idx, tf.chain_encoding_all, randn_2,
        #                    use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
        # mask_for_loss = tf.mask*tf.chain_M*tf.chain_M_pos
        # scores = _scores(S_sample, log_probs, mask_for_loss)
        # scores = scores.cpu().data.numpy()

        all_seqs.append(S_seq)
        all_S.append(S_sample)

    all_S = torch.stack(all_S, dim=1)
    return all_seqs, all_S
