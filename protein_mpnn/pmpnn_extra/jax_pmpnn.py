"ConditionalModel wrapper of a ProteinMPNN torch module"
# pyright: basic, reportPrivateUsage=false, reportPrivateImportUsage=false
from collections import namedtuple
from typing import Any, Callable, Literal, cast, Optional

import jax
import jax.numpy as jnp
from pmpnn_extra.preprocessing import AAseq_from_pmpnn_seq, encode_sequences, encode_sequences_sequential
import torch
from calibration.conditional_models.base import SampleableModel
from flax import struct
from kwgflows.pytypes import Array, Numeric, PRNGKeyArray, PyTree_T

from pmpnn_extra.pmpnn_utils import get_pmpnn_model, sample, score

# from pytorch2jax import convert_pytnn_to_flax, convert_pytnn_to_jax, convert_to_jax
from pmpnn_extra.pytorch2jax import as_jax_function, as_jax_pytree
from ProteinMPNN.protein_mpnn_utils import ProteinMPNN

jax_tfOutputTuple = namedtuple(
    "jax_tfOutputTuple",
    [
        "X",
        "S",
        "mask",
        "lengths",
        "chain_M",
        "chain_encoding_all",
        # "chain_list_list",
        # "visible_list_list",
        # "masked_list_list",
        # "masked_chain_length_list_list",
        "chain_M_pos",
        "omit_AA_mask",
        "residue_idx",
        "dihedral_mask",
        # "tied_pos_list_of_lists_list",
        "pssm_coef",
        "pssm_bias",
        "pssm_log_odds_all",
        "bias_by_res_all",
        # "tied_beta"
    ],
)

ESM_2_MODEL = None
ESM_2_ALPHABET = None
ESM_2_BATCH_CONVERTER = None


def get_esm2_model():
    global ESM_2_MODEL
    global ESM_2_ALPHABET
    global ESM_2_BATCH_CONVERTER

    if ESM_2_MODEL is None:
        import esm
        import time
        t0 = time.time()

        esm2_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        esm2_model.eval() 
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device for sequence encoding:")
        print(device)
        esm2_model = esm2_model.to(device)

        print("time loading model", time.time() - t0)

        ESM_2_MODEL = esm2_model
        ESM_2_ALPHABET = alphabet
        ESM_2_BATCH_CONVERTER = batch_converter

    return ESM_2_MODEL, ESM_2_ALPHABET, ESM_2_BATCH_CONVERTER


class JaxProteinMPNN(SampleableModel):
    # conditioned structure. The pytree paradigm allows to easily create a collection
    # of models with different conditioning structures.
    tf: jax_tfOutputTuple

    # reference to the orginal pytorch ProteinMPNN model
    model: ProteinMPNN = struct.field(pytree_node=False)

    temperature: float = struct.field(pytree_node=False)

    return_esm2_embeddings: bool = struct.field(pytree_node=False)

    pad_value: int = struct.field(pytree_node=False)

    batch_size: int = struct.field(pytree_node=False)

    force_batching: int = struct.field(pytree_node=False)

    # pytorch sample/score functions (generated during PytorchModel.create)
    _torch_log_prob_fn: Callable[[Any], torch.Tensor] = struct.field(pytree_node=False)
    _torch_sample_fn: Callable[[Any, int], torch.Tensor] = struct.field(
        pytree_node=False
    )

    # JAX-converted sample/score functions (generated during PytorchModel.create)
    _jax_log_prob_fn: Callable[[PyTree_T], Numeric] = struct.field(pytree_node=False)
    _jax_sample_fn: Callable = struct.field(pytree_node=False)

    _torch_esm2_embedding_fn: Callable = struct.field(pytree_node=False)
    _jax_esm2_embedding_fn: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls, tf, batching_method: Literal["native", "vmap", "passhtrough_batched"] = "native", temperature=0.1,
        return_esm2_embeddings=False, pad_value=99, force_batching=False, batch_size=5,
    ):
        model = get_pmpnn_model()
        jax_tf = jax_tfOutputTuple(
            X=jnp.array(tf.X),
            S=jnp.array(tf.S),
            mask=jnp.array(tf.mask),
            lengths=jnp.array(tf.lengths),
            chain_M=jnp.array(tf.chain_M),
            chain_encoding_all=jnp.array(tf.chain_encoding_all),
            # chain_list_list=tf.chain_list_list,
            # visible_list_list=tf.visible_list_list,
            # masked_list_list=tf.masked_list_list,
            # masked_chain_length_list_list=tf.masked_chain_length_list_list,
            chain_M_pos=jnp.array(tf.chain_M_pos),
            omit_AA_mask=jnp.array(tf.omit_AA_mask),
            residue_idx=jnp.array(tf.residue_idx),
            dihedral_mask=jnp.array(tf.dihedral_mask),
            # tied_pos_list_of_lists_list=tf.tied_pos_list_of_lists_list,
            pssm_coef=jnp.array(tf.pssm_coef),
            pssm_bias=jnp.array(tf.pssm_bias),
            pssm_log_odds_all=jnp.array(tf.pssm_log_odds_all),
            bias_by_res_all=jnp.array(tf.bias_by_res_all),
            # tied_beta=tf.tied_beta
        )

        # keep a tf in the signatures to allow for dynamic reallocation
        # using model.replace(tf=tf)
        def torch_log_prob_fn(tf, S=None):
            return score(model, tf, S=S, decoding_order_type="sequential")

        def torch_sample_fn(tf, num_samples, temperature, pad_value):
            print("using temperature", temperature)
            return sample(
                model,
                tf,
                num_samples=num_samples,
                temperature=temperature,
                decoding_order_type="sequential",
                return_seqs=False,
                pad_value=pad_value,
            )[1]

        def torch_esm2_embedding_fn(sequences, pad_value):
            if len(sequences.shape)  == 1:
                one_d_seq = True
                sequences = sequences[None]
            else:
                one_d_seq = False

            model, alphabet, batch_converter = get_esm2_model()

            sequences_formatted = AAseq_from_pmpnn_seq(sequences, pad_value)
            encoded_sequences = encode_sequences_sequential(
                sequences_formatted,
                model=model,
                alphabet=alphabet,
                batch_converter=batch_converter,
            )
            # if one_d_seq:
            #     encoded_sequences = encoded_sequences.iloc[:1]
            # encoded_sequences = encode_sequences(
            #     sequences_formatted,
            #     dask_cluster_args=None,
            #     use_cluster=False,
            # )
            return torch.Tensor(encoded_sequences.values)


        jax_log_prob_fn = cast(
            Callable[[PyTree_T], Numeric],
            as_jax_function(torch_log_prob_fn),
        )

        jax_sample_fn = cast(
            Callable,
            as_jax_function(torch_sample_fn, mode="passhtrough_batched")
            # as_jax_function(torch_sample_fn)
        )

        jax_esm2_embedding_fn = cast(
            Callable,
            as_jax_function(torch_esm2_embedding_fn, mode="sequential"),
        )

        if jax_tf.X.shape[0] != 1 and batching_method in ("vmap",):
            # multiple protein structures. If vmap is used to vectorize sampling over
            # protein structures (e.g `batching_method == "vmap"`), then we need to
            # add one dimension to all fields, as ProteinMPNN uses a batch dimension
            # even for sampling from a single protein structure.
            # dim 0: actual batch dimension (number of protein structure)
            # dim 1: dummy batch dimension so that single structure sampling works
            jax_tf = jax.tree_map(lambda x: x[:, None], jax_tf)

        return cls(
            jax_tf,
            model,
            temperature,
            return_esm2_embeddings,
            pad_value,
            batch_size,
            force_batching,
            torch_log_prob_fn,
            torch_sample_fn,
            jax_log_prob_fn,
            jax_sample_fn,
            torch_esm2_embedding_fn,
            jax_esm2_embedding_fn,
        )

    def sample_from_conditional(self, key: PRNGKeyArray, num_samples: int = 1) -> Array:

        if self.force_batching:
            print(f"using batch size: {self.batch_size}")
            batch_size = self.batch_size
            num_structures = len(self.tf.X)
            num_batches = num_structures // batch_size  + (num_structures % batch_size > 0)
            print("num_batches", num_batches)
            samples = []
            for i in range(num_batches):
                tfs_batch = jax.tree_map(lambda x: x[i*batch_size:(i+1)*batch_size], self.tf)
                samples_batch = self._jax_sample_fn(tfs_batch, num_samples, self.temperature, self.pad_value)
                samples.append(samples_batch)

            samples = jnp.concatenate(samples, axis=0)
        else:
            samples = self._jax_sample_fn(self.tf, num_samples, self.temperature, self.pad_value)
        # samples = self._jax_sample_fn(self.tf, num_samples, self.temperature, self.pad_value)

        if self.return_esm2_embeddings:
            # __import__('pdb').set_trace()
            return self._jax_esm2_embedding_fn(samples, self.pad_value)
        else:
            return self._jax_sample_fn(self.tf, num_samples, self.temperature)

    def score(self, S):
        if self.return_esm2_embeddings:
            raise ValueError(
                "cannot score ProteinMPNN on esm2 embeddings"
            )
        else:
            return self._jax_log_prob_fn(self.tf, S)
