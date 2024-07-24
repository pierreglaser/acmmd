from typing import Optional
import torch

from pmpnn_extra.pmpnn_utils import get_pmpnn_model, sample
from pmpnn_extra.protein_file_utils import protein_structure_from_file, sequence_from_fasta


# arguments
class Arguments:
    def __init__(
        self,
        pdb_path: str,  # Path to a single PDB to be designed
        fasta_path: Optional[str] = None,  # Path to file containing one sequence to be scored in fasta format - currently incompatible with multiple sequences
        ca_only: bool = False,  # Parse CA-only structures and use CA-only models
        backbone_noise: float = 0,  # Standard deviation of Gaussian noise to add to backbone atoms
        max_length: int = 200000,  # Max sequence length
        model_path: str = "ProteinMPNN/vanilla_model_weights/v_48_020.pt",  # Path to model weights folder
        temperature: float = 0.1,  # the default sampling temperature
        num_samples: int = 1,  # the number of sequences to sample from the pdb
    ):
        self.pdb_path = pdb_path
        self.fasta_path = fasta_path
        self.ca_only = ca_only
        self.backbone_noise = backbone_noise
        self.max_length = max_length
        self.model_path = model_path
        self.temperature = temperature
        self.num_samples = num_samples


if __name__ == "__main__":
    # scoring pdb given a sequence
    print("\nexample - sampling sequences from 5L33 ")
    args = args = Arguments(
        pdb_path="examples/pdbs/5L33.pdb", temperature=0.1, num_samples=10
    )

    model = get_pmpnn_model(args.model_path, args.ca_only, args.backbone_noise)
    tf = protein_structure_from_file(args.pdb_path, args.ca_only, args.max_length)

    # read in sequence from fasta if given
    if args.fasta_path:
        S_input = sequence_from_fasta(args.fasta_path, tf.X)
        tf.S[
            :, : S_input.shape[1]
        ] = S_input  # assumes that S and S_input are alphabetically sorted for masked_chains

    seqs, S = sample(model, tf, args.num_samples, args.temperature)
    print(seqs)
