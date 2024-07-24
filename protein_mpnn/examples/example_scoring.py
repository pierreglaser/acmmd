from typing import Optional

import torch

from pmpnn_extra.pmpnn_utils import get_pmpnn_model, score
from pmpnn_extra.protein_file_utils import protein_structure_from_file, sequence_from_fasta


# arguments
class Arguments:
    def __init__(
        self,
        pdb_path: str,  # Path to a single PDB to be designed
        fasta_path: Optional[
            str
        ] = None,  # Path to file containing one sequence to be scored in fasta format - currently incompatible with multiple sequences
        ca_only: bool = False,  # Parse CA-only structures and use CA-only models
        backbone_noise: float = 0,  # Standard deviation of Gaussian noise to add to backbone atoms
        max_length: int = 200000,  # Max sequence length
        model_path: str = "ProteinMPNN/vanilla_model_weights/v_48_020.pt",  # Path to model weights folder
    ):
        self.pdb_path = pdb_path
        self.fasta_path = fasta_path
        self.ca_only = ca_only
        self.backbone_noise = backbone_noise
        self.max_length = max_length
        self.model_path = model_path


def pmpnn_score_pdb_seq(args):
    tf = protein_structure_from_file(args.pdb_path, args.ca_only, args.max_length)

    scores = score(model, tf)
    return scores


torch.manual_seed(0)
if __name__ == "__main__":
    args = Arguments(
        pdb_path="examples/pdbs/5L33.pdb",
        fasta_path="examples/fastas/5L33-mut_seq.fasta",
    )

    model = get_pmpnn_model(args.model_path, args.ca_only, args.backbone_noise)

    # scoring pdb given a sequence
    print("\nexample - scoring 5L33 with mutant sequence from fasta file")
    tf = protein_structure_from_file(
        args.pdb_path, args.ca_only, args.max_length, fasta_path=args.fasta_path
    )
    print(score(model, tf))

    # scoring pdb given a sequence
    print("\nexample - scoring 6MRR with mutant sequence from fasta file")
    args = Arguments(
        pdb_path="examples/pdbs/6MRR.pdb",
        fasta_path="examples/fastas/6MRR-mut_seq.fasta",
    )
    tf = protein_structure_from_file(
        args.pdb_path, args.ca_only, args.max_length, fasta_path=args.fasta_path
    )
    print(score(model, tf))

    # scoring pdb given no sequence – score sequence in pdb
    print("\nexample - scoring 5L33 with wild-type sequence from pdb")
    args = Arguments(pdb_path="examples/pdbs/5L33.pdb")
    tf = protein_structure_from_file(args.pdb_path, args.ca_only, args.max_length)
    print(score(model, tf))

    # scoring pdb given no sequence – score sequence in pdb
    print("\nexample - scoring 6MRR with wild-type sequence from pdb")
    args = Arguments(pdb_path="examples/pdbs/6MRR.pdb")
    tf = protein_structure_from_file(args.pdb_path, args.ca_only, args.max_length)
    print(score(model, tf))
