# ipython -i protein_mpnn/helper_scripts/get_gearnet_embeddings.py -- \
#     --ref_fnm structures/cath_reference_files/cath-single_chain_domains-topology_number_count_gte10.csv \
#     --complex_pdb_repo structures/pdb_repo \
#     --pdb_chain_repo structures/pdb_chain_repo \
#     --ckpt_fnm structures/checkpoints/angle_gearnet_edge.pth \
#     --outdir structures/gearnet_edge_graph_embeddings \
#     --max_num_pdbs 100 \
#     --profile=anon
#
ipython -i ./get_gearnet_embeddings.py -- --max_num_pdbs 20 --profile=anon
