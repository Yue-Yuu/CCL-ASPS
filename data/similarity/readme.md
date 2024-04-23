
#### directory

- `sim_mat_drug.txt`: similarity learned from Similarity_Matrix_Drugs;Drug similarity scores based on chemical structures of drugs
- `sim_mat_drug_disease.txt`: similarity learned from Drug-Disease association matrix
- `sim_mat_drug_drug.txt`: similarity learned from Drug-Drug interaction matrix
- `sim_mat_drug_protein.txt`: similarity learned from Drug_Protein interaction matrix (transpose of the above matrix)
- `sim_mat_drug_protein_remove_homo.txt`: similarity learned from  Drug_Protein interaction matrix, in which homologous proteins with identity score >40% were excluded
- `sim_mat_drug_se.txt`: similarity learned from Drug-SideEffect association matrix
- `sim_mat_protein.txt`: similarity learned from Protein similarity scores based on primary sequences of proteins
- `sim_mat_protein_disease.txt`: similarity learned from Protein-Disease association matrix
- `sim_mat_protein_drug.txt`: similarity learned from Protein-Drug interaction matrix
- `sim_mat_protein_protein.txt`: Protein-Protein interaction matrix
- `association_sim_drug.txt`: combine `sim_mat_drug_drug`, `sim_mat_drug_disease`, `sim_mat_drug_se`get the association similarity
- `association_sim_protein.txt`: combine `sim_mat_protein_disease.txt`, `sim_mat_protein_protein.txt` get the association similarity


