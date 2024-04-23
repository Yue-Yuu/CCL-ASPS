import numpy as np
import csv
import os
from tqdm import tqdm
import sys
import math
import argparse
import re
from itertools import combinations
import requests
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import torch

parser = argparse.ArgumentParser(description='make_adj_set')

parser.add_argument('--distance', default=12, type=float,
                    help="distance threshold")
args = parser.parse_args()

def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)


def read_atoms(file, chain=".", model=1):
    pattern = re.compile(chain)

    current_model = model
    atoms = []
    for line in file:
        line = line.strip()
        if line.startswith("ATOM"):
            type = line[12:16].strip()
            chain = line[21:22]
            if type == "CA" and re.match(pattern, chain):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atoms.append((x, y, z))
        # elif line.startswith("MODEL"):
        #     current_model = int(line[10:14].strip())
    return atoms


def compute_contacts(atoms, threshold):
    contacts = []
    for i in range(len(atoms)-2):
        for j in range(i+2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i, j))
    return contacts


def write_output(contacts, file):
    for c in contacts:
        file.write("\t".join(map(str, c))+"\n")


def pdb_to_cm(file, threshold, chain=".", model=1):
    atoms = read_atoms(file, chain, model)
    return compute_contacts(atoms, threshold)

# get druf structure adja and features
def get_drug_information(drug_names,drug_id2smiles):
    atom_featurizer = CanonicalAtomFeaturizer()  # 原子的默认特征
    bond_featurizer = CanonicalBondFeaturizer(self_loop=True)  # Bond默认特征
    fc = partial(smiles_to_bigraph, add_self_loop=True)

    drug_x_all = []
    drug_edge_all = []
    drug_atom_num_edges = []
    drug_atom_edges = []
    drug_ids = np.arange(len(drug_names))
    drug_smiles = drug_id2smiles[:,1]

    for smiles in tqdm(drug_smiles):
        a = 0
        seq_smile = smiles
        seq_graph = fc(smiles=seq_smile, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
        actual_node_feats = seq_graph.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]  # drug 中的原子数量 # 不满最大值时的原子的填充数量
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)

        drug_x_all.append(actual_node_feats)
        drug_edges = torch.cat((seq_graph.edges()[0].unsqueeze(0),seq_graph.edges()[1].unsqueeze(0)),dim=0)
        drug_edge_all.append(drug_edges)

    torch.save(drug_x_all, '../x_list_drug.pt')
    torch.save(drug_edge_all,'../drug_smile_structure_edge_list.pt')
    return drug_x_all, drug_edge_all

# todo generate protein structure adj
c = 0
count1 = -1
protein_names = np.loadtxt("../data/protein.txt",dtype=str)
list_all = []
for liness1 in tqdm(protein_names):
    pdb_file_name = "../data_collect/best_pdb_file/" + liness1 + '.pdb'
    print(pdb_file_name)
    c = c + 1
    contacts = pdb_to_cm(open(pdb_file_name, "r"), args.distance)
    list_all.append(contacts)

list_all = np.array(list_all)
np.save('../protein_structure_edge_list.npy', list_all)

a = 0

# generate drug structure adj and features

drug_names = np.loadtxt("../data/drug.txt",dtype=str)
drug_id2smiles = np.loadtxt("../data/drug_file/drug_smiles_list.txt",dtype=str,delimiter=":",comments=None)

get_drug_information(drug_names,drug_id2smiles)
a = 0
