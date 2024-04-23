import numpy as np
import csv
import os
from tqdm import tqdm
import sys
import math
import argparse
import re
import torch
from itertools import combinations


def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def read_atoms(file, chain=".", model=1):
    pattern = re.compile(chain)

    current_model = model
    atoms = []
    ajs = []
    for line in file:
        line = line.strip()
        if line.startswith("ATOM"):
            type = line[12:16].strip()
            chain = line[21:22]
            if type == "CA" and re.match(pattern, chain):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                ajs_id = line[17:20]
                atoms.append((x, y, z))
                ajs.append(ajs_id)
        # elif line.startswith("MODEL"):
        #     current_model = int(line[10:14].strip())
    return atoms, ajs


def compute_contacts(atoms, threshold):
    contacts = []
    for i in range(len(atoms) - 2):
        for j in range(i + 2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i + 1, j + 1))
    return contacts


def write_output(contacts, file):
    for c in contacts:
        file.write("\t".join(map(str, c)) + "\n")


def pdb_to_x(file, threshold, chain=".", model=1):
    atoms, ajs = read_atoms(file, chain, model)
    return ajs


# x_set = pdb_to_x(open('P84085.pdb', "r"), 7.5)
# replace_dict = {'ALA':'89.09', '71.07':'', '':'',}
c = 0
count1 = -1
list_all = []
x_set = np.zeros(())
all_for_assign = np.loadtxt("../data_collect/all_assign.txt") # 20个氨基酸的维度为7的特征表示
for liness1 in tqdm(open('../data_collect/protein.txt')):
    line = liness1.split()[0]
    pdb_file_name = '../data_collect/best_pdb_file/' + line + '.pdb'
    print(pdb_file_name)
    c = c + 1
    xx = pdb_to_x(open(pdb_file_name, "r"), 7.5)
    x_p = np.zeros((len(xx), 7))
    for j in range(len(xx)):
        if xx[j] == 'ALA':
            x_p[j] = all_for_assign[0, :]
        elif xx[j] == 'CYS':
            x_p[j] = all_for_assign[1, :]
        elif xx[j] == 'ASP':
            x_p[j] = all_for_assign[2, :]
        elif xx[j] == 'GLU':
            x_p[j] = all_for_assign[3, :]
        elif xx[j] == 'PHE':
            x_p[j] = all_for_assign[4, :]
        elif xx[j] == 'GLY':
            x_p[j] = all_for_assign[5, :]
        elif xx[j] == 'HIS':
            x_p[j] = all_for_assign[6, :]
        elif xx[j] == 'ILE':
            x_p[j] = all_for_assign[7, :]
        elif xx[j] == 'LYS':
            x_p[j] = all_for_assign[8, :]
        elif xx[j] == 'LEU':
            x_p[j] = all_for_assign[9, :]
        elif xx[j] == 'MET':
            x_p[j] = all_for_assign[10, :]
        elif xx[j] == 'ASN':
            x_p[j] = all_for_assign[11, :]
        elif xx[j] == 'PRO':
            x_p[j] = all_for_assign[12, :]
        elif xx[j] == 'GLN':
            x_p[j] = all_for_assign[13, :]
        elif xx[j] == 'ARG':
            x_p[j] = all_for_assign[14, :]
        elif xx[j] == 'SER':
            x_p[j] = all_for_assign[15, :]
        elif xx[j] == 'THR':
            x_p[j] = all_for_assign[16, :]
        elif xx[j] == 'VAL':
            x_p[j] = all_for_assign[17, :]
        elif xx[j] == 'TRP':
            x_p[j] = all_for_assign[18, :]
        elif xx[j] == 'TYR':
            x_p[j] = all_for_assign[19, :]
    list_all.append(x_p)

torch.save(list_all, '../x_list_protein.pt')

