# Bubble: Yue Yu
# train.py
# -*- utf-8-*-


import datetime
import os
import re
import time
import argparse
import torch.nn as nn
import torch.optim as optim
from models.models import CCL_ASPS
from utils import *
from Sample_processing import *

###############################################################
# Training settings
parser = argparse.ArgumentParser(description='CCL_ASPS')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument("--fold_num", type=int, default=5,
                    help="the num k of k-fold")
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')

###############################################################
# data
parser.add_argument('--drug_num', type=int, default=708,
                    help='drug node num')
parser.add_argument('--protein_num', type=int, default=1512,
                    help='microbe node num')
parser.add_argument('--data_path', type=str, default='./data',
                    help='dataset root path')

parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--tau', type=float, default=0.8)
parser.add_argument('--lam', type=float, default=0.5)
parser.add_argument("--alpha", type= float, default=0.8)
parser.add_argument("--beta", type= float, default=0.8)
parser.add_argument("--gamma", type= float, default=0.3)

parser.add_argument('--protein_hidden', default=64, type=int, help='protein hidden dim')
parser.add_argument('--drug_hidden', default=64, type=int, help='drug hidden dim')
parser.add_argument('--batch_size', default=256, type=int, help='drug batch size')
parser.add_argument('--pseq_path', default="./protein_info/protein.SHS27k.sequences.dictionary.pro3.tsv", type=str, help="protein sequence path")
parser.add_argument('--vec_path', default='./protein_info/vec5_CTC.txt', type=str, help='protein sequence vector path')
parser.add_argument('--p_feat_matrix', default="data/processed/x_list_protein.pt", type=str, help="protein feature matrix")
parser.add_argument('--p_adj_matrix', default="data/processed/protein_structure_edge_list.npy", type=str, help="protein adjacency matrix")
parser.add_argument('--d_feat_matrix', default="data/processed/x_list_drug.pt", type=str, help="protein feature matrix")
parser.add_argument('--d_adj_matrix', default="data/processed/drug_smile_structure_edge_list.pt", type=str, help="protein adjacency matrix")
parser.add_argument('--model_type', default="CCL-ASPS", type=str, help="3 model types. pretrain_drug,pretrain_protein,CCL-ASPS")

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    args.device = "cuda"
else:
    args.device = "cpu"
args.mlp_dim =[64,128,128]

def train(args,epoch,dti_train_index, dti_train_label,dti_train_pos_mat,ddi_train_index, ddi_train_label,ppi_train_index, ppi_train_label,
          drug_data,protein_data):

    model.train()
    optimizer.zero_grad()
    Loss_func = nn.BCELoss()
    day_time = datetime.datetime.strftime(datetime.datetime.now(), '%m%d')
    embs_path = "data/Trained_embs"


    "pretrain_drug"
    "pretrain_protein"
    "CCL-ASPS"
    if args.model_type == "pretrain_drug":
        print("pretrain embeddings")
        drug_embs, ddi_score = model(drug_data,protein_data,ddi_train_index,ppi_train_index,dti_train_index,dti_train_label,dti_train_pos_mat)
        output = ddi_score
        labels = ddi_train_label
        loss_ddi = Loss_func(output, labels)
        loss = loss_ddi
        if (epoch + 1) % 50 == 0:
            torch.save(drug_embs.detach().cpu(),embs_path+"/drug_epoch_{}.pt".format(epoch+1))
        auc_dti_train = auc(output, labels)

    elif args.model_type == "pretrain_drug":
        protein_embs, ppi_score = model(drug_data,protein_data,ddi_train_index,ppi_train_index,dti_train_index,dti_train_label,dti_train_pos_mat,)
        output = ppi_score
        labels = ppi_train_label
        loss_ppi = Loss_func(output, labels)
        loss = loss_ppi
        if (epoch + 1) % 50 == 0:
            torch.save(protein_embs.detach().cpu(),embs_path+"/protein_epoch_{}.pt".format(epoch+1))
        auc_dti_train = auc(output, labels)

    elif args.model_type =="CCL-ASPS":
        output, loss1 = model(drug_data, protein_data, ddi_train_index, ppi_train_index, dti_train_index, dti_train_label, dti_train_pos_mat)
        labels = dti_train_label.type(torch.float32)
        loss_val = Loss_func(output, labels)
        loss = loss1*args.gamma + loss_val
        auc_dti_train = auc(output, labels)


    loss.backward()
    optimizer.step()
    # model._update_target()
    if (epoch + 1) % 10 == 0:
        print('Epoch {:04d} Train '.format(epoch + 1),
              'loss_all: {:.4f}'.format(loss.item()),
              'auc_dti_train: {:.4f}'.format(auc_dti_train))
    optimizer.zero_grad()

def test(args,dti_test_index,dti_test_labels, dti_train_pos_mat,ddi_test_index, ddi_test_labels,ppi_test_index, ppi_test_labels,drug_data,protein_data,):

    model.eval()
    with torch.no_grad():
        Loss_func = nn.BCELoss()
        if args.model_type == "pretrain_drug":
            drug_embs, ddi_score = model(drug_data,protein_data,ddi_test_index,ppi_test_index,dti_test_index,dti_test_labels,dti_train_pos_mat)
            output = ddi_score
            labels = ddi_test_labels
            loss_ddi = Loss_func(output, labels)
            loss = loss_ddi
            acc_test = accuracy(output, labels)

        elif args.model_type == "pretrain_protein":
            protein_embs, ppi_score = model(drug_data,protein_data,ddi_test_index,ppi_test_index,dti_test_index,dti_test_labels,dti_train_pos_mat,)
            output = ppi_score
            labels = ppi_test_labels
            loss_ppi = Loss_func(output, labels)
            loss = loss_ppi
            acc_test = accuracy(output, labels)

        elif args.model_type =="CCL-ASPS":
            output, loss1 = model(drug_data,protein_data,ddi_test_index,ppi_test_index,dti_test_index,dti_test_labels,dti_train_pos_mat)
            labels = dti_test_labels.type(torch.float32)
            loss_val = Loss_func(output, labels)
            loss = loss1 + loss_val
            acc_test = accuracy(output, labels)

        if (epoch + 1) % 10 == 0:
            print('train test loss: {:.4f}'.format(loss.item()),
                "train test acc: {}".format(acc_test))
    return acc_test, output, labels
for i in range(5):
    time_start = time.time()
    """
    model_type:
    "pretrain_drug"
    "pretrain_protein"
    "CCL-ASPS"
    """
    args.model_type ="CCL-ASPS"  # pretrain_drug pretain_protein

    # Train model
    t_total = time.time()
    acc_score = np.zeros(5)
    auc_score = np.zeros(5)
    aupr_score = np.zeros(5)
    fold_acc_score = np.zeros(5)
    fold_auc_score = np.zeros(5)
    fold_aupr_score = np.zeros(5)

    fold_acc_score1 = np.zeros([5,200])
    fold_auc_score1 = np.zeros([5,200])
    fold_aupr_score1 = np.zeros([5,200])


    """
    构建五折交叉数据，及对应的初始负样本
    """
    dti = get_DTI("./data/data/mat_drug_protein.txt")
    dti_pos_fold, dti_neg_data_fold, pos, neg = cross_k_folds(dti,args.fold_num,args.drug_num,args.protein_num)

    if args.model_type == "pretrain_drug" or args.model_type == "pretrain_protein": # apart from the contrastive mode, pretrain do not require  initial embeddings
        drug_embs = torch.randn((2,3))
        protein_embs = torch.randn((2,3))
    else:
        # load pre-trained embeddings
        drug_embs = torch.load("data/Trained_embs/drug")
        protein_embs = torch.load("data/Trained_embs/protein")

    drug_embs = drug_embs.to(args.device)
    protein_embs = protein_embs.to(args.device)

    args.drug_dim = args.drug_hidden
    args.protein_dim = args.protein_hidden

    DrugSimNet, ProteinSimNet,DrugSimNet_ori,ProteinSimNet_ori = load_sim(args.device)

    """
    get drug and protein structure data
    """
    if args.model_type == "pretrain_drug":
        d_x_all, d_x_num_index = multi2big_x(args.d_feat_matrix)
        d_edge_all, d_edge_num_index = multi2big_edge(args.d_adj_matrix, d_x_num_index)
        d_x_num_index = multi2big_batch(d_x_num_index)
        drug_data = (d_x_all.to(args.device), d_x_num_index.type(torch.int64).to(args.device),
                     d_edge_all.type(torch.int64).to(args.device), d_edge_num_index.type(torch.int).to(args.device))
    else:
        drug_data = None

    # get protein structure information , amino acid sequence represation, the relation between different amino acid
    if args.model_type == "pretrain_protein":
        protein_structure_edge = np.load('data/processed/protein_structure_edge_list.npy', allow_pickle=True)
        p_x_all, p_x_num_index = multi2big_x(args.p_feat_matrix)  # 将所有的蛋白质的氨基酸级的特征拼接在一起，x_num_index表示每个蛋白质的氨基酸个数
        p_edge_all, p_edge_num_index = multi2big_edge(args.p_adj_matrix, p_x_num_index)
        protein_data = (p_x_all.type(torch.float).to(args.device), p_x_num_index.type(torch.int).to(args.device),
                    p_edge_all.type(torch.int64).to(args.device),p_edge_num_index.type(torch.int64).to(args.device))
    else:
        protein_data = None

    ppi_path = "data/data/mat_protein_protein.txt"
    ddi_path = "data/data/mat_drug_drug.txt"
    ppi, ddi = get_ppi_ddi(ppi_path, ddi_path)
    ddi_pos_fold, ddi_neg_fold, ddi_pos, ddi_neg = cross_k_folds(ddi, args.fold_num, args.drug_num, args.drug_num)
    ppi_pos_fold, ppi_neg_fold, ppi_pos, ppi_neg = cross_k_folds(ppi, args.fold_num, args.protein_num, args.protein_num)

    args.drug_in_dim = args.drug_hidden
    args.protein_in_dim = args.protein_hidden

    for fold_num in range(args.fold_num):

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        dti_test_index = torch.cat((dti_pos_fold[fold_num],dti_neg_data_fold[fold_num]),dim=1).type(torch.int32)
        dti_test_labels = torch.cat((torch.ones_like(dti_pos_fold[fold_num][0,:]),torch.zeros_like(dti_neg_data_fold[fold_num][0,:])),dim=0).type(torch.int32)
        shuff_temp = np.arange(0, len(dti_test_labels))
        random.shuffle(shuff_temp)
        dti_test_index = dti_test_index[:, shuff_temp].to(args.device)
        dti_test_labels = dti_test_labels[shuff_temp].to(args.device)

        # 获得test用的ddi ppi数据
        ddi_test_index = torch.cat((ddi_pos_fold[fold_num], ddi_neg_fold[fold_num]), dim=-1).to(args.device)
        ddi_test_labels = torch.cat(
            (torch.ones(ddi_pos_fold[fold_num].shape[1]), torch.zeros(ddi_neg_fold[fold_num].shape[1]))).to(args.device)
        shuff_temp = np.arange(0, len(ddi_test_labels))
        random.shuffle(shuff_temp)
        ddi_test_index = ddi_test_index[:, shuff_temp]
        ddi_test_labels = ddi_test_labels[shuff_temp]

        ppi_test_index = torch.cat((ppi_pos_fold[fold_num], ppi_neg_fold[fold_num]), dim=-1).to(args.device)
        ppi_test_labels = torch.cat(
            (torch.ones(ppi_pos_fold[fold_num].shape[1]), torch.zeros(ppi_neg_fold[fold_num].shape[1]))).to(args.device)
        shuff_temp = np.arange(0, len(ppi_test_labels))
        random.shuffle(shuff_temp)
        ppi_test_index = ppi_test_index[:, shuff_temp]
        ppi_test_labels = ppi_test_labels[shuff_temp]

        # 获得训练用的ddi ppi dti正样本
        ddi_train_pos,_ = get_train_pos_from_fold(ddi_pos_fold,fold_num,args.device,args.drug_num,args.drug_num)
        ppi_train_pos,_ = get_train_pos_from_fold(ppi_pos_fold, fold_num,args.device,args.protein_num,args.protein_num)
        dti_train_pos,dti_train_pos_mat =  get_train_pos_from_fold(dti_pos_fold, fold_num,args.device,args.drug_num,args.protein_num)

        # args,train_pos_ddi_index,train_pos_ppi_index,drug_emb, protein_emb,DrugSimNet, ProteinSimNet
        model = CCL_ASPS(args,ddi_train_pos,ppi_train_pos,drug_embs,protein_embs,DrugSimNet,ProteinSimNet,DrugSimNet_ori,ProteinSimNet_ori)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model = model.to(args.device)

        for epoch in range(args.epochs):
            args.current_epoch = epoch
            dti_train_index, dti_train_label = gen_train_data(fold_num, dti_pos_fold, dti_neg_data_fold, args.drug_num, args.protein_num,args.device)
            ddi_train_index, ddi_train_label = gen_train_data(fold_num, ddi_pos_fold, ddi_neg_fold, args.drug_num,args.drug_num, args.device)
            ppi_train_index, ppi_train_label = gen_train_data(fold_num, ppi_pos_fold, ppi_neg_fold, args.protein_num,args.protein_num, args.device)

            if (epoch + 1) % 10 == 0:
                print('Epoch: {:04d}'.format(epoch + 1), 'Train_times:', fold_num)
            if args.model_type == "CCL-ASPS":
                train(args, epoch, dti_train_index, dti_train_label, dti_train_pos_mat, ddi_train_index,
                      ddi_train_label, ppi_train_index, ppi_train_label,drug_data, protein_data)
            else:
                train(args,epoch,dti_train_index, dti_train_label, dti_train_pos_mat,ddi_train_index, ddi_train_label,ppi_train_index, ppi_train_label,
                      drug_data,protein_data)
            test_acc, output, test_labels = test(args,dti_test_index,dti_test_labels, dti_train_pos_mat,
                    ddi_test_index, ddi_test_labels,ppi_test_index, ppi_test_labels,drug_data,protein_data)


            # todo 在model_type=reconstruct时，不使用以下代码
            if (epoch + 1) % 100 == 0:
                print('Epoch: {:04d}'.format(epoch + 1), 'Train_times:', fold_num)
                fold_acc_score1[fold_num][(epoch + 1) // 100] = round(accuracy(output, test_labels), 4)
                fold_auc_score1[fold_num][(epoch + 1) // 100] = round(auc(output, test_labels), 4)
                fold_aupr_score1[fold_num][(epoch + 1) // 100] = round(aupr(output, test_labels), 4)

                print("-***************************************----")
                print("acc socre:{}".format(fold_acc_score1[fold_num][(epoch + 1) // 100]))
                print("auc socre:{}".format( fold_auc_score1[fold_num][(epoch + 1) // 100]))
                print("aupr score:{}".format(fold_aupr_score1[fold_num][(epoch + 1) // 100]))

        # ：final test result for each fold
        test_acc, output, test_labels = test(args,dti_test_index,dti_test_labels, dti_train_pos_mat,
                                                 ddi_test_index, ddi_test_labels,ppi_test_index, ppi_test_labels,drug_data,protein_data)
        fold_test = test_acc
        fold_acc_score[fold_num] = round(accuracy(output, test_labels), 4)
        fold_auc_score[fold_num] = round(auc(output, test_labels), 4)
        fold_aupr_score[fold_num] = round(aupr(output, test_labels), 4)

        print("-***************************************----")
        print("acc socre:{} avg:{}".format(fold_acc_score, np.mean(fold_acc_score)))
        print("auc socre:{} avg:{}".format(fold_auc_score, np.mean(fold_auc_score)))
        print("aupr score:{} avg:{}".format(fold_aupr_score, np.mean(fold_aupr_score)))






