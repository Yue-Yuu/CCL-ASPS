import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GCNConv
from models.PGAT import GATConv as PGATConv

class DAE(nn.Module):
    def __init__(self, input_data, n_input, n_hidden, scale=0.2):
        super(DAE,self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = F.softplus
        self.training_scale = scale
        self.W1 = torch.nn.Linear(n_input, n_hidden)
        self.W2 = torch.nn.Linear(n_hidden,n_input)
        nn.init.xavier_normal_(self.W1.weight, gain=1)
        nn.init.constant(self.W1.bias, 0.)
        nn.init.constant(self.W2.weight,0.)
        nn.init.constant(self.W2.bias, 0.)
        self.X = input_data

    def forward(self, X):
        noise = torch.randn((self.n_input,)).cuda()
        noisex = X + self.training_scale * noise
        hidden = self.W1(noisex)
        # todo softplus 函数
        hidden = self.transfer(hidden)
        reconstruction = self.W2(hidden)
        cost = 0.5 * torch.pow(torch.sub(reconstruction, X), 2.0).sum()
        return cost

    def get_hidden(self):
        noise = torch.randn((self.n_input,)).cuda()
        noisex = self.X + self.training_scale * noise
        hidden = self.W1(noisex)
        hidden = self.transfer(hidden)
        return hidden

"""
对比模块
"""
class Model_Contrast(nn.Module):
    # 需要正样本pos、负样本neg，头尾所有节点的表示drug_embs、protein_embs
    def __init__(self, hidden_dim, tau, lam):
        super(Model_Contrast,self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            # nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):

        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        # todo 23-08-29 修改sim_matrix计算方法
        # sim_matrix = torch.exp(dot_numerator/self.tau) / torch.exp(dot_denominator/self.tau)
        return sim_matrix


    def forward(self, v1_embs, v2_embs, pos=None, neg=None):

        v1_embs = self.proj(v1_embs)
        v2_embs = self.proj(v2_embs)
        matrix_1to2 = self.sim(v1_embs, v2_embs)

        sim_pos_1to2 = matrix_1to2.mul(pos)
        sim_neg_1to2 = matrix_1to2.mul(neg)
        sum_1to2 = (sim_pos_1to2).sum() / (sim_pos_1to2 + sim_neg_1to2).sum()
        loss_1to2 = -torch.log(sum_1to2)

        return loss_1to2

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, contrast = 1, bias=True):
        super(GCN, self).__init__()
        out_ft = int(out_ft)
        self.fc = nn.Linear(in_ft, out_ft, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.act = nn.PReLU()
        self.eps = 0.01

    def forward(self, seq, adj, contrast=1):

        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        # out = F.elu(out)


        # out = F.dropout(out, 0.5)
        if contrast == 1:
            # out = F.dropout(out, 0.3)
            random_noise = torch.rand(out.shape).cuda()
            out = out + torch.multiply(torch.sign(out), F.normalize(random_noise,p=2, dim=1)) + self.eps

        return self.act(out)


class model_reconstruct(nn.Module):

    def __init__(self, args, drug_emb, protein_emb,DrugSimNet, ProteinSimNet,DrugSimNet_ori,ProteinSimNet_ori):
        super(model_reconstruct, self).__init__()

        self.drug_dim = args.drug_dim
        self.protein_dim = args.protein_dim
        self.args = args
        self.dropout = 0.
        self.drug_emb = drug_emb
        self.protein_emb = protein_emb
        self.momentum = 0.995
        self.gcn_drug_outdim = args.drug_dim
        self.gcn_protein_outdim = args.protein_dim
        self.tau = args.tau

        self.input_dim = self.gcn_drug_outdim + self.gcn_protein_outdim

        # 1.
        temp_drug = torch.randn(self.drug_emb.shape).to(args.device)
        temp_protein = torch.randn(self.protein_emb.shape).to(args.device)
        self.drug_emb_aug = drug_emb +  temp_drug
        self.protein_emb_aug = protein_emb +  temp_protein

        self.DrugSimNet = DrugSimNet
        self.ProteinSimNet = ProteinSimNet

        self.DrugSimNet_ori = DrugSimNet_ori
        self.ProteinSimNet_ori = ProteinSimNet_ori
        self.neg_num = 5

        self.contrast_drug_2 = Model_Contrast(self.gcn_drug_outdim, args.tau, args.lam)

        """
        PGAT
        """
        nhead = 4
        negative_slope = 0.2
        dropout = 0.2
        self.PGAT1_drug = PGATConv(args.drug_dim, self.gcn_drug_outdim,nhead,True,negative_slope,dropout)
        self.PGAT2_drug = PGATConv(args.drug_dim, self.gcn_drug_outdim, nhead, True, negative_slope,dropout)
        self.PGAT3_drug = PGATConv(args.drug_dim, self.gcn_drug_outdim, nhead, True, negative_slope,dropout)
        self.PGAT4_drug = PGATConv(args.drug_dim, self.gcn_drug_outdim, nhead, True, negative_slope,dropout)
        self.PGAT1_protein = PGATConv(args.protein_dim, self.gcn_protein_outdim,nhead, True, negative_slope,dropout)
        self.PGAT2_protein = PGATConv(args.protein_dim, self.gcn_protein_outdim,nhead, True, negative_slope,dropout)
        self.PGAT3_protein = PGATConv(args.protein_dim, self.gcn_protein_outdim,nhead, True, negative_slope,dropout)

        """第二层PGAT"""
        self.PGAT12_drug = PGATConv(self.gcn_drug_outdim*nhead, self.gcn_drug_outdim,1, False, negative_slope,dropout)
        self.PGAT22_drug = PGATConv(self.gcn_drug_outdim*nhead, self.gcn_drug_outdim,1, False, negative_slope,dropout)
        self.PGAT32_drug = PGATConv(self.gcn_drug_outdim*nhead, self.gcn_drug_outdim,1, False, negative_slope,dropout)
        self.PGAT42_drug = PGATConv(self.gcn_drug_outdim*nhead, self.gcn_drug_outdim,1, False, negative_slope,dropout)
        self.PGAT12_protein = PGATConv(self.gcn_protein_outdim*nhead, self.gcn_protein_outdim,1,False, negative_slope,dropout)
        self.PGAT22_protein = PGATConv(self.gcn_protein_outdim*nhead, self.gcn_protein_outdim,1,False, negative_slope,dropout)
        self.PGAT32_protein = PGATConv(self.gcn_protein_outdim*nhead, self.gcn_protein_outdim,1,False, negative_slope,dropout)

        """
        卷积融合来自多个sim net的embs
        """
        self.drug_conv1 = nn.Conv1d(in_channels=4,out_channels=16,kernel_size=3,stride=1,padding=1, )
        self.drug_pooling1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.drug_linear1 = torch.nn.Linear(self.gcn_drug_outdim * 8, self.gcn_drug_outdim)

        self.protein_conv1 = nn.Conv1d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1, )
        self.protein_pooling1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.protein_linear1 = torch.nn.Linear(self.gcn_protein_outdim * 6, self.gcn_protein_outdim)
        self.proj_drug = nn.Sequential(
            nn.Linear(args.drug_dim, args.drug_dim),
            nn.ELU(),
            # nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj_protein = nn.Sequential(
            nn.Linear(args.drug_dim, args.drug_dim),
            nn.ELU(),
            # nn.Linear(hidden_dim, hidden_dim)
        )



    def get_contrast_pair(self,args, feat_sim,sim_net):

        current_epoch = args.current_epoch
        total_epoch = args.epochs
        device = args.device

        max_pos_num=1

        max_neg_num = len(feat_sim) - max_pos_num
        beta = self.args.beta
        neg_num = int(max_neg_num * current_epoch*beta / total_epoch) + 1

        # todo 直接获得所有正负样本
        pos_num = max_pos_num

        # 获取正样本
        vals, indices = sim_net.topk(k=pos_num, dim=1, largest=True, sorted=True)
        vals = torch.where(vals>0,1,0).to(dtype=torch.float32)
        sim_drug_index_row = torch.arange(len(sim_net), dtype=torch.int32).repeat_interleave(pos_num, dim=0).unsqueeze(0)
        sim_drug_index_col = torch.zeros_like(sim_drug_index_row)
        sim_drug_index = torch.cat((sim_drug_index_row, sim_drug_index_col), dim=0)
        sim_drug_index[1, :] = indices.view(-1)
        pos = torch.zeros_like(sim_net)

        pos[sim_drug_index[0, :], sim_drug_index[1, :]] = 1
        pos = pos.to(device)

        # # 获取负样本
        neg_vals, neg_indices = (-sim_net).topk(k=neg_num, dim=1, largest=True, sorted=True)
        neg_vals = torch.where(neg_vals==0,1,0).to(dtype=torch.float32)
        neg_sim_drug_index_row = torch.arange(len(sim_net), dtype=torch.int32).repeat_interleave(neg_num,
                                                                                                 dim=0).unsqueeze(0)
        neg_sim_drug_index_col = torch.zeros_like(neg_sim_drug_index_row)
        neg_sim_drug_index = torch.cat((neg_sim_drug_index_row, neg_sim_drug_index_col), dim=0)
        neg_sim_drug_index[1, :] = neg_indices.view(-1)
        neg = torch.zeros_like(sim_net)
        neg[neg_sim_drug_index[0, :], neg_sim_drug_index[1, :]] = 1
        neg = neg.to(device)


        if (not feat_sim is None) and (current_epoch > 100):
            # 获取feature sim的负样本
            neg_vals_fea, neg_indices_fea = (-feat_sim).topk(k=neg_num, dim=1, largest=True, sorted=True)
            neg_sim_drug_index_row_fea = torch.arange(len(feat_sim), dtype=torch.int32).repeat_interleave(neg_num,dim=0).unsqueeze(0)
            neg_sim_drug_index_col_fea = torch.zeros_like(neg_sim_drug_index_row_fea)
            neg_sim_drug_index_fea = torch.cat((neg_sim_drug_index_row_fea, neg_sim_drug_index_col_fea), dim=0)
            neg_sim_drug_index_fea[1, :] = neg_indices_fea.view(-1)
            neg_fea = torch.zeros_like(feat_sim)
            neg_fea[neg_sim_drug_index_fea[0, :], neg_sim_drug_index_fea[1, :]] = 1
            neg_fea = neg_fea.to(device)

            # 取并集
            neg = neg + neg_fea
            neg = torch.where(neg >= 1,1,0)

        # 节点自身为pos pair
        eye = torch.arange(len(feat_sim), dtype=torch.int32)
        pos[eye, eye] = 1
        # 去除负样本pair中存在的pos pair
        neg = neg - pos
        neg = torch.where(neg > 0, 1, 0)

        return pos, neg


    def forward(self,dti_index,dti_labels,train_pos_dti,drug_emb=None,protein_emb=None):


        if drug_emb is None:
            drug_emb = self.drug_emb
            protein_emb = self.protein_emb
        DrugSimNet = self.DrugSimNet
        ProteinSimNet = self.ProteinSimNet

        d_edge_row_1 = torch.where(DrugSimNet[0] > 0)[0].unsqueeze(dim=0)
        d_edge_col_1 = torch.where(DrugSimNet[0] > 0)[1].unsqueeze(dim=0)
        d_edge_index_1 = torch.cat((d_edge_row_1,d_edge_col_1),dim= 0).to(self.args.device)
        d_alpha_ori_1 = self.DrugSimNet_ori[0][d_edge_row_1,d_edge_col_1].to(self.args.device)
        d_alpha_ori_1 = d_alpha_ori_1.view(-1,1)

        d_edge_row_2 = torch.where(DrugSimNet[1] > 0)[0].unsqueeze(dim=0)
        d_edge_col_2 = torch.where(DrugSimNet[1] > 0)[1].unsqueeze(dim=0)
        d_edge_index_2 = torch.cat((d_edge_row_2, d_edge_col_2), dim=0).to(self.args.device)
        d_alpha_ori_2 = self.DrugSimNet_ori[1][d_edge_row_2, d_edge_col_2].to(self.args.device)
        d_alpha_ori_2 = d_alpha_ori_2.view(-1, 1)

        d_edge_row_3 = torch.where(DrugSimNet[2] > 0)[0].unsqueeze(dim=0)
        d_edge_col_3 = torch.where(DrugSimNet[2] > 0)[1].unsqueeze(dim=0)
        d_edge_index_3 = torch.cat((d_edge_row_3, d_edge_col_3), dim=0).to(self.args.device)
        d_alpha_ori_3 = self.DrugSimNet_ori[2][d_edge_row_3, d_edge_col_3].to(self.args.device)
        d_alpha_ori_3 = d_alpha_ori_3.view(-1, 1)

        d_edge_row_4 = torch.where(DrugSimNet[3] > 0)[0].unsqueeze(dim=0)
        d_edge_col_4 = torch.where(DrugSimNet[3] > 0)[1].unsqueeze(dim=0)
        d_edge_index_4 = torch.cat((d_edge_row_4, d_edge_col_4), dim=0).to(self.args.device)
        d_alpha_ori_4 = self.DrugSimNet_ori[3][d_edge_row_4, d_edge_col_4].to(self.args.device)
        d_alpha_ori_4 = d_alpha_ori_4.view(-1, 1)

        p_edge_row_1 = torch.where(ProteinSimNet[0] > 0)[0].unsqueeze(dim=0)
        p_edge_col_1 = torch.where(ProteinSimNet[0] > 0)[1].unsqueeze(dim=0)
        p_edge_index_1 = torch.cat((p_edge_row_1, p_edge_col_1), dim=0).to(self.args.device)
        p_alpha_ori_1 = self.ProteinSimNet_ori[0][p_edge_row_1, p_edge_col_1].to(self.args.device)
        p_alpha_ori_1 = p_alpha_ori_1.view(-1, 1)

        p_edge_row_2 = torch.where(ProteinSimNet[1] > 0)[0].unsqueeze(dim=0)
        p_edge_col_2 = torch.where(ProteinSimNet[1] > 0)[1].unsqueeze(dim=0)
        p_edge_index_2 = torch.cat((p_edge_row_2, p_edge_col_2), dim=0).to(self.args.device)
        p_alpha_ori_2 = self.ProteinSimNet_ori[1][p_edge_row_2, p_edge_col_2].to(self.args.device)
        p_alpha_ori_2 = p_alpha_ori_2.view(-1, 1)

        p_edge_row_3 = torch.where(ProteinSimNet[2] > 0)[0].unsqueeze(dim=0)
        p_edge_col_3 = torch.where(ProteinSimNet[2] > 0)[1].unsqueeze(dim=0)
        p_edge_index_3 = torch.cat((p_edge_row_3, p_edge_col_3), dim=0).to(self.args.device)
        p_alpha_ori_3 = self.ProteinSimNet_ori[0][p_edge_row_3, p_edge_col_3].to(self.args.device)
        p_alpha_ori_3 = p_alpha_ori_3.view(-1, 1)


        """第一层PGAT"""
        drug_embs_1 = self.PGAT1_drug(drug_emb, d_edge_index_1,alpha_ori=d_alpha_ori_1)
        drug_embs_2 = self.PGAT2_drug(drug_emb, d_edge_index_2, alpha_ori=d_alpha_ori_2)
        drug_embs_3 = self.PGAT3_drug(drug_emb, d_edge_index_3, alpha_ori=d_alpha_ori_3)
        drug_embs_4 = self.PGAT4_drug(drug_emb, d_edge_index_4, alpha_ori=d_alpha_ori_4)
        protein_embs_1 = self.PGAT1_protein(protein_emb, p_edge_index_1, alpha_ori=p_alpha_ori_1)
        protein_embs_2 = self.PGAT2_protein(protein_emb, p_edge_index_2, alpha_ori=p_alpha_ori_2)
        protein_embs_3 = self.PGAT3_protein(protein_emb, p_edge_index_3, alpha_ori=p_alpha_ori_3)

        """第二层PGAT"""
        drug_embs_12 = self.PGAT12_drug(drug_embs_1, d_edge_index_1, alpha_ori=d_alpha_ori_1)
        drug_embs_22 = self.PGAT22_drug(drug_embs_2, d_edge_index_2, alpha_ori=d_alpha_ori_2)
        drug_embs_32 = self.PGAT32_drug(drug_embs_3, d_edge_index_3, alpha_ori=d_alpha_ori_3)
        drug_embs_42 = self.PGAT42_drug(drug_embs_4, d_edge_index_4, alpha_ori=d_alpha_ori_4)

        protein_embs_12 = self.PGAT12_protein(protein_embs_1, p_edge_index_1, alpha_ori=p_alpha_ori_1)
        protein_embs_22 = self.PGAT22_protein(protein_embs_2, p_edge_index_2, alpha_ori=p_alpha_ori_2)
        protein_embs_32 = self.PGAT32_protein(protein_embs_3, p_edge_index_3, alpha_ori=p_alpha_ori_3)

        """
        todo 使用卷积操作融合多个sim net的特征
        """
        drug_embs_12 = drug_embs_12.unsqueeze(1)
        drug_embs_22 = drug_embs_22.unsqueeze(1)
        drug_embs_32 = drug_embs_32.unsqueeze(1)
        drug_embs_42 = drug_embs_42.unsqueeze(1)
        drug_embs_all = torch.cat((drug_embs_12,drug_embs_22,drug_embs_32,drug_embs_42),dim=1)

        drug_embs_all = self.drug_conv1(drug_embs_all)
        drug_embs_all = self.drug_pooling1(drug_embs_all)
        drug_embs_all = torch.flatten(drug_embs_all, 1, 2)
        drug_embs_all = F.dropout(drug_embs_all, 0.1)
        drug_embs_all = self.drug_linear1(drug_embs_all)
        drug_embs = drug_embs_all

        protein_embs_12 = protein_embs_12.unsqueeze(1)
        protein_embs_22 = protein_embs_22.unsqueeze(1)
        protein_embs_32 = protein_embs_32.unsqueeze(1)
        protein_embs_all = torch.cat((protein_embs_12, protein_embs_22, protein_embs_32), dim=1)

        protein_embs_all = self.protein_conv1(protein_embs_all)
        protein_embs_all = self.protein_pooling1(protein_embs_all)
        protein_embs_all = torch.flatten(protein_embs_all, 1, 2)
        protein_embs_all = F.dropout(protein_embs_all, 0.1)
        protein_embs_all = self.protein_linear1(protein_embs_all)
        protein_embs = protein_embs_all

        drug_norm = torch.norm(drug_embs, dim=-1, keepdim=True)
        drug_dot_numerator = torch.mm(drug_embs, drug_embs.t())
        drug_dot_denominator = torch.mm(drug_norm, drug_norm.t())
        drug_re_sim = drug_dot_numerator / drug_dot_denominator
        drug_re_sim = torch.sigmoid(drug_re_sim)

        protein_norm = torch.norm(protein_embs, dim=-1, keepdim=True)
        protein_dot_numerator = torch.mm(protein_embs, protein_embs.t())
        protein_dot_denominator = torch.mm(protein_norm, protein_norm.t())
        protein_re_sim = protein_dot_numerator / protein_dot_denominator
        protein_re_sim = torch.sigmoid(protein_re_sim)

        pos_pair_drug_0,neg_pair_drug_0 = self.get_contrast_pair(self.args,drug_re_sim,DrugSimNet[0])
        pos_pair_protein_0, neg_pair_protein_0 = self.get_contrast_pair(self.args, protein_re_sim,ProteinSimNet[0])
        pos_pair_drug_1, neg_pair_drug_1 = self.get_contrast_pair(self.args, drug_re_sim, DrugSimNet[1])
        pos_pair_protein_1, neg_pair_protein_1 = self.get_contrast_pair(self.args, protein_re_sim, ProteinSimNet[1])
        pos_pair_drug_2, neg_pair_drug_2 = self.get_contrast_pair(self.args, drug_re_sim, DrugSimNet[2])
        pos_pair_protein_2, neg_pair_protein_2 = self.get_contrast_pair(self.args, protein_re_sim, ProteinSimNet[2])
        pos_pair_drug_3, neg_pair_drug_3 = self.get_contrast_pair(self.args, drug_re_sim, DrugSimNet[3])

        loss1d =self.contrast_drug_2(drug_embs_12.squeeze(1),drug_embs_all,pos_pair_drug_0,neg_pair_drug_0)
        loss2d = self.contrast_drug_2(drug_embs_22.squeeze(1), drug_embs_all,pos_pair_drug_1,neg_pair_drug_1)
        loss3d = self.contrast_drug_2(drug_embs_32.squeeze(1), drug_embs_all,pos_pair_drug_2,neg_pair_drug_2)
        loss4d = self.contrast_drug_2(drug_embs_42.squeeze(1), drug_embs_all,pos_pair_drug_3,neg_pair_drug_3)
        loss1p = self.contrast_drug_2(protein_embs_12.squeeze(1), protein_embs_all,pos_pair_protein_0, neg_pair_protein_0)
        loss2p = self.contrast_drug_2(protein_embs_22.squeeze(1), protein_embs_all,pos_pair_protein_1, neg_pair_protein_1)
        loss3p = self.contrast_drug_2(protein_embs_32.squeeze(1), protein_embs_all,pos_pair_protein_2, neg_pair_protein_2)
        lossContrast =loss1d+loss2d+loss3d+loss4d+loss1p+loss2p+loss3p
        loss = lossContrast

        return drug_embs,protein_embs, loss
