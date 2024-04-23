import os.path

import numpy as np
import torch
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score

def accuracy(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    # threshold_value = (outputs.max() + outputs.min()) /2.
    threshold_value = 0.5
    outputs = outputs.ge(threshold_value).type(torch.int32)
    labels = labels.type(torch.int32)
    corrects = (1 - (outputs ^ labels)).type(torch.int32)
    if labels.size() == 0:
        return np.nan
    return corrects.sum().item() / labels.size()[0]


def precision(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.ge(0.5).type(torch.int32).detach().cpu().numpy()
    return precision_score(labels, outputs)


def recall(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.ge(0.5).type(torch.int32).detach().cpu().numpy()
    return recall_score(labels, outputs)


def specificity(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.ge(0.5).type(torch.int32).detach().cpu().numpy()
    return recall_score(labels, outputs, pos_label=0)


def f1(outputs, labels):
    return (precision(outputs, labels) + recall(outputs, labels)) / 2


def mcc(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    outputs = outputs.ge(0.5).type(torch.float64)
    labels = labels.type(torch.float64)
    true_pos = (outputs * labels).sum()
    true_neg = ((1 - outputs) * (1 - labels)).sum()
    false_pos = (outputs * (1 - labels)).sum()
    false_neg = ((1 - outputs) * labels).sum()
    numerator = true_pos * true_neg - false_pos * false_neg
    deno_2 = outputs.sum() * (1 - outputs).sum() * labels.sum() * (1 - labels).sum()
    if deno_2 == 0:
        return np.nan
    return (numerator / (deno_2.type(torch.float32).sqrt())).item()


def auc(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    return roc_auc_score(labels, outputs)

def aupr(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    return average_precision_score(labels, outputs)

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def jaccard_sim(m):
    sum_A = m.sum(-1)
    sum_A = np.expand_dims(sum_A,0)
    A_and_B = np.matmul(m, m.T)
    A_or_B = sum_A + sum_A.T - A_and_B

    sim_m = np.true_divide(A_and_B, A_or_B)
    # sim_m.replace([np.inf,-np.inf], np.nan)
    sim_m = np.nan_to_num(sim_m)
    np.inf

    return sim_m


def joint(maxiter, restartProb, Nets, path):
    # Q 用于拼接所有的net得到的结果矩阵
    Q = np.zeros(1)
    # 该for循环用于计算所有相似度网络上的RWR结果，并拼接得到的矩阵
    for net_name in Nets:
        file_name = str(net_name) + ".txt"
        file_path = os.path.join(path, file_name)
        net = np.loadtxt(file_path)

        # sum_i要么为1，要么为0，0表示该节点是孤立的节点，在该网络下没有相似的邻居
        sum_net = net.sum(-1)

        """
        diffusionRWR
        """
        # 一维向量长度为矩阵i的大小，保存的是矩阵i上需要self-loop的对应位置的值
        diagValue = np.zeros(net.shape[0])
        # 对于孤立节点，对角值应为1
        diagValue = np.where(sum_net == 0, 1, diagValue)
        # 构建孤立节点的对角矩阵
        diag_net = np.diag(diagValue)
        # 孤立节点加上self-loop
        net = net + diag_net

        # normalize每一行
        sum_net = net.sum(-1)
        sum_net = np.expand_dims(sum_net,axis=1)

        # np.divide(net, sum_net)得到的结果是net每一行和sum_net每一行所有的除法
        normNet = np.divide(net, sum_net)

        """
        Personalized PageRank
        构建diffusion state matrices：
        """
        restart = np.eye(net.shape[0])
        diffNet = np.eye(net.shape[0])
        # for循环得到 diffNet
        for i in range(maxiter):
            # np两个矩阵进行矩阵乘法，使用dot或者符号@
            net_new = (1 - restartProb) * normNet @ diffNet + restartProb * restart
            """计算 ”diffNet-net_new"的Frobenius norm（Frobenius范数）"""
            # 求矩阵中每个元素的平方和
            square_net = np.square(diffNet-net_new).sum()
            delta = np.sqrt(square_net)
            diffNet = net_new
            if delta < 1e-6:
                break
        if Q.shape[0] == 1:
            Q = diffNet
        else:
            Q = np.concatenate((Q, diffNet), axis=1)

    nnode = Q.shape[0]
    alpha = 1 / nnode
    Q = np.log(Q + alpha) - np.log(alpha)

    return Q

'''
from view generate sub_view
return sub_view's network
'''
def gen_view(view):

    rate = 0.2
    num = view.shape[0] * view.shape[1]

    mask = np.ones_like(view)
    mask1 = np.random.randint(0, num, size=int(num * rate))
    row = np.floor_divide(mask1, view.shape[1])
    col = np.mod(mask1, view.shape[1])
    mask[row, col] = 0
    view1 = np.multiply(view, mask)

    mask = np.ones_like(view)
    mask2 = np.random.randint(0, num, size=int(num * rate))
    row = np.floor_divide(mask2, view.shape[1])
    col = np.mod(mask2, view.shape[1])
    mask[row, col] = 0
    view2 = np.multiply(view, mask)

    # todo 以下为更改第二个子图生成
    view2 = view


    i = np.arange(view.shape[0])
    view1[i,i] = 1
    view2[i,i] = 1

    return view1, view2


def load_sim(device):

    "data/similarity/sim_mat_drug_disease.txt",
    "data/similarity/sim_mat_drug_drug.txt",
    "data/similarity/sim_mat_drug.txt",
    "data/similarity/sim_mat_drug_se.txt"
    sim_mat_drug_disease = np.loadtxt("./data/similarity/sim_mat_drug_disease.txt")
    sim_mat_drug_drug = np.loadtxt("./data/similarity/sim_mat_drug_drug.txt")
    sim_mat_drug = np.loadtxt("./data/similarity/sim_mat_drug.txt")
    sim_mat_drug_se = np.loadtxt("./data/similarity/sim_mat_drug_se.txt")

    "data/similarity/sim_mat_protein_disease.txt",
    "data/similarity/sim_mat_protein_protein.txt",
    "data/similarity/sim_mat_protein_normalize.txt"
    sim_mat_protein_disease = np.loadtxt("./data/similarity/sim_mat_protein_disease.txt")
    sim_mat_protein_protein = np.loadtxt("./data/similarity/sim_mat_protein_protein.txt")
    sim_mat_protein = np.loadtxt("./data/similarity/sim_mat_protein.txt")
    sim_mat_protein = sim_mat_protein / 100

    """将drug相似度网络放在一起，protein相似度网络放在一起"""
    DrugSimNet = []
    ProteinSimNet = []
    DrugSimNet_ori = []
    ProteinSimNet_ori = []
    DrugSimNet.append(sim_mat_drug_disease)
    DrugSimNet.append(sim_mat_drug_drug)
    DrugSimNet.append(sim_mat_drug)
    DrugSimNet.append(sim_mat_drug_se)

    ProteinSimNet.append(sim_mat_protein_disease)
    ProteinSimNet.append(sim_mat_protein_protein)
    ProteinSimNet.append(sim_mat_protein)

    DrugSimNet_ori.append(sim_mat_drug_disease)
    DrugSimNet_ori.append(sim_mat_drug_drug)
    DrugSimNet_ori.append(sim_mat_drug)
    DrugSimNet_ori.append(sim_mat_drug_se)

    ProteinSimNet_ori.append(sim_mat_protein_disease)
    ProteinSimNet_ori.append(sim_mat_protein_protein)
    ProteinSimNet_ori.append(sim_mat_protein)

    idx_drug = np.arange(len(DrugSimNet[0]))
    for i, net in enumerate(DrugSimNet):
        net = np.where(net > 0.5, 1, 0)
        net[idx_drug,idx_drug] = 0
        net = preprocess_adj(net)
        net = torch.Tensor(net)
        DrugSimNet[i] = net.to(device)

        net_ori = DrugSimNet_ori[i]
        # net_ori[idx_drug, idx_drug] = 0
        # net_ori = preprocess_adj(net_ori)
        net_ori = torch.Tensor(net_ori)
        DrugSimNet_ori[i] = net_ori.to(device)

    idx_protein = np.arange(len(ProteinSimNet[0]))
    for i, net in enumerate(ProteinSimNet):
        net = np.where(net > 0.5, 1, 0)
        net[idx_protein, idx_protein] = 0
        net = preprocess_adj(net)
        net = torch.Tensor(net)
        ProteinSimNet[i] = net.to(device)

        net_ori = ProteinSimNet_ori[i]
        # net_ori[idx_protein, idx_protein] = 0
        # net_ori = preprocess_adj(net_ori)
        net_ori = torch.Tensor(net_ori)
        ProteinSimNet_ori[i] = net_ori.to(device)

    return DrugSimNet, ProteinSimNet,DrugSimNet_ori,ProteinSimNet_ori


def multi2big_x(x_path):
    x_ori = torch.load(x_path)
    x_cat = torch.zeros(1, x_ori[0].shape[1])
    x_num_index = torch.zeros(len(x_ori))
    for i in range(len(x_ori)):
        x_now = torch.tensor(x_ori[i])
        x_num_index[i] = torch.tensor(x_now.size(0))
        x_cat = torch.cat((x_cat, x_now), 0)
    return x_cat[1:, :], x_num_index

def multi2big_batch(x_num_index):
    num_sum = x_num_index.sum()
    num_sum = num_sum.int()
    batch = torch.zeros(num_sum)
    count = 1
    for i in range(1,len(x_num_index)):
        zj1 = x_num_index[:i]
        zj11 = zj1.sum()
        zj11 = zj11.int()
        zj22 = zj11 + x_num_index[i]
        zj22 = zj22.int()
        size1 = x_num_index[i]
        size1 = size1.int()
        tc = count * torch.ones(size1)
        batch[zj11:zj22] = tc
        test = batch[zj11:zj22]
        count = count + 1
    batch = batch.type(torch.int64)
    return batch

def multi2big_edge(edge_path, num_index):
    if edge_path[-2:] == "py":
        edge_ori = np.load(edge_path, allow_pickle=True)
        for i in range(len(edge_ori)):
            edge_ori[i] = torch.tensor(edge_ori[i])
        edge_ori = list(edge_ori)
    elif edge_path[-2:] =="pt":
        edge_ori = torch.load(edge_path)
    edge_cat = torch.zeros(2, 1)
    edge_num_index = torch.zeros(len(num_index))
    for i in range(len(num_index)):
        edge_index_p = edge_ori[i]
        edge_index_p = np.asarray(edge_index_p)
        if not len(edge_index_p) == 2:
            edge_index_p = torch.tensor(edge_index_p.T)
        else:
            edge_index_p = torch.tensor(edge_index_p)
        edge_num_index[i] = torch.tensor(edge_index_p.shape[1])
        if i == 0:
            offset = 0
        else:
            zj = torch.tensor(num_index[:i])
            offset = zj.sum()
        edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
    return edge_cat[:, 1:], edge_num_index


def drug_multi2big_batch(x_num_index):
    num_sum = x_num_index.sum()
    num_sum = num_sum.int()
    batch = torch.zeros(num_sum)
    count = 1
    for i in range(1,len(x_num_index)):
        zj1 = x_num_index[:i]
        zj11 = zj1.sum()
        zj11 = zj11.int()
        zj22 = zj11 + x_num_index[i]
        zj22 = zj22.int()
        size1 = x_num_index[i]
        size1 = size1.int()
        tc = count * torch.ones(size1)
        batch[zj11:zj22] = tc
        test = batch[zj11:zj22]
        count = count + 1
    batch = batch.int()
    return batch

def drug_multi2big_edge(edge_ori, num_index):
    edge_cat = torch.zeros(2, 1)
    edge_num_index = torch.zeros(len(num_index))
    for i in range(len(num_index)):
        edge_index_p = edge_ori[i]
        edge_index_p = np.asarray(edge_index_p)
        edge_index_p = torch.tensor(edge_index_p.T)
        edge_num_index[i] = torch.tensor(edge_index_p.size(1))
        if i == 0:
            offset = 0
        else:
            zj = torch.tensor(num_index[:i])
            offset = zj.sum()
        edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
    return edge_cat[:, 1:], edge_num_index



def get_ppi_ddi(ppi_path,ddi_path):
    mat_ppi = np.loadtxt(ppi_path)
    mat_ddi = np.loadtxt(ddi_path)
    ppi_index = np.where(mat_ppi == 1)
    ddi_index = np.where(mat_ddi == 1)
    ppi = np.ones(shape=(len(ppi_index[0]),3))
    ddi = np.ones(shape=(len(ddi_index[0]),3))

    ppi[:,0] = ppi_index[0]
    ppi[:, 1] = ppi_index[1]
    ppi = ppi.astype(int)

    ddi[:, 0] = ddi_index[0]
    ddi[:, 1] = ddi_index[1]
    ddi = ddi.astype(int)
    return ppi,ddi


def get_train_pos_from_fold(pos_fold,fold_num,device,row_num, col_num):
    train_pos  = [0]
    for i in range(len(pos_fold)):
        if not isinstance(pos_fold[0], torch.Tensor):
            temp_fold = torch.tensor(pos_fold[i])
        else:
            temp_fold = pos_fold[i]
        if i == fold_num:
            continue
        elif len(train_pos) == 1:
            train_pos = temp_fold
        else:
             train_pos = torch.cat((train_pos,temp_fold),dim=-1)

    train_pos_matrix = torch.zeros((row_num, col_num)).to(device)
    train_pos_matrix[train_pos[0,:],train_pos[1,:]] = 1

    return train_pos.to(device),train_pos_matrix

def get_contrast_pair(args,sim_net, min,max, trained_sim,device,current_epoch,total_epoch,threshold=0.8):
    """
    sim_net: reconstructed or meta-path based sim net
    min: control the threshold for negative sample pairs
    max: control the threshold for positive sample pairs
    pos: positive sample pairs already allocated
    neg: negative sample pairs already allocated
    feature_sim: similarity based on trained feature
    """
    """
    以rate，每个node选择固定数量的pos样本和neg样本
    """

    rate = 0.005
    max_pos_num = int(len(sim_net) * rate)
    pos_num = int(max_pos_num * current_epoch /total_epoch)+1

    max_neg_num = len(sim_net) - max_pos_num
    # max_neg_num = int(len(sim_net)*0.5)
    neg_num = int(max_neg_num* current_epoch /total_epoch)+1

    # 获取正样本
    vals,indices = sim_net.topk(k=pos_num,dim=1,largest=True,sorted=True)
    sim_drug_index_row = torch.arange(len(sim_net), dtype=torch.int32).repeat_interleave(pos_num, dim=0).unsqueeze(0)
    sim_drug_index_col = torch.zeros_like(sim_drug_index_row)
    sim_drug_index = torch.cat((sim_drug_index_row, sim_drug_index_col), dim=0)
    sim_drug_index[1, :] = indices.view(-1)
    pos = torch.zeros_like(sim_net)
    pos[sim_drug_index[0, :],sim_drug_index[1, :]] = 1
    pos = pos.to(device)

    # 获得feature sim中的正样本
    if not trained_sim is None:
        vals_fea, indices_fea = trained_sim.topk(k=pos_num, dim=1, largest=True, sorted=True)
        sim_drug_index_row_fea = torch.arange(len(trained_sim), dtype=torch.int32).repeat_interleave(pos_num, dim=0).unsqueeze(0)
        sim_drug_index_col_fea = torch.zeros_like(sim_drug_index_row_fea)
        sim_drug_index_fea = torch.cat((sim_drug_index_row_fea, sim_drug_index_col_fea), dim=0)
        sim_drug_index_fea[1, :] = indices_fea.view(-1)
        pos_fea = torch.zeros_like(trained_sim)
        pos_fea[sim_drug_index_fea[0, :], sim_drug_index_fea[1, :]] = 1
        pos_fea = pos_fea.to(device)

        # 取交集
        pos = pos + pos_fea
        pos = torch.where(pos == 2,1,0)



    # 获取负样本
    neg_vals, neg_indices = (-sim_net).topk(k=neg_num, dim=1, largest=True, sorted=True)
    neg_sim_drug_index_row = torch.arange(len(sim_net), dtype=torch.int32).repeat_interleave(neg_num, dim=0).unsqueeze(0)
    neg_sim_drug_index_col = torch.zeros_like(neg_sim_drug_index_row)
    neg_sim_drug_index = torch.cat((neg_sim_drug_index_row, neg_sim_drug_index_col), dim=0)
    neg_sim_drug_index[1, :] = neg_indices.view(-1)
    neg = torch.zeros_like(sim_net)
    neg[neg_sim_drug_index[0, :], neg_sim_drug_index[1, :]] = 1
    neg = neg.to(device)

    if not trained_sim is None:
        # 获取feature sim的负样本
        neg_vals_fea, neg_indices_fea = (-trained_sim).topk(k=neg_num, dim=1, largest=True, sorted=True)
        neg_sim_drug_index_row_fea = torch.arange(len(trained_sim), dtype=torch.int32).repeat_interleave(neg_num, dim=0).unsqueeze(0)
        neg_sim_drug_index_col_fea = torch.zeros_like(neg_sim_drug_index_row_fea)
        neg_sim_drug_index_fea = torch.cat((neg_sim_drug_index_row_fea, neg_sim_drug_index_col_fea), dim=0)
        neg_sim_drug_index_fea[1, :] = neg_indices_fea.view(-1)
        neg_fea = torch.zeros_like(trained_sim)
        neg_fea[neg_sim_drug_index_fea[0, :], neg_sim_drug_index_fea[1, :]] = 1
        neg_fea = neg_fea.to(device)

        # 取并集
        # neg = neg + neg_fea
        # neg = torch.where(neg >= 1,1,0)
        # 取交集
        neg = neg + neg_fea
        neg = torch.where(neg == 2, 1, 0)

    # 节点自身为pos pair
    eye = torch.arange(len(sim_net), dtype=torch.int32)
    pos[eye,eye] = 1
    # 去除负样本pair中存在的pos pair
    neg = neg - pos
    neg = torch.where(neg >0,1,0)



    """
    以threshold为界限,大于threshold的为pos,小于threshold的为neg,在train过程中,阈值逐渐向0.5靠拢
    """
    # beta1 = max - threshold
    # beta2 = threshold - min
    # gama = 1
    # # update threshold
    # min = min + (current_epoch/total_epoch)**gama * beta2
    # max = max - (current_epoch/total_epoch)**gama * beta1
    #
    # # init pos, neg pair set
    # pos =torch.eye(sim_net[4].shape[0], sim_net[4].shape[0]).to(device)
    # neg = torch.zeros(sim_net[4].shape[0], sim_net[4].shape[0]).to(device)
    #
    # if trained_sim is not None:
    #     temp_pos = torch.where(trained_sim >= max, 1, 0) + torch.where(sim_net >= max, 1, 0)
    #     temp_neg = torch.where(trained_sim < min, 1,0) + torch.where(sim_net < min, 1,0)
    #     pos_add = torch.where(temp_pos == 2)
    #     neg_add = torch.where(temp_neg == 2)
    # else:
    #     pos_add = torch.where(sim_net >= max)
    #     neg_add = torch.where(sim_net < min)
    # if len(pos_add[0]) >0:
    #     pos[pos_add[0], pos_add[1]] = 1
    # if len(neg_add[0]) > 0:
    #     neg[neg_add[0], neg_add[1]] = 1


    """
    按照比例选择pos neg样本
    初始选择top 1%
     每个node pos pair 最终选择总结点数的1%
     每个node neg pair 选择总结点数的50%
    """
    # pos =torch.eye(sim_net[4].shape[0], sim_net[4].shape[0]).to(device)
    # neg = torch.zeros(sim_net[4].shape[0], sim_net[4].shape[0]).to(device)
    #
    # max_pos_num = int(0.01 * len(sim_net))
    # ori_rate = 0.01
    # pos_num = 1
    # # neg_num = int((ori_rate + 0.49 * current_epoch / total_epoch)*len(sim_net))
    # # pos_num = pos_num + int(max_pos_num * current_epoch / total_epoch)
    #
    # neg_num = int(0.5 * len(sim_net))
    # pos_num = max_pos_num
    #
    # _,pos_indices = torch.sort(sim_net,descending=True)
    # pos_indices = pos_indices[:,:pos_num]
    # pos_index = torch.where(pos_indices > -1)
    # pos[pos_index[0], pos_indices.reshape(-1)] = 1
    #
    #
    # _,neg_indices = torch.sort(sim_net,descending=False)
    # neg_indices = neg_indices[:,:neg_num]
    # neg_index = torch.where(neg_indices > -1)
    # neg[neg_index[0], neg_indices.reshape(-1)] = 1


    return pos, neg

def get_hop_neighbor(M, C, mp_neighbor,sp_KG,max_mp_num,curr_mp_num,max_C):
    # m = i + 1
    mp_neighbor = mp_neighbor.dot(sp_KG.transpose()).tocoo()
    hop_row = mp_neighbor.row
    hop_col = mp_neighbor.col
    hop_data = mp_neighbor.data
    hop_data = np.where(hop_data > max_C,max_C,hop_data)
    mp_neighbor = sp.coo_matrix((hop_data,(hop_row,hop_col)),shape=mp_neighbor.shape)
    if max_mp_num > 1:
        M,C = get_hop_neighbor(M, C, mp_neighbor,sp_KG,max_mp_num-1,curr_mp_num+1,max_C)
    M[hop_row, hop_col] = curr_mp_num
    C[hop_row, hop_col] = torch.tensor(hop_data)
    return M,C


def cul_sim(M, C,max_num):
    # calculate the association strength based on meta path
    """
    简单点儿的，不考虑系数C
    y = 1/e^x
    对于每一个pair，距离x为：3(M-1+1/C-1/max_c+bias),max_c为该阶数中的C的最大值
    """
    bias = 1e-7
    sim = torch.zeros_like(M,dtype=torch.float32)
    for i in range(max_num):
        mask = torch.zeros_like(M)
        hop_num = i+1
        index = torch.where(M == hop_num)
        mask[index[0],index[1]] = 1
        M_hop = torch.mul(M, mask)
        C_hop = torch.mul(C,mask)
        max_c = torch.max(C_hop)
        X_hop = 3*(M_hop - 1) + 3*(1/C_hop - 1/max_c) + bias
        sim_hop = torch.exp(-X_hop)
        sim[index[0],index[1]] = sim_hop[index[0],index[1]]

    return sim


def get_sim_mp(ori_KG,node_num,device):

    # 设置系数C的最大值
    max_C = 1000
    ori_KG = ori_KG.to('cpu')
    M = -torch.ones((node_num,node_num),dtype=torch.int32)
    C = -torch.ones((node_num,node_num),dtype=torch.int32)
    nei_num = 0
    max_mp_num = 5 # 最大meta path阶数
    # M表示同质邻居阶数，C表示关联系数
    ori_row = np.array(ori_KG[:,0])
    ori_col = np.array(ori_KG[:,1])
    ori_data = np.array(ori_KG[:,3])
    # 获得逆关系
    row = np.concatenate((ori_row,ori_col))
    col = np.concatenate((ori_col,ori_row))
    data = np.concatenate((ori_data,ori_data))
    sp_KG = sp.coo_matrix((data,(row,col)),shape=(node_num,node_num))

    mp_neighbor = sp_KG
    collected_neighbor = torch.zeros_like(M)
    curr_mp_num = 1
    if not os.path.exists("data/MetaPath/metapath_M.pt"):
        M,C = get_hop_neighbor(M,C,mp_neighbor,sp_KG,max_mp_num,curr_mp_num,max_C)
    else:
        M = torch.load("data/MetaPath/metapath_M.pt")
        C = torch.load("data/MetaPath/metapath_C.pt")

    sim = cul_sim(M,C,max_mp_num)

    return sim





