"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
from numpy.lib.function_base import append
from model import BaseRGCN
import json
from sklearn.metrics import f1_score
from scipy import sparse
from dgl.nn.pytorch import RelGraphConv
import dgl
import torch.nn.functional as F
import torch
import time
import numpy as np
from os import link
import argparse
import torch.nn as nn
import pynvml
import os
import gc
import psutil
from utils.util import training_scheduler, sort_training_nodes, sort_training_nodes_2,sort_training_nodes_3, setup_seed, get_noisy_data, get_noisy_multilabel_data,sort_mccl,sort_mccl_2,sort_mccl_3
from torch_geometric.data import Data
import wandb
import sys
import random
sys.path.append('../../')
pynvml.nvmlInit()

import math
mccl = False

def training_scheduler(lam, t, T, scheduler='linear'):
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))
    
def seed_torch(seed):
    # seed = int(seed)
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True
    
def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def evaluate(model_pred, labels):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    pred_result = model_pred.argmax(dim=1)
    labels = labels.cpu()
    pred_result = pred_result.cpu()

    micro = f1_score(labels, pred_result, average='micro')
    macro = f1_score(labels, pred_result, average='macro')

    return micro, macro


def multi_evaluate(model_pred, labels):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    model_pred = torch.sigmoid(model_pred)
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    labels = labels.cpu()
    pred_result = pred_result.cpu()

    micro = f1_score(labels, pred_result, average='micro')
    macro = f1_score(labels, pred_result, average='macro')

    return micro, macro


class EntityClassify(BaseRGCN):
    def build_input_layer(self):
        return nn.ModuleList([nn.Linear(in_dim, self.h_dim, bias=True) for in_dim in self.in_dims])

    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                            dropout=self.dropout)

    def build_output_layer(self):
        return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis",
                            self.num_bases, activation=None,
                            self_loop=self.use_self_loop)


def main(args,lamb,big_t,i,noise_rate):
    seed_torch(int(123+i))
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    if args.gpu >= 0:
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_begin = meminfo.used
    dataset = ['dblp', 'imdb', 'acm', 'freebase']
    if args.dataset in dataset:
        dataset = None
    else:
        raise ValueError()

    # Load from hetero-graph
    if args.dataset in ['imdb']:
        LOSS = F.binary_cross_entropy_with_logits
    else:
        LOSS = F.cross_entropy

    folder = '/HGB/NC/benchmark/methods/data/'+args.dataset.upper()
    from scripts.data_loader import data_loader
    dl = data_loader(folder)

    all_data = {}
    for etype in dl.links['meta']:
        etype_info = dl.links['meta'][etype]
        metrix = dl.links['data'][etype]
        all_data[(etype_info[0], 'link', etype_info[1])] = (
            sparse.find(metrix)[0]-dl.nodes['shift'][etype_info[0]], sparse.find(metrix)[1]-dl.nodes['shift'][etype_info[1]])
    hg = dgl.heterograph(all_data)
    category_id = list(dl.labels_train['count'].keys())[0]
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    if args.dataset == 'imdb':
        labels = torch.FloatTensor(
            dl.labels_train['data']+dl.labels_test['data'])
    else:
        labels = torch.LongTensor(
            dl.labels_train['data']+dl.labels_test['data']).argmax(dim=1)
    num_classes = dl.labels_test['num_classes']

    num_rels = len(hg.canonical_etypes)
    if args.dataset in ['imdb']:
        EVALUATE = multi_evaluate
    else:
        EVALUATE = evaluate

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx
    # calculate norm for each edge type and store in edge
    for canonical_etype in hg.canonical_etypes:
        u, v, eid = hg.all_edges(form='all', etype=canonical_etype)
        _, inverse_index, count = torch.unique(
            v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = torch.ones(eid.shape[0]).float() / degrees.float()
        norm = norm.unsqueeze(1)
        hg.edges[canonical_etype].data['norm'] = norm

    g = dgl.to_homogeneous(hg, edata=['norm'])
    num_nodes = g.number_of_nodes()
    node_ids = torch.arange(num_nodes)
    edge_norm = g.edata['norm']
    edge_type = g.edata[dgl.ETYPE].long()

    # find out the target node ids in g
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda:'+str(args.gpu) if use_cuda else 'cpu')
    torch.cuda.set_device(args.gpu)
    edge_type = edge_type.to(device)
    edge_norm = edge_norm.to(device)
    labels = labels.to(device)
    features_list = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features_list.append(np.eye(dl.nodes['count'][i]))
        else:
            features_list.append(th)
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    feats_type = args.feats_type
    in_dims = []
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)

    model = EntityClassify(in_dims,
                           args.n_hidden,
                           num_classes,
                           num_rels,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop)

    model.to(device)
    g = g.to('cuda:%d' % args.gpu)

    adjM = sum(dl.links['data'].values())
    edge_index = torch.Tensor(adjM.toarray()).nonzero().t().contiguous()
    data = Data(edge_index= edge_index, y = labels)
    if args.dataset == 'dblp':
        data.num_nodes = 26128
    elif args.dataset =='imdb':
        data.num_nodes = 21420
    else: 
        data.num_nodes = 10942
    data.train_mask = torch.tensor(train_idx)
    data.val_mask = torch.tensor(val_idx)
    data.test_mask = test_idx
    data.num_classes = num_classes
    data.num_edge_type = 7
    ## noising label 
    if args.dataset =='imdb':
        data = get_noisy_multilabel_data(data,noise_rate*0.05)
    else:
        data = get_noisy_data(data,noise_rate*0.05)
    labels = data.y
    data.train_mask = train_idx
    data.val_mask = val_idx

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    forward_time = []
    backward_time = []
    save_dict_micro = {}
    save_dict_macro = {}
    best_result_micro = 0
    best_result_macro = 0
    best_epoch_micro = 0
    best_epoch_macro = 0
    
    for train_round in range(1):
        model.train()
        label = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
        label[train_idx] = dl.labels_train['data'][train_idx]
        label[val_idx] = dl.labels_train['data'][val_idx]
        if args.dataset != 'imdb':
            label = label.argmax(axis=1)
        label = torch.tensor(label).to(device)
        if not mccl:
            if args.dm >= 1:
                if args.dm == 1: # Topo
                    alpha = [0,1,1,1,0]
                elif args.dm == 2: # Feat
                    alpha = [1,0,0,0,0]
                elif args.dm == 3: # Loss
                    alpha = [0,0,0,0,1]
                elif args.dm == 4: # CLNode
                    alpha = [1,0.33,0.33,0.33,0]
                elif args.dm == 5: # AvgDiff
                    alpha = [1,1,1,1,1]
                elif args.dm == 6: # Ours
                    alpha = [0,0,0,0,0]
                elif args.dm == 7: # Topo 1
                    alpha = [0,1,0,0,0]
                elif args.dm == 8: # Topo 2
                    alpha = [0,0,1,0,0]
                elif args.dm == 9: # Topo 3
                    alpha = [0,0,0,1,0]
                embedding = torch.load('/MDCL/RGCN/embeding.pt')
                embedding = F.normalize(embedding)
                sorted_trainset = sort_training_nodes(data, label, embedding, alpha,loss_alpha=args.loss_alpha)
                if not torch.is_tensor(train_idx):
                    train_idx = torch.tensor(train_idx)
                mask = torch.isin(sorted_trainset, train_idx.to(device))
                sorted_trainset = sorted_trainset[mask]
        for epoch in range(args.n_epochs):
            
            if args.dm >= 1:
                size = training_scheduler(lamb, epoch, big_t, 'linear')
                if mccl:
                    embedding = torch.load('/MDCL/RGCN/embeding.pt')
                    embedding = F.normalize(embedding)
                    sorted_trainset = sort_mccl(data, label, embedding, size)
                    if not torch.is_tensor(train_idx):
                        train_idx = torch.tensor(train_idx)
                    mask = torch.isin(sorted_trainset, train_idx.to(device))
                    sorted_trainset = sorted_trainset[mask]
                    train_mask = sorted_trainset[:int(size*sorted_trainset.shape[0])]
                else:
                    train_mask = sorted_trainset[:int(size * sorted_trainset.shape[0])]
            optimizer.zero_grad()
            t0 = time.time()
            logits = model(g, features_list, edge_type, edge_norm)
            logits = logits[target_idx]
            if args.dm == 0:    
                if args.dataset in ['imdb']:            
                    loss = LOSS(logits[train_idx], labels[train_idx],reduction= 'none').mean(dim=1)
                else:
                    loss = LOSS(logits[train_idx], labels[train_idx],reduction= 'none')
                if epoch == 0:
                    averaged_loss = loss
                else:
                    averaged_loss += loss
                if epoch == (args.n_epochs-1):
                    averaged_loss /= args.n_epochs
                    torch.save(averaged_loss,'/MDCL/RGCN/averaged_loss.pt')
                loss = loss.mean()
            else:
                loss = LOSS(logits[train_mask], labels[train_mask])
            
            t1 = time.time()
            loss.backward()
            optimizer.step()
            t2 = time.time()

            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)
            # print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
            #       format(epoch, forward_time[-1], backward_time[-1]))
            # val_loss = LOSS(logits[val_idx], labels[val_idx])
            # train_micro, train_macro = EVALUATE(
            #     logits[train_idx], labels[train_idx])
            valid_micro, valid_macro = EVALUATE(
                logits[val_idx], labels[val_idx])
            if valid_micro > best_result_micro:
                save_dict_micro = model.state_dict()
                best_result_micro = valid_micro
                best_epoch_micro = epoch
            if valid_macro > best_result_macro:
                save_dict_macro = model.state_dict()
                best_result_macro = valid_macro
                best_epoch_macro = epoch
            test_micro, test_macro = EVALUATE(
                    logits[test_idx], labels[test_idx])

            # print("Train micro: {:.4f} | Train macro: {:.4f} | Train Loss: {:.4f} | Validation micro: {:.4f} | Validation macro: {:.4f} | Validation loss: {:.4f}".
            #     format(train_micro, train_macro, loss.item(), valid_micro, valid_macro, val_loss.item()))

        model.eval()
        result = [save_dict_micro, save_dict_macro]
        torch.cuda.empty_cache()
        with torch.no_grad():
            for i in range(1):
                model.load_state_dict(result[i])
                t0 = time.time()
                logits = model.forward(g, features_list, edge_type, edge_norm)
                t1 = time.time()
                # print("test time:"+str(t1-t0))
                logits = logits[target_idx]
                test_loss = LOSS(logits[test_idx], labels[test_idx])
                test_micro, test_macro = EVALUATE(
                    logits[test_idx], labels[test_idx])
                # print("Test micro: {:.4f} | Test macro: {:.4f} | Test loss: {:.4f}".format(
                #     test_micro, test_macro, test_loss.item()))
                print(test_macro,test_micro)
            # print("Mean forward time: {:4f}".format(
            #     np.mean(forward_time[len(forward_time) // 4:])))
            # print("Mean backward time: {:4f}".format(
            #     np.mean(backward_time[len(backward_time) // 4:])))
            if args.dm == 0:
                logits = model.forward(g, features_list, edge_type, edge_norm)
                embedding = logits
                torch.save(embedding,'/MDCL/RGCN/embeding.pt')
                label = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
                label[train_idx] = dl.labels_train['data'][train_idx]
                label[val_idx] = dl.labels_train['data'][val_idx]
                if args.dataset != 'imdb':
                    label = label.argmax(axis=1)
                label = torch.tensor(label).to(device)                
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_end = meminfo.used


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument('--feats-type', type=int, default=2,
                        help='Type of the node features used. ' +
                        '0 - loaded features; ' +
                        '1 - only target node features (zero vec for others); IMDB' +
                        '2 - only target node features (id vec for others); ACM ' +
                        '3 - all id vec. Default is 2; DBLP ' +
                        '4 - only term features (id vec for others);' +
                        '5 - only term features (zero vec for others).')
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
                        help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=150,
                        help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
                        help="l2 norm coef")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
                        help="include self feature as a special relation")
    parser.add_argument('--dm', type=int,default=0)
    parser.add_argument('--loss_alpha', type=float,default=0)
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()

    print('linear')
    for k in range(7):
        for i in range(3):
            print('seed:',i)
            if os.path.exists('/MDCL/RGCN/averaged_loss.pt'):
                os.remove('/MDCL/RGCN/averaged_loss.pt')
                os.remove('/MDCL/RGCN/diff.pt')
                os.remove('/MDCL/RGCN/embeding.pt')
            args.dm = 0
            seed_torch(int(123+i))
            main(args,1,200,i,k)
            for j in range(1,7):
                args.dm = j
                seed_torch(int(123+i))
                main(args,2/10,75,i,k)
