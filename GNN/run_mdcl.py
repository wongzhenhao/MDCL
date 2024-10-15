import sys
sys.path.append('../../')
import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_data
#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from GNN import GCN, GAT
import dgl
import warnings

import math
import os
import random
from utils.util import training_scheduler, sort_training_nodes, sort_training_nodes_2,sort_training_nodes_3, setup_seed, get_noisy_data,sort_mccl_2,sort_mccl_3
from torch_geometric.data import Data
import wandb
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
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

def run_model_DBLP(args,lamb,big_t,i,noise_rate):
    seed_torch(int(123+i))
    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset)
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = [] # [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
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
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)

    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)
    
    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    num_classes = dl.labels_train['num_classes']
    edge_index = torch.Tensor(adjM.toarray()).nonzero().t().contiguous()
    data = Data(edge_index= edge_index, y = labels)
    if args.dataset == 'DBLP':
        data.num_nodes = 26128
    else:
        data.num_nodes = 10942
    data.train_mask = torch.tensor(train_idx)
    data.val_mask = torch.tensor(val_idx)
    data.test_mask = test_idx
    data.num_classes = num_classes
    data.num_edge_type = 7
    
    ## noising label 
    data = get_noisy_data(data,noise_rate*0.05)
    labels = data.y
    data.train_mask = train_idx
    data.val_mask = val_idx
    for train_round in range(1):
        num_classes = dl.labels_train['num_classes']
        if args.model_type == 'gat':
            heads = [args.num_heads] * args.num_layers + [1]
            net = GAT(g, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, False)
        elif args.model_type == 'gcn':
            net = GCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
        else:
            raise Exception('{} model is not defined!'.format(args.model_type))
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.dm >=1:
            net.load_state_dict(torch.load('/MDCL/GNN/net.pth'))
            optimizer.load_state_dict(torch.load('/MDCL/GNN/optim.pth'))
        else:
            torch.save(net.state_dict(),'/MDCL/GNN/net.pth')   
            torch.save(optimizer.state_dict(),'/MDCL/GNN/optim.pth')
        # training loop
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='/MDCL/GNN/checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.model_type))
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
                embedding = torch.load('/MDCL/GNN/embeding.pt')
                sorted_trainset = sort_training_nodes_2(data, labels, embedding, alpha, loss_alpha=args.loss_alpha)
                train_idx = torch.tensor(train_idx)
                mask = torch.isin(sorted_trainset, train_idx.to(device))
                sorted_trainset = sorted_trainset[mask]
        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            size = training_scheduler(lamb, epoch, big_t, 'linear')
            if args.dm >= 1:
                if mccl:
                    embedding = torch.load('/MDCL/GNN/embeding.pt')
                    sorted_trainset = sort_mccl_3(data, labels, embedding, size)
                    train_idx = torch.tensor(train_idx)
                    mask = torch.isin(sorted_trainset, train_idx.to(device))
                    sorted_trainset = sorted_trainset[mask]
                    train_mask = sorted_trainset[:int(size*sorted_trainset.shape[0])]
                else:
                    train_mask = sorted_trainset[:int(size * sorted_trainset.shape[0])]
            net.train()

            logits = net(features_list)
            logp = F.log_softmax(logits, 1)
            if args.dm == 0:                
                train_loss = F.nll_loss(logp[train_idx], labels[train_idx], reduction= 'none')
                if epoch == 0:
                    averaged_loss = train_loss
                else:
                    averaged_loss += train_loss
                if epoch == (args.epoch-1):
                    averaged_loss /= args.epoch
                    torch.save(averaged_loss,'/MDCL/GNN/averaged_loss.pt')
                train_loss = train_loss.mean()
            else:
                train_loss = F.nll_loss(logp[train_mask], labels[train_mask])
            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t_end = time.time()

            # print training info
            # print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, train_loss.item(), t_end-t_start))

            t_start = time.time()
            # validation
            net.eval()
            with torch.no_grad():
                logits = net(features_list)
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
                logits = net(features_list)
                test_logits = logits[test_idx]
                pred = test_logits.cpu().numpy().argmax(axis=1)
                onehot = np.eye(num_classes, dtype=np.int32)
                pred = onehot[pred]
                test = dl.evaluate(pred)
            t_end = time.time()
            # print validation info
            # print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                # epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                # print('Early stopping!')
                break
        # testing with evaluate_results_nc
        net.load_state_dict(torch.load('/MDCL/GNN/checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.model_type)))
        net.eval()
        test_logits = []
        with torch.no_grad():
            logits = net(features_list)
            test_logits = logits[test_idx]
            pred = test_logits.cpu().numpy().argmax(axis=1)
            onehot = np.eye(num_classes, dtype=np.int32)
            dl.gen_file_for_evaluate(test_idx=test_idx,label=pred,file_name='/MDCL/GNN/eval.txt')
            pred = onehot[pred]
            print(dl.evaluate(pred))
            if args.dm == 0:
                embedding = logits
                torch.save(embedding,'/MDCL/GNN/embeding.pt')
                label = data.y[data.train_mask]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=2,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=300, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--model-type', type=str, help="gcn or gat")
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--dataset', type=str, default = 'ACM')
    ap.add_argument('--dm', type=int,default=0)
    ap.add_argument('--loss_alpha', type=float,default=0)

    warnings.filterwarnings('ignore')
    
    
    print('linear')
    for k in range(7):
        for i in range(3):
            print('seed:',i)
            if os.path.exists('/MDCL/GNN/averaged_loss.pt'):
                os.remove('/MDCL/GNN/averaged_loss.pt')
                os.remove('/MDCL/GNN/diff_gcn.pt')
                os.remove('/MDCL/GNN/embeding.pt')
            args = ap.parse_args()
            args.dm = 0        
            seed_torch(int(123+i))
            run_model_DBLP(args,1,200,i,k)
            for j in range(1,7):
                args.dm = j
                seed_torch(int(123+i))
                run_model_DBLP(args,9/10,75,i,k)

