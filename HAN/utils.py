import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
import copy
import torch as th
import scipy.sparse as sp
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
import torch.nn as nn

import sys

sys.path.append('../../')
from scripts.data_loader import data_loader

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

def spearman_correlation(tensor1, tensor2):
    rank1 = torch.argsort(torch.argsort(tensor1))
    rank2 = torch.argsort(torch.argsort(tensor2))
    
    diff = rank1 - rank2
    diff_squared = diff ** 2
    
    n = tensor1.shape[0]
    rho = 1 - (6 * torch.sum(diff_squared).float()) / (n * (n ** 2 - 1))
    
    return rho

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            #print('Created directory {}'.format(path))
            temp = True
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix


def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir


# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,  # Learning rate
    'num_heads': [8],  # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
}

sampling_configure = {
    'batch_size': 20
}


def setup(args,i):
    args.update(default_configure)
    set_random_seed(123+i)
    args['log_dir'] = setup_log_dir(args)
    return args


def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args['device'] = 'cuda:5' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)
    return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def load_acm(feat_type=0):
    dl = data_loader('/HGB/NC/benchmark/methods/data/ACM')
    link_type_dic = {0: 'pp', 1: '-pp', 2: 'pa', 3: 'ap', 4: 'ps', 5: 'sp', 6: 'pt', 7: 'tp'}
    paper_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)
    adjM = sum(dl.links['data'].values())
    features = []
    for i in range(len(dl.nodes['count'])):
        the = dl.nodes['attr'][i]
        if the is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(the)
    in_dims = [feature.shape[1] for feature in features]
    features = [mat2tensor(feature).to(device) for feature in features]
    features[0] = th.FloatTensor(dl.nodes['attr'][0]).to(device)
    fc_list = nn.ModuleList([nn.Linear(in_dim, 1902, bias=True) for in_dim in in_dims[1:]]).to(device)
    for fc in fc_list:
        nn.init.xavier_normal_(fc.weight, gain=1.414)
    h = []
    h.append(features[0])
    for fc, feature in zip(fc_list, features[1:]):
        h.append(fc(feature).detach())
    labels = dl.labels_test['data'][:paper_num] + dl.labels_train['data'][:paper_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = 3

    train_valid_mask = dl.labels_train['mask'][:paper_num]
    test_mask = dl.labels_test['mask'][:paper_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['pp', 'ps', 'sp'], ['-pp', 'ps', 'sp'], ['pa', 'ap'], ['ps', 'sp'], ['pt', 'tp']]
    features = h.copy()
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths,adjM



def load_dblp(feat_type=0):
    prefix = '/HGB/NC/benchmark/methods/data/DBLP'
    dl = data_loader('/HGB/NC/benchmark/methods/data/DBLP')
    link_type_dic = {0: 'ap', 1: 'pc', 2: 'pt', 3: 'pa', 4: 'cp', 5: 'tp'}
    author_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)
    adjM = sum(dl.links['data'].values())
    features = []
    for i in range(len(dl.nodes['count'])):
        the = dl.nodes['attr'][i]
        if the is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(the)
    in_dims = [feature.shape[1] for feature in features]
    features = [mat2tensor(feature).to(device) for feature in features]
    features[0] = th.FloatTensor(dl.nodes['attr'][0]).to(device)
    fc_list = nn.ModuleList([nn.Linear(in_dim, 334, bias=True) for in_dim in in_dims[1:]]).to(device)
    for fc in fc_list:
        nn.init.xavier_normal_(fc.weight, gain=1.414)
    h = []
    h.append(features[0])
    for fc, feature in zip(fc_list, features[1:]):
        h.append(fc(feature).detach())
    # author labels
    labels = dl.labels_test['data'][:author_num] + dl.labels_train['data'][:author_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = 4

    train_valid_mask = dl.labels_train['mask'][:author_num]
    test_mask = dl.labels_test['mask'][:author_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]
    meta_paths = [['ap', 'pa'], ['ap', 'pt', 'tp', 'pa'], ['ap', 'pc', 'cp', 'pa']]
    meta_paths_1 = [['pa', 'ap'], ['pc', 'cp'], ['pt', 'tp']]
    meta_path_2 = [['cp', 'pc']]
    meta_path_3 = [['tp', 'pt']]
    features = h.copy()
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths, adjM


def load_imdb(feat_type=0):
    prefix = '/HGB/NC/benchmark/methods/data/IMDB'
    dl = data_loader(prefix)
    link_type_dic = {0: 'md', 1: 'dm', 2: 'ma', 3: 'am', 4: 'mk', 5: 'km'}
    movie_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)
    adjM = sum(dl.links['data'].values())
    features = []
    for i in range(len(dl.nodes['count'])):
        the = dl.nodes['attr'][i]
        if the is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(the)
    in_dims = [feature.shape[1] for feature in features]
    features = [mat2tensor(feature).to(device) for feature in features]
    features[0] = th.FloatTensor(dl.nodes['attr'][0]).to(device)
    fc_list = nn.ModuleList([nn.Linear(in_dim, in_dims[0], bias=True) for in_dim in in_dims]).to(device)
    for fc in fc_list:
        nn.init.xavier_normal_(fc.weight, gain=1.414)
    h = []
    # h.append(features[0])
    for fc, feature in zip(fc_list, features):
        h.append(fc(feature).detach())
    labels = dl.labels_test['data'][:movie_num] + dl.labels_train['data'][:movie_num]
    labels = th.FloatTensor(labels)

    num_classes = 5

    train_valid_mask = dl.labels_train['mask'][:movie_num]
    test_mask = dl.labels_test['mask'][:movie_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['md', 'dm'], ['ma', 'am'], ['mk', 'km']]
    features = h.copy()
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths, dl,adjM


def load_data(dataset, feat_type=0):
    load_fun = None
    if dataset == 'ACM':
        load_fun = load_acm
    elif dataset == 'DBLP':
        load_fun = load_dblp
    elif dataset == 'IMDB':
        load_fun = load_imdb
    return load_fun(feat_type=feat_type)


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = '/MDCL/HAN/early_stop/early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc <= self.best_acc):
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
                
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))

import copy
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from collections import Counter, defaultdict

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


device = 'cuda:7'

def neighborhood_difficulty_measurer(data, label,meta):
    edge_index = data.edge_index
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    num_nodes = torch.max(edge_index) + 1
    adj_list = [[]for _ in range(num_nodes)]
    for src, tgt in zip(source_nodes, target_nodes):
        adj_list[src].append(tgt)
        adj_list[tgt].append(src)
    two_hop_neighbours = [[]for _ in range(int(label.size()[0]))]
    
    # if meta == 1: # 010
    #     for node in range(int(label.size()[0])):
    #         if int(node) in data.train_mask:
    #             one_hop_neighbors = adj_list[node]
    #             for neighbour in one_hop_neighbors:
    #                 if neighbour != node and neighbour>= 4932 and neighbour <= 7324: # select which metapath 
    #                     for second_hop in adj_list[neighbour]:
    #                         if second_hop != node and int(second_hop) in data.train_mask:
    #                             two_hop_neighbours[node].append(second_hop)
    # elif meta ==2: # 020
    #     for node in range(int(label.size()[0])):
    #         if int(node) in data.train_mask:
    #             one_hop_neighbors = adj_list[node]
    #             for neighbour in one_hop_neighbors:
    #                 if neighbour != node and neighbour>= 7325 and neighbour <= 13448: # select which metapath 
    #                     for second_hop in adj_list[neighbour]:
    #                         if second_hop != node and int(second_hop) in data.train_mask:
    #                             two_hop_neighbours[node].append(second_hop)
    # elif meta ==3: # 030
    #     for node in range(int(label.size()[0])):
    #         if int(node) in data.train_mask:
    #             one_hop_neighbors = adj_list[node]
    #             for neighbour in one_hop_neighbors:
    #                 if neighbour != node and neighbour>= 13449 and neighbour <= 24119: # select which metapath 
    #                     for second_hop in adj_list[neighbour]:
    #                         if second_hop != node and int(second_hop) in data.train_mask:
    #                             two_hop_neighbours[node].append(second_hop)

    # for i in range(len(two_hop_neighbours)):
    #     if two_hop_neighbours[i]:
    #         two_hop_neighbours[i] = torch.stack(two_hop_neighbours[i])
    #         two_hop_neighbours[i] = torch.unique(two_hop_neighbours[i])

    # You could use the pre-compute neighbour or those commented code to load/conmpute the neighbour
    if meta == 1: # 010
        two_hop_neighbours = torch.load('/MDCL/GNN/2-hop_IMDB_010.pt')
    elif meta == 2: # 020
        two_hop_neighbours = torch.load('/MDCL/GNN/2-hop_IMDB_020.pt')
    elif meta == 3: # 030
        two_hop_neighbours = torch.load('/MDCL/GNN/2-hop_IMDB_030.pt')

    two_hop_label = [[]for _ in range(int(label.size()[0]))]
    two_hop_label_0 = [[]for _ in range(int(label.size()[0]))]
    two_hop_label_1 = [[]for _ in range(int(label.size()[0]))]
    two_hop_label_2 = [[]for _ in range(int(label.size()[0]))]
    two_hop_label_3 = [[]for _ in range(int(label.size()[0]))]
    two_hop_label_4 = [[]for _ in range(int(label.size()[0]))]
    for i in range(int(label.size()[0])):
        two_hop_label[i]=label[two_hop_neighbours[i]]
        for j in range(len(two_hop_label[i])):
            two_hop_label_0[i].append(two_hop_label[i][j][0])
            two_hop_label_1[i].append(two_hop_label[i][j][1])
            two_hop_label_2[i].append(two_hop_label[i][j][2])
            two_hop_label_3[i].append(two_hop_label[i][j][3])
            two_hop_label_4[i].append(two_hop_label[i][j][4])
        two_hop_label_0[i].append(label[i][0])
        two_hop_label_1[i].append(label[i][1])
        two_hop_label_2[i].append(label[i][2])
        two_hop_label_3[i].append(label[i][3])
        two_hop_label_4[i].append(label[i][4])
        two_hop_label_0[i] = torch.stack(two_hop_label_0[i])
        two_hop_label_1[i] = torch.stack(two_hop_label_1[i])
        two_hop_label_2[i] = torch.stack(two_hop_label_2[i])
        two_hop_label_3[i] = torch.stack(two_hop_label_3[i])
        two_hop_label_4[i] = torch.stack(two_hop_label_4[i])

    element_counts_0 = [{} for _ in range(int(label.size()[0]))]
    element_counts_1 = [{} for _ in range(int(label.size()[0]))]
    element_counts_2 = [{} for _ in range(int(label.size()[0]))]
    element_counts_3 = [{} for _ in range(int(label.size()[0]))]
    element_counts_4 = [{} for _ in range(int(label.size()[0]))]
    for i in range(int(label.size()[0])):
        unique_elements, counts = torch.unique(two_hop_label_0[i], return_counts=True)
        element_counts_0[i] = {elem.item(): count.item() for elem, count in zip(unique_elements, counts)}
        unique_elements, counts = torch.unique(two_hop_label_1[i], return_counts=True)
        element_counts_1[i] = {elem.item(): count.item() for elem, count in zip(unique_elements, counts)}
        unique_elements, counts = torch.unique(two_hop_label_2[i], return_counts=True)
        element_counts_2[i] = {elem.item(): count.item() for elem, count in zip(unique_elements, counts)}
        unique_elements, counts = torch.unique(two_hop_label_3[i], return_counts=True)
        element_counts_3[i] = {elem.item(): count.item() for elem, count in zip(unique_elements, counts)}
        unique_elements, counts = torch.unique(two_hop_label_4[i], return_counts=True)
        element_counts_4[i] = {elem.item(): count.item() for elem, count in zip(unique_elements, counts)}

  
    neighbour_class_0 = [[]for _ in range(int(label.size()[0]))]
    neighbour_class_1 = [[]for _ in range(int(label.size()[0]))]
    neighbour_class_2 = [[]for _ in range(int(label.size()[0]))]
    neighbour_class_3 = [[]for _ in range(int(label.size()[0]))]
    neighbour_class_4 = [[]for _ in range(int(label.size()[0]))]
    neighbor_entropy_0 = [torch.tensor(1) for _ in range(int(label.size()[0]))]
    neighbor_entropy_1 = [torch.tensor(1) for _ in range(int(label.size()[0]))]
    neighbor_entropy_2 = [torch.tensor(1) for _ in range(int(label.size()[0]))]
    neighbor_entropy_3 = [torch.tensor(1) for _ in range(int(label.size()[0]))]
    neighbor_entropy_4 = [torch.tensor(1) for _ in range(int(label.size()[0]))]
    neighbor_entropy = [torch.tensor(1) for _ in range(int(label.size()[0]))]

    for i in range(int(label.size()[0])):
        if i in data.train_mask:
            for sublist,count in element_counts_0[i].items():
                neighbour_class_0[i].append(count)
            if len(neighbour_class_0[i]) == 0 or len(neighbour_class_0[i]) == 1: continue
            neighbour_class_0[i] = torch.Tensor(neighbour_class_0[i])
            neighbour_class_0[i] = neighbour_class_0[i]/neighbour_class_0[i].sum()
            neighbor_entropy_0[i] = -1 * neighbour_class_0[i] * torch.log(neighbour_class_0[i] + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
            neighbor_entropy_0[i] = neighbor_entropy_0[i].sum()

            for sublist,count in element_counts_1[i].items():
                neighbour_class_1[i].append(count)
            if len(neighbour_class_1[i]) == 0 or len(neighbour_class_1[i]) == 1: continue
            neighbour_class_1[i] = torch.Tensor(neighbour_class_1[i])
            neighbour_class_1[i] = neighbour_class_1[i]/neighbour_class_1[i].sum()
            neighbor_entropy_1[i] = -1 * neighbour_class_1[i] * torch.log(neighbour_class_1[i] + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
            neighbor_entropy_1[i] = neighbor_entropy_1[i].sum()

            for sublist,count in element_counts_2[i].items():
                neighbour_class_2[i].append(count)
            if len(neighbour_class_2[i]) == 0 or len(neighbour_class_2[i]) == 1: continue
            neighbour_class_2[i] = torch.Tensor(neighbour_class_2[i])
            neighbour_class_2[i] = neighbour_class_2[i]/neighbour_class_2[i].sum()
            neighbor_entropy_2[i] = -1 * neighbour_class_2[i] * torch.log(neighbour_class_2[i] + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
            neighbor_entropy_2[i] = neighbor_entropy_2[i].sum()

            for sublist,count in element_counts_3[i].items():
                neighbour_class_3[i].append(count)
            if len(neighbour_class_3[i]) == 0 or len(neighbour_class_3[i]) == 1: continue
            neighbour_class_3[i] = torch.Tensor(neighbour_class_3[i])
            neighbour_class_3[i] = neighbour_class_3[i]/neighbour_class_3[i].sum()
            neighbor_entropy_3[i] = -1 * neighbour_class_3[i] * torch.log(neighbour_class_3[i] + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
            neighbor_entropy_3[i] = neighbor_entropy_3[i].sum()

            for sublist,count in element_counts_4[i].items():
                neighbour_class_4[i].append(count)
            if len(neighbour_class_4[i]) == 0 or len(neighbour_class_4[i]) == 1: continue
            neighbour_class_4[i] = torch.Tensor(neighbour_class_4[i])
            neighbour_class_4[i] = neighbour_class_4[i]/neighbour_class_4[i].sum()
            neighbor_entropy_4[i] = -1 * neighbour_class_4[i] * torch.log(neighbour_class_4[i] + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
            neighbor_entropy_4[i] = neighbor_entropy_4[i].sum()
            
            neighbor_entropy[i] = (neighbor_entropy_0[i] + neighbor_entropy_1[i] + neighbor_entropy_2[i] + neighbor_entropy_3[i] + neighbor_entropy_4[i])/5

    local_difficulty = [t.unsqueeze(0) if t.dim() == 0 else t for t in neighbor_entropy]
    local_difficulty = torch.cat(local_difficulty)
    
    return local_difficulty.to(device)

def neighborhood_difficulty_measurer_2(data, label,meta):
    edge_index = data.edge_index
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    num_nodes = torch.max(edge_index) + 1
    adj_list = [[]for _ in range(num_nodes)]
    for src, tgt in zip(source_nodes, target_nodes):
        adj_list[src].append(tgt)
        adj_list[tgt].append(src)
    two_hop_neighbours = [[]for _ in range(int(label.size()[0]))]
    
    # if meta == 1: # 010
    #     # for node in range(int(label.size()[0])):
    #     #     if int(node) in data.train_mask:
    #     #         one_hop_neighbors = adj_list[node]
    #     #         for neighbour in one_hop_neighbors:
    #     #             if neighbour != node and neighbour>= 3025 and neighbour <= 8983: # select which metapath 
    #     #                 for second_hop in adj_list[neighbour]:
    #     #                     if second_hop != node and int(second_hop) in data.train_mask:
    #     #                         two_hop_neighbours[node].append(second_hop)
    #     two_hop_neighbours=torch.load('/MDCL/GNN/2-hop_ACM_010.pt')
    # elif meta ==2: # 020
    #     # for node in range(int(label.size()[0])):
    #     #     if int(node) in data.train_mask:
    #     #         one_hop_neighbors = adj_list[node]
    #     #         for neighbour in one_hop_neighbors:
    #     #             if neighbour != node and neighbour>= 8984 and neighbour <= 9039: # select which metapath 
    #     #                 for second_hop in adj_list[neighbour]:
    #     #                     if second_hop != node and int(second_hop) in data.train_mask:
    #     #                         two_hop_neighbours[node].append(second_hop)
    #     two_hop_neighbours=torch.load('/MDCL/GNN/2-hop_ACM_020.pt')
    # elif meta ==3: # 030
    #     # for node in range(int(label.size()[0])):
    #     #     if int(node) in data.train_mask:
    #     #         one_hop_neighbors = adj_list[node]
    #     #         for neighbour in one_hop_neighbors:
    #     #             if neighbour != node and neighbour>= 9040 and neighbour <= 10942: # select which metapath 
    #     #                 for second_hop in adj_list[neighbour]:
    #     #                     if second_hop != node and int(second_hop) in data.train_mask:
    #     #                         two_hop_neighbours[node].append(second_hop)
    #     two_hop_neighbours=torch.load('/MDCL/GNN/2-hop_ACM_030.pt')
    # for node in range(int(label.size()[0])):
    #     if int(node) in data.train_mask:
    #         continue
    #     else:
    #         two_hop_neighbours[node].clear()


    # two_hop_label = [[]for _ in range(int(label.size()[0]))]
    # for i in range(int(label.size()[0])):
    #     for j in range(len(two_hop_neighbours[i])):
    #         if int(label[two_hop_neighbours[i][j]]) >= 0:
    #             two_hop_label[i].append(label[two_hop_neighbours[i][j]])
    #     two_hop_label[i].append(label[i])
    #     two_hop_label[i]= torch.stack(two_hop_label[i])

    # element_counts = [{} for _ in range(int(label.size()[0]))]
    # for i in range(int(label.size()[0])):
    #     unique_elements, counts = torch.unique(two_hop_label[i], return_counts=True)
    #     element_counts[i] = {elem.item(): count.item() for elem, count in zip(unique_elements, counts)}
    
    # neighbour_class = [[]for _ in range(int(label.size()[0]))]
    # neighbor_entropy = [torch.tensor(1) for _ in range(int(label.size()[0]))]

    # for i in range(int(label.size()[0])):
    #     if i in data.train_mask:
    #         for sublist,count in element_counts[i].items():
    #             neighbour_class[i].append(count)
    #         if len(neighbour_class[i]) == 0 or len(neighbour_class[i]) == 1: continue
    #         neighbour_class[i] = torch.Tensor(neighbour_class[i])
    #         neighbour_class[i] = neighbour_class[i]/neighbour_class[i].sum()
    #         neighbor_entropy[i] = -1 * neighbour_class[i] * torch.log(neighbour_class[i] + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
    #         neighbor_entropy[i] = neighbor_entropy[i].sum()

    # local_difficulty = [t.unsqueeze(0) if t.dim() == 0 else t for t in neighbor_entropy]
    # local_difficulty = torch.cat(local_difficulty)
    
    ### quicker
    if meta == 1:
        local_difficulty =torch.load('/MDCL/GNN/2-hop-diff_ACM_010.pt')
    elif meta == 2:
        local_difficulty =torch.load('/MDCL/GNN/2-hop-diff_ACM_020.pt')
    elif meta == 3:
        local_difficulty =torch.load('/MDCL/GNN/2-hop-diff_ACM_030.pt')

    return local_difficulty.to(device)

def neighborhood_difficulty_measurer_3(data, label,meta):
    edge_index = data.edge_index
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    num_nodes = torch.max(edge_index) + 1
    adj_list = [[]for _ in range(num_nodes)]
    for src, tgt in zip(source_nodes, target_nodes):
        adj_list[src].append(tgt)
        adj_list[tgt].append(src)
    two_hop_neighbours = [[]for _ in range(int(label.size()[0]))]
    four_hop_neighbours = [[]for _ in range(int(label.size()[0]))]

    if meta == 1: # 010
        for node in range(int(label.size()[0])):
            if int(node) in data.train_mask:
                one_hop_neighbors = adj_list[node]
                for neighbour in one_hop_neighbors:
                    if neighbour != node and neighbour>= 4057 and neighbour <= 18384: # select which metapath 
                        for second_hop in adj_list[neighbour]:
                            if second_hop != node and int(second_hop) in data.train_mask:
                                two_hop_neighbours[node].append(second_hop)
    elif meta ==2: # 01210
        four_hop_neighbours = torch.load('/MDCL/GNN/4-hop_DBLP-01210.pt')
        for node in range(int(label.size()[0])):
            if int(node) in data.train_mask:
                continue
            else:
                four_hop_neighbours[node].clear()
        
    elif meta ==3: # 01310
        four_hop_neighbours = torch.load('/MDCL/GNN/4-hop_DBLP-01310.pt')
        for node in range(int(label.size()[0])):
            if int(node) in data.train_mask:
                continue
            else:
                four_hop_neighbours[node].clear()
    if meta !=1:
        two_hop_neighbours = four_hop_neighbours
        
    two_hop_label = [[]for _ in range(int(label.size()[0]))]
    for i in range(int(label.size()[0])):
        for j in range(len(two_hop_neighbours[i])):
            if int(label[two_hop_neighbours[i][j]]) >= 0:
                two_hop_label[i].append(label[two_hop_neighbours[i][j]])
        two_hop_label[i].append(label[i])
        two_hop_label[i]= torch.stack(two_hop_label[i])

    
    element_counts = [{} for _ in range(int(label.size()[0]))]
    for i in range(int(label.size()[0])):
        unique_elements, counts = torch.unique(two_hop_label[i], return_counts=True)
        element_counts[i] = {elem.item(): count.item() for elem, count in zip(unique_elements, counts)}
    
    neighbour_class = [[]for _ in range(int(label.size()[0]))]
    neighbor_entropy = [torch.tensor(1) for _ in range(int(label.size()[0]))]

    for i in range(int(label.size()[0])):
        if i in data.train_mask:
            for sublist,count in element_counts[i].items():
                neighbour_class[i].append(count)
            if len(neighbour_class[i]) == 0 or len(neighbour_class[i]) == 1: continue
            neighbour_class[i] = torch.Tensor(neighbour_class[i])
            neighbour_class[i] = neighbour_class[i]/neighbour_class[i].sum()
            neighbor_entropy[i] = -1 * neighbour_class[i] * torch.log(neighbour_class[i] + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
            neighbor_entropy[i] = neighbor_entropy[i].sum()

    local_difficulty = [t.unsqueeze(0) if t.dim() == 0 else t for t in neighbor_entropy]
    local_difficulty = torch.cat(local_difficulty)

    return local_difficulty.to(device)

# feature-based difficulty measurer
def feature_difficulty_measurer(data, label, embedding):
    normalized_embedding = F.normalize(torch.exp(embedding))
    classes = label.unique()
    class_features = {}
    global_difficulty = [[] for _ in range(5)]
    for i in range(5):
        for c in classes:
            class_nodes = torch.nonzero(label[:,i] == c).squeeze(1)
            node_features = normalized_embedding[class_nodes,i]
            class_feature = node_features.sum(dim=0)
            class_feature = class_feature / len(node_features)
            class_features[c.item()] = class_feature

        similarity = {}
        for u in data.train_mask:
            feature = normalized_embedding[u,i]
            class_feature = class_features[label[u,i].item()]
            sim = feature * class_feature
            similarity[u] = sim
        
        class_avg = {}
        for c in classes:
            count = 0.
            sum = 0.
            for u in data.train_mask:
                if label[u,i] == c:
                    count += 1
                    sum += similarity[u]
            class_avg[c.item()] = sum / count
        
        for u in data.train_mask:
            sim = similarity[u] / class_avg[label[u,i].item()]
            sim = torch.tensor(1).to(device) if sim > 0.95 else sim
            node_difficulty = 1-sim
            global_difficulty[i].append(node_difficulty)
        global_difficulty[i] = torch.stack(global_difficulty[i])        
    diff = global_difficulty[0]
    for i in range(1,5):
        diff += global_difficulty[i]
    diff /= 5
    return diff.clone().detach().to(device)

def feature_difficulty_measurer_2(data, label, embedding):
    normalized_embedding = F.normalize(torch.exp(embedding))
    classes = label.unique()
    class_features = {}
    global_difficulty = []

    edge_index = data.edge_index
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    num_nodes = torch.max(edge_index) + 1
    adj_list = [[]for _ in range(num_nodes)]
    for src, tgt in zip(source_nodes, target_nodes):
        adj_list[src].append(tgt)
        adj_list[tgt].append(src)    

    for c in classes:
        class_nodes = torch.nonzero(label == c).squeeze(1)
        node_features = normalized_embedding[class_nodes]
        for i,node in enumerate(class_nodes):
            count_1 = count_2 = count_3 = 0
            one_hop_neighbors = adj_list[node]
            for neighbour in one_hop_neighbors:
                if neighbour != node and neighbour>= 3025 and neighbour <= 8983: #01
                    if count_1 == 0:
                        averaged_1 = normalized_embedding[neighbour].clone()
                        count_1 += 1                   
                    else: 
                        averaged_1 += normalized_embedding[neighbour]
                        count_1 +=1
                elif neighbour != node and neighbour>= 8984 and neighbour <= 9039: #02
                    if count_2 == 0:
                        averaged_2 = normalized_embedding[neighbour].clone()
                        count_2 += 1
                    else:
                        averaged_2 += normalized_embedding[neighbour]
                        count_2 +=1
                elif neighbour != node and neighbour>= 9040 and neighbour <= 10942: #03
                    if count_3 == 0:
                        averaged_3 = normalized_embedding[neighbour].clone()
                        count_3 += 1
                    else:
                        averaged_3 += normalized_embedding[neighbour]
                        count_3 +=1
                else:
                    continue
            node_type = 4
            averaged_1 /= count_1
            averaged_2 /= count_2
            averaged_3 /= count_3
            if not count_1:
                averaged_1 = torch.zeros(len(averaged_1)).to(device)
                node_type -= 1
            if not count_2:
                averaged_2 = torch.zeros(len(averaged_2)).to(device)
                node_type -= 1
            if not count_3:
                averaged_3 = torch.zeros(len(averaged_3)).to(device)
                node_type -= 1
            node_features[i] = (node_features[i]+averaged_1+averaged_2+averaged_3)/node_type
            node_features[i]/=node_features[i].sum()
        class_feature = node_features.sum(dim=0)
        class_feature = class_feature / torch.sqrt((class_feature * class_feature).sum())
        class_features[c.item()] = class_feature

    similarity = {}
    for u in data.train_mask:
        feature = normalized_embedding[u]
        class_feature = class_features[label[u].item()]
        sim = torch.dot(feature, class_feature)
        similarity[u.item()] = sim
    
    class_avg = {}
    for c in classes:
        count = 0.
        sum = 0.
        for u in data.train_mask:
            if label[u] == c:
                count += 1
                sum += similarity[u.item()]
        class_avg[c.item()] = sum / count
    sim_array = []
    for u in data.train_mask:
        sim = similarity[u] / class_avg[label[u].item()]
        sim = torch.tensor(1).to(device) if sim > 0.95 else sim
        node_difficulty = 1-sim
        global_difficulty.append(node_difficulty)
    global_difficulty = torch.stack(global_difficulty)
    return global_difficulty.clone().detach().to(device)

def feature_difficulty_measurer_3(data, label, embedding):
    normalized_embedding = F.normalize(torch.exp(embedding)) 
    classes = label.unique()
    class_features = {}
    global_difficulty = []

    edge_index = data.edge_index
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    num_nodes = torch.max(edge_index) + 1
    adj_list = [[]for _ in range(num_nodes)]
    for src, tgt in zip(source_nodes, target_nodes):
        adj_list[src].append(tgt)
        adj_list[tgt].append(src)    

    for c in classes:
        class_nodes = torch.nonzero(label == c).squeeze(1)
        node_features = normalized_embedding[class_nodes]
        for i,node in enumerate(class_nodes):
            count_1 = count_2 = count_3 = 0
            one_hop_neighbors = adj_list[node]
            for neighbour in one_hop_neighbors:
                if neighbour != node and neighbour>= 4057 and neighbour <= 18384: #01
                    if count_1 == 0:
                        averaged_1 = normalized_embedding[neighbour].clone()
                        count_1 += 1                   
                    else: 
                        averaged_1 += normalized_embedding[neighbour]
                        count_1 +=1
                else:
                    continue
            node_type = 2
            averaged_1 /= count_1
            if not count_1:
                averaged_1 = torch.zeros(len(averaged_1)).to(device)
                node_type -= 1
            node_features[i] = (node_features[i]+averaged_1)/node_type

            node_features[i]/=node_features[i].sum()
        class_feature = node_features.sum(dim=0)

        class_feature = class_feature / torch.sqrt((class_feature * class_feature).sum())
        class_features[c.item()] = class_feature
    
    similarity = {}
    for u in data.train_mask:
        feature = normalized_embedding[u]
        class_feature = class_features[label[u].item()]
        sim = torch.dot(feature, class_feature)
        similarity[u.item()] = sim
    
    class_avg = {}
    for c in classes:
        count = 0.
        sum = 0.
        for u in data.train_mask:
            if label[u] == c:
                count += 1
                sum += similarity[u.item()]
        class_avg[c.item()] = sum / count

    for u in data.train_mask:
        sim = similarity[u] / class_avg[label[u].item()]
        sim = torch.tensor(1).to(device) if sim > 0.95 else sim
        node_difficulty = 1-sim
        global_difficulty.append(node_difficulty)
    global_difficulty = torch.stack(global_difficulty)
    return global_difficulty.clone().detach().to(device)

# multi-perspective difficulty measurer
def difficulty_measurer(data, label, embedding):
    diff = []   
    global_difficulty = feature_difficulty_measurer(data, label, embedding)
    glb_diff = torch.zeros(int(label.size()[0])).to(device)
    glb_diff[data.train_mask] = global_difficulty
    diff.append(glb_diff)
    for meta in range(1,4):
        local_difficulty = neighborhood_difficulty_measurer(data, label, meta)
        diff.append(local_difficulty)
    averaged_loss = torch.load('/MDCL/HAN/averaged_loss.pt')
    loss_difficulty = torch.ones(int(label.size()[0])).to(device)
    loss_difficulty[data.train_mask] = averaged_loss/averaged_loss.max()
    diff.append(loss_difficulty)
    torch.save(diff,'/MDCL/HAN/diff.pt')
    return diff

def difficulty_measurer_2(data, label, embedding):
    diff = []
    global_difficulty = feature_difficulty_measurer_2(data, label, embedding)
    glb_diff = torch.zeros(int(label.size()[0])).to(device)
    glb_diff[data.train_mask] = global_difficulty
    diff.append(glb_diff)
    for meta in range(1,4):
        local_difficulty = neighborhood_difficulty_measurer_2(data, label, meta)
        diff.append(local_difficulty)
    averaged_loss = torch.load('/MDCL/HAN/averaged_loss.pt')
    loss_difficulty = torch.ones(int(label.size()[0])).to(device)
    loss_difficulty[data.train_mask] = averaged_loss/averaged_loss.max()
    diff.append(loss_difficulty)
    torch.save(diff,'/MDCL/HAN/diff.pt')
    return diff

def difficulty_measurer_3(data, label, embedding):
    diff = []
    global_difficulty = feature_difficulty_measurer_3(data, label, embedding)
    glb_diff = torch.zeros(int(label.size()[0])).to(device)
    glb_diff[data.train_mask] = global_difficulty
    diff.append(glb_diff)
    for meta in range(1,4):
        local_difficulty = neighborhood_difficulty_measurer_3(data, label, meta)
        diff.append(local_difficulty)
    averaged_loss = torch.load('/MDCL/HAN/averaged_loss.pt')
    loss_difficulty = torch.ones(int(label.size()[0])).to(device)
    loss_difficulty[data.train_mask] = averaged_loss/averaged_loss.max()
    diff.append(loss_difficulty)
    torch.save(diff,'/MDCL/HAN/diff.pt')
    return diff


def weighting(data, label, embedding, alpha,loss_alpha):
    correlation = []
    if os.path.exists('/MDCL/HAN/diff.pt'):
        diff = torch.load('/MDCL/HAN/diff.pt')
    else:
        diff = difficulty_measurer(data, label, embedding)
    if alpha == [0,0,0,0,0]:
        averaged_loss = torch.load('/MDCL/HAN/averaged_loss.pt')
        mean_averaged_loss = torch.mean(averaged_loss)
        for i in range(5):
            for j in range(i+1,5):
                if j-i>0:
                    a = diff[i][data.train_mask]
                    b = diff[j][data.train_mask]
                    correlation.append(spearman_correlation(a,b))
            diff[i] = diff[i][data.train_mask]
            alpha[i] = spearman_correlation(diff[i],averaged_loss)
            if math.isnan(alpha[i]):
                alpha[i] = torch.tensor(0)
        alpha = [torch.abs(tensor).item() for tensor in alpha]
        alpha[4] = (alpha[0] + alpha[1] + alpha[2] + alpha[3])/4      
        diff = torch.load('/MDCL/HAN/diff.pt')
        print(alpha)
        node_difficulty = alpha[0]* diff[0]
        for i in range(1,5):
            node_difficulty += alpha[i]* diff[i]
        return node_difficulty
    node_difficulty = alpha[0]* diff[0]
    for i in range(1,5):
        node_difficulty += alpha[i]* diff[i]
    return node_difficulty

def weighting_2(data, label, embedding, alpha,loss_alpha):
    correlation = []
    if os.path.exists('/MDCL/HAN/diff.pt'):
        diff = torch.load('/MDCL/HAN/diff.pt')
    else:
        diff = difficulty_measurer_2(data, label, embedding)
    if alpha == [0,0,0,0,0]:
        averaged_loss = torch.load('/MDCL/HAN/averaged_loss.pt')
        mean_averaged_loss = torch.mean(averaged_loss)
        for i in range(5):
            for j in range(i+1,5):
                if j-i>0:
                    a = diff[i][data.train_mask]
                    b = diff[j][data.train_mask]
                    correlation.append(spearman_correlation(a,b))
            diff[i] = diff[i][data.train_mask]
            alpha[i] = spearman_correlation(diff[i],averaged_loss)
            if math.isnan(alpha[i]):
                alpha[i] = torch.tensor(0)
        alpha = [torch.abs(tensor).item() for tensor in alpha]
        alpha[4] = (alpha[0] + alpha[1] + alpha[2] + alpha[3])/4       
        diff = torch.load('/MDCL/HAN/diff.pt')
        print(alpha)
        node_difficulty = alpha[0]* diff[0]
        for i in range(1,5):
            node_difficulty += alpha[i]* diff[i]
        return node_difficulty
    node_difficulty = alpha[0]* diff[0]
    for i in range(1,5):
        node_difficulty += alpha[i]* diff[i]
    return node_difficulty

def weighting_3(data, label, embedding, alpha,loss_alpha):
    correlation = []
    if os.path.exists('/MDCL/HAN/diff.pt'):
        diff = torch.load('/MDCL/HAN/diff.pt')
    else:
        diff = difficulty_measurer_3(data, label, embedding)
    if alpha == [0,0,0,0,0]:
        averaged_loss = torch.load('/MDCL/HAN/averaged_loss.pt')
        mean_averaged_loss = torch.mean(averaged_loss)
        for i in range(5):
            for j in range(i+1,5):
                if j-i>0:
                    a = diff[i][data.train_mask]
                    b = diff[j][data.train_mask]
                    correlation.append(spearman_correlation(a,b))
            diff[i] = diff[i][data.train_mask]
            alpha[i] = spearman_correlation(diff[i],averaged_loss)
            if math.isnan(alpha[i]):
                alpha[i] = torch.tensor(0)
        alpha = [torch.abs(tensor).item() for tensor in alpha]
        alpha[4] = (alpha[0] + alpha[1] + alpha[2] + alpha[3])/4
        diff = torch.load('/MDCL/HAN/diff.pt')
        print(alpha)
        node_difficulty = alpha[0]* diff[0]
        for i in range(1,5):
            node_difficulty += alpha[i]* diff[i]
        return node_difficulty
    node_difficulty = alpha[0]* diff[0]
    for i in range(1,5):
        node_difficulty += alpha[i]* diff[i]
    return node_difficulty

# sort training nodes by difficulty
def sort_training_nodes(data, label, embedding, alpha,loss_alpha =1):
    node_difficulty = weighting(data, label, embedding, alpha,loss_alpha)
    idx, indices = torch.sort(node_difficulty)
    return indices.to(device)

def sort_training_nodes_2(data, label, embedding, alpha,loss_alpha =1):
    node_difficulty = weighting_2(data, label, embedding, alpha,loss_alpha)
    idx, indices = torch.sort(node_difficulty)
    return indices.to(device)

def sort_training_nodes_3(data, label, embedding, alpha,loss_alpha =1):
    node_difficulty = weighting_3(data, label, embedding, alpha,loss_alpha)
    idx, indices = torch.sort(node_difficulty)
    return indices.to(device)

def sort_mccl(data, label, embedding, size):
    if os.path.exists('/MDCL/HAN/diff.pt'):
        diff = torch.load('/MDCL/HAN/diff.pt')
    else:
        diff = difficulty_measurer(data, label, embedding)
    average_index = []
    for i in range(len(diff)):
        diff[i] = diff[i][data.train_mask]
        x = int(size * diff[i].shape[0])
        idx, indices = torch.sort(diff[i])
        diff[i] = diff[i][indices]        
        average_index.append(diff[i][:x].mean())
    chosen_index = min(range(len(average_index)), key=lambda i: average_index[i])
    diff = torch.load('/MDCL/HAN/diff.pt')
    idx, indices = torch.sort(diff[chosen_index])
    return indices.to(device)

def sort_mccl_2(data, label, embedding, size):
    if os.path.exists('/MDCL/HAN/diff.pt'):
        diff = torch.load('/MDCL/HAN/diff.pt')
    else:
        diff = difficulty_measurer_2(data, label, embedding)
    average_index = []
    for i in range(len(diff)):
        diff[i] = diff[i][data.train_mask]
        x = int(size * diff[i].shape[0])
        idx, indices = torch.sort(diff[i])
        diff[i] = diff[i][indices]        
        average_index.append(diff[i][:x].mean())
    chosen_index = min(range(len(average_index)), key=lambda i: average_index[i])
    diff = torch.load('/MDCL/HAN/diff.pt')
    idx, indices = torch.sort(diff[chosen_index])
    return indices.to(device)


def sort_mccl_3(data, label, embedding,size):
    if os.path.exists('/MDCL/HAN/diff.pt'):
        diff = torch.load('/MDCL/HAN/diff.pt')
    else:
        diff = difficulty_measurer_3(data, label, embedding)
    average_index = []
    for i in range(len(diff)):
        diff[i] = diff[i][data.train_mask]
        x = int(size * diff[i].shape[0])
        idx, indices = torch.sort(diff[i])
        diff[i] = diff[i][indices]        
        average_index.append(diff[i][:x].mean())
    chosen_index = min(range(len(average_index)), key=lambda i: average_index[i])
    diff = torch.load('/MDCL/HAN/diff.pt')
    idx, indices = torch.sort(diff[chosen_index])
    return indices.to(device)

def sort(data, label):
    difficulty = neighborhood_difficulty_measurer(data, label)
    _, indices = torch.sort(difficulty)
    sorted_trainset = data.train_id[indices]
    return sorted_trainset


def training_scheduler(lam, t, T, scheduler='linear'):
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))


def standard_split(data, classes=5):
    data.train_mask = torch.full(data.y.shape, False)
    data.val_mask = torch.full(data.y.shape, False)
    data.test_mask = torch.full(data.y.shape, False)
    for i in range(0, classes):
        count = 0
        for index in range(10000):
            if data.y[index] == i:
                data.train_mask[index] = True
                count += 1
                if count >= 20:
                    break
    data.val_mask[-1500:-1000] = True
    data.test_mask[-1000:] = True
    data.train_id = data.train_mask.nonzero().squeeze(dim=1)
    return data


def random_split(data, percent, num_classes):
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)
    training_nodes_num = int(data.num_nodes * percent)

    # pick at least one node for each class
    class_to_node = {}
    for i in range(data.num_nodes - num_classes):
        node = node_id[i]
        node_class = data.y[node].item()
        if node_class not in class_to_node:
            class_to_node[node_class] = node
            node_id = np.delete(node_id, i)

    for i in range(num_classes):
        node = class_to_node[i]
        node_id = np.insert(node_id, 0, node)

    train_ids = torch.tensor(node_id[:training_nodes_num], dtype=torch.long)
    val_ids = torch.tensor(node_id[training_nodes_num:training_nodes_num + 500], dtype=torch.long)
    test_ids = torch.tensor(node_id[training_nodes_num + 500:training_nodes_num + 1500], dtype=torch.long)

    data.train_mask = torch.full(data.y.shape, False)
    data.train_mask[train_ids] = True
    data.val_mask = torch.full(data.y.shape, False)
    data.val_mask[val_ids] = True
    data.test_mask = torch.full(data.y.shape, False)
    data.test_mask[test_ids] = True
    data.train_id = data.train_mask.nonzero().squeeze(dim=1)
    return data



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_wrong_label(true_label, classes, attack='uniform'):
    if attack == 'uniform':
        labels = np.arange(classes)
        np.delete(labels, true_label)
        return random.choice(labels)
    else:
        return (true_label + classes - 1) % classes


def get_noisy_data(data, percent, attack='uniform'):
    data.train_id = data.train_mask.nonzero().squeeze(dim=1)
    noisy_data = copy.deepcopy((data))
    train_ids = data.train_id.cpu() 
    np.random.shuffle(train_ids.numpy())
    wrong_ids = train_ids[:int(percent * train_ids.shape[0])].to(device)

    for wrong_id in wrong_ids:
        noisy_data.y[wrong_id] = get_wrong_label(noisy_data.y[wrong_id].cpu(), data.num_classes, attack)

    val_ids = torch.nonzero(noisy_data.val_mask).squeeze(dim=1).cpu()
    np.random.shuffle(val_ids.numpy())
    val_wrong_ids = val_ids[:int(percent * val_ids.shape[0])].to(device)
    for wrong_id in val_wrong_ids:
        noisy_data.y[wrong_id] = get_wrong_label(noisy_data.y[wrong_id].cpu(), data.num_classes)
    return noisy_data

def get_wrong_multilabel_label(label, num_classes, attack='uniform'):
    label = label.numpy()  
    if attack == 'uniform':
        wrong_indices = np.random.randint(0, num_classes, size=np.random.randint(1, num_classes))
        label[wrong_indices] = 1 - label[wrong_indices]  
    return torch.tensor(label)

def get_noisy_multilabel_data(data, percent, attack='uniform'):
    data.train_id = data.train_mask.nonzero().squeeze(dim=1)
    noisy_data = copy.deepcopy((data))
    
    train_ids = data.train_id.cpu()
    np.random.shuffle(train_ids.numpy())
    wrong_ids = train_ids[:int(percent * train_ids.shape[0])].to(device)

    for wrong_id in wrong_ids:
        noisy_data.y[wrong_id] = get_wrong_multilabel_label(
            noisy_data.y[wrong_id].cpu(), data.num_classes, attack)

    val_ids = torch.nonzero(noisy_data.val_mask).squeeze(dim=1).cpu()
    np.random.shuffle(val_ids.numpy())
    val_wrong_ids = val_ids[:int(percent * val_ids.shape[0])].to(device)
    for wrong_id in val_wrong_ids:
        noisy_data.y[wrong_id] = get_wrong_multilabel_label(
            noisy_data.y[wrong_id].cpu(), data.num_classes)
    
    return noisy_data