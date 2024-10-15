import torch
from sklearn.metrics import f1_score
import dgl
from utils import load_data, EarlyStopping
#import torch.nn.functional as F
from model_hetero_multi import HAN
import numpy as np
import random
import os
import math
from utils import training_scheduler, sort_training_nodes, sort_training_nodes_2, setup_seed, get_noisy_multilabel_data, sort_mccl
from torch_geometric.data import Data
import wandb
from scripts.data_loader import data_loader

mccl = False
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

def training_scheduler(lam, t, T, scheduler='linear'):
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))

def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def main(args,lamb,big_t,i,noise_rate):
    seed_torch(123+i)
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask, meta_paths, dl, adjM = load_data(args['dataset'], feat_type=0)

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()
    
    dl = data_loader('/HGB/NC/benchmark/methods/data/IMDB')
    labels_1 = np.zeros((21420, dl.labels_train['num_classes']), dtype=int)
    meta_path_1 = [['dm', 'md']]
    meta_path_2 = [['am', 'ma']]
    meta_path_3 = [['km', 'mk']]

    labels_1[train_idx] = dl.labels_train['data'][train_idx]
    labels_1[val_idx] = dl.labels_train['data'][val_idx]
    labels_1 = [np.argmax(l) for l in labels_1]  # one-hot to value
    labels_1 = torch.LongTensor(labels_1)
    test_mask_1 = dl.labels_test['mask'][:]
    test_mask_1 = torch.BoolTensor(test_mask_1)

    feats_type =0
    if feats_type == 0:
        in_dims = [feature.shape[1] for feature in features]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = [] # [features[0].shape[1]] + [10] * (len(features) - 1)
        for i in range(0, len(features)):
            if i == save:
                in_dims.append(features[i].shape[1])
            else:
                in_dims.append(10)
                features[i] = torch.zeros((features[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features]
        for i in range(0, len(features)):
            if i == save:
                in_dims[i] = features[i].shape[1]
                continue
            dim = features[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features]
        for i in range(len(features)):
            dim = features[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)

    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])
    test_mask_1 = test_mask_1.to(args['device'])
    labels_1 = labels_1.to(args['device'])
    model = HAN(
            meta_paths=meta_paths,
            in_size=in_dims[0],
            hidden_size=args['hidden_units'],
            out_size=num_classes,
            num_heads=args['num_heads'],
            dropout=args['dropout']).to(args['device'])
    model_1 = HAN(
            meta_paths=meta_path_1,
            in_size=in_dims[1],
            hidden_size=args['hidden_units'],
            out_size=num_classes,
            num_heads=args['num_heads'],
            dropout=args['dropout']).to(args['device'])
    model_2 = HAN(
            meta_paths=meta_path_2,
            in_size=in_dims[2],
            hidden_size=args['hidden_units'],
            out_size=num_classes,
            num_heads=args['num_heads'],
            dropout=args['dropout']).to(args['device'])
    model_3 = HAN(
            meta_paths=meta_path_3,
            in_size=in_dims[3],
            hidden_size=args['hidden_units'],
            out_size=num_classes,
            num_heads=args['num_heads'],
            dropout=args['dropout']).to(args['device'])

    g = g.to(args['device'])

    edge_index = torch.Tensor(adjM.toarray()).nonzero().t().contiguous()
    data = Data(edge_index= edge_index, y = labels)
    data.num_nodes = 21420
    data.train_mask = torch.tensor(train_idx)
    data.val_mask = torch.tensor(val_idx)
    data.test_mask = test_idx
    data.num_classes = num_classes
    data.num_edge_type = 7
    data = get_noisy_multilabel_data(data,noise_rate*0.05)
    labels = data.y
    data.train_mask = train_idx
    data.val_mask = val_idx

    stopper = EarlyStopping(patience=args['patience'])
    stopper_2 = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    

    for train_round in range(1):
        if not mccl:
            if args['dm'] >= 1:
                if args['dm'] == 1: # Topo
                    alpha = [0,1,1,1,0]
                elif args['dm'] == 2: # Feat
                    alpha = [1,0,0,0,0]
                elif args['dm'] == 3: # Loss
                    alpha = [0,0,0,0,1]
                elif args['dm'] == 4: # CLNode
                    alpha = [1,0.33,0.33,0.33,0]
                elif args['dm'] == 5: # AvgDiff
                    alpha = [1,1,1,1,1]
                elif args['dm'] == 6: # Ours
                    alpha = [0,0,0,0,0]
                elif args['dm'] == 7: # Topo 1
                    alpha = [0,1,0,0,0]
                elif args['dm'] == 8: # Topo 2
                    alpha = [0,0,1,0,0]
                elif args['dm'] == 9: # Topo 3
                    alpha = [0,0,0,1,0]
                embedding = torch.load('/MDCL/HAN/embedding.pt')
                sorted_trainset = sort_training_nodes(data, labels, embedding, alpha,loss_alpha=args['loss_alpha'])
                if not torch.is_tensor(train_idx):
                    train_idx = torch.tensor(train_idx)
                mask = torch.isin(sorted_trainset, train_idx.to(args['device']))
                sorted_trainset = sorted_trainset[mask]
        for epoch in range(args['num_epochs']):
            if args['dm'] >= 1:
                size = training_scheduler(lamb, epoch, big_t, 'linear')
                if mccl:
                    embedding = torch.load('/MDCL/HAN/embedding.pt')
                    sorted_trainset = sort_mccl(data, labels, embedding, size)
                    if not torch.is_tensor(train_idx):
                        train_idx = torch.tensor(train_idx)
                    mask = torch.isin(sorted_trainset, train_idx.to(args['device']))
                    sorted_trainset = sorted_trainset[mask]
                    train_m = sorted_trainset[:int(size*sorted_trainset.shape[0])]
                else:
                    train_m = sorted_trainset[:int(size * sorted_trainset.shape[0])]
            model.train()
            model_1.train()
            model_2.train()
            model_3.train()
            logits = model(g, features[0])
            if args['dm'] == 0:
                loss_fcn = torch.nn.BCEWithLogitsLoss(reduction='none')
                loss = loss_fcn(logits[train_idx], labels[train_idx]).mean(dim=1)
                if epoch == 0:
                    averaged_loss = loss
                else:
                    averaged_loss += loss
                loss = loss.mean()
                loss_fcn = torch.nn.BCEWithLogitsLoss()
            else:
                loss_fcn = torch.nn.BCEWithLogitsLoss()
                loss = loss_fcn(logits[train_m], labels[train_m])
                    
            optimizer.zero_grad()
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            optimizer_3.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_1.step()
            optimizer_2.step()
            optimizer_3.step()

            loss_fcn = torch.nn.BCEWithLogitsLoss()
            model.eval()
            with torch.no_grad():
                logits = model(g, features[0])
                val_loss = loss_fcn(logits[val_mask], labels[val_mask])
                early_stop = stopper.step(val_loss.data.item(), 1, model)
                logits = model(g, features[0])
                test_loss = loss_fcn(logits[test_mask], labels[test_mask])
                test_pred = (logits[test_mask].cpu().numpy() > 0).astype(int)
                test_score = dl.evaluate(test_pred)
                # print('Epoch {:d} | Train Loss {:.4f} | Val Loss {:.4f} |'.format(
                #     epoch + 1, loss.item(), val_loss.item()))
            if early_stop:
                if args['dm'] == 0:
                    averaged_loss /= epoch
                    torch.save(averaged_loss,'/MDCL/HAN/averaged_loss.pt')
                break

        model.eval()
        model_1.eval()
        model_2.eval()
        model_3.eval()
        feat = torch.cat(features,dim=0)
        with torch.no_grad():
            logits = model(g, features[0])
            test_loss = loss_fcn(logits[test_mask], labels[test_mask])
            test_pred = (logits[test_mask].cpu().numpy() > 0).astype(int)
            test_score = dl.evaluate(test_pred)
            logits_1 = model_1(g, feat[:7325])
            logits_2 = model_2(g,feat[:13449])
            logits_3 = model_3(g,feat)
            logits = torch.concat((logits,logits_1[7325:]))
            logits = torch.concat((logits,logits_2[13449:]))
            logits = torch.concat((logits,logits_3[21420:]))
            
            # print('Test loss {:.4f} | Test Score {} '.format(
            #     test_loss.item(), test_score))
            print(test_score)
        if args['dm'] == 0:
            embedding = logits
            torch.save(embedding,'/MDCL/HAN/embedding.pt')
            label = data.y[data.train_mask]


if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    parser.add_argument('--model', type=str)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='IMDB',
                        choices=['IMDB'])
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--dm', type=float,default=0)
    parser.add_argument('--loss_alpha', type=float,default=0)
    args = parser.parse_args().__dict__
    

    args = setup(args,0)
    print('linear')
    for k in range(7):
        for i in range(3): # seed
            print('seed:',i)
            if os.path.exists('/MDCL/HAN/averaged_loss.pt'):
                os.remove('/MDCL/HAN/averaged_loss.pt')
                os.remove('/MDCL/HAN/diff.pt')
                os.remove('/MDCL/HAN/embedding.pt')
            args['dm'] = 0
            args = setup(args,i)
            main(args,1,200,i,k)
            for j in range(1,7):
                args['dm'] = j
                set_random_seed(123+i)
                main(args,8/10,50,i,k)
