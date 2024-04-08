import time

import scipy
import os
import random
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.profile import count_parameters
from torch_geometric.transforms import RandomNodeSplit
from our_utils import get_model, get_dataset, graph_sparsifier, augment_se, se_fe
import time as te
import torch_geometric.transforms as T
from scipy.sparse import coo_matrix


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# train function for GNN
def train(model, graph, loss_fn, optimizer, weighted=False, model_name='ebdgnn', state='pre', fi_type='ori',
          device='cuda:0'):
    model.train()
    optimizer.zero_grad()

    if model_name != 'ebdgnn' and model_name != 'ebdgnn2':
        # whether the edge should be reweighted before training
        if not weighted:
            pred = model.forward(graph.x, graph.edge_index, graph.edge_weight)
        else:
            pred = model.forward(graph.x, graph.edge_index, graph.edge_attr)
    else:
        # ebdnet has different input with other GNN, the stctural embedding
        if fi_type == 'ori':
            pred = model.forward(graph.x, graph.sebd, graph.edge_index, state)
        else:
            pred = model.forward(graph.febd, graph.sebd, graph.edge_index, state)

    y = torch.tensor(graph.y)
    y = y.to(device)
    y = y.squeeze()
    loss = loss_fn(pred[graph.train_mask], y[graph.train_mask])
    del graph
    loss.backward()
    optimizer.step()

# inference function for GNN
@torch.no_grad()
def infer(model, graph, mode: str = 'val',  model_name: str = 'ebdgnn', state: str = 'pre',
          fi_type: str = 'ori', device='cuda:0'):
    model.eval()
    assert mode in ['val', 'test', 'ogb_val', 'ogb_test']
    mask = graph.val_mask if 'val' in mode else graph.test_mask

    if model_name != 'ebdgnn':
        pred = model.forward(graph.x, graph.edge_index)

    else:
        if fi_type == 'ori':
            pred = model.forward(graph.x, graph.sebd, graph.edge_index, state)
        else:
            pred = model.forward(graph.febd, graph.sebd, graph.edge_index, state)

    pred = pred.argmax(dim=1)
    pred1 = pred.clone()
    y = torch.tensor(graph.y)
    y = y.to(device)
    y = y.squeeze()
    correct = (pred[mask] == y[mask]).sum()
    acc = int(correct) / int(mask.sum())

    return acc , pred1


# @profile  # if want to see the code memory usage, use @profile
# running the whole process, augment, train and test
def run(args, verbose: bool = True):
    start_time = te.time()

    # set seed
    if args['seed'] != -1:
        random.seed(args['seed'])
        os.environ['PYTHONHASHSEED'] = str(args['seed'])
        np.random.seed(args['seed'])
        torch.manual_seed(args['seed'])
        torch.cuda.manual_seed(args['seed'])
        torch.cuda.manual_seed_all(args['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # set the device
    device = (args['device'] if torch.cuda.is_available() else 'cpu')
    times = args['times']
    epochs = args['epochs']
    # accuracy lists
    val_accs = []
    test_accs = []

    # load dataset
    dataset = get_dataset(root='../data/{}'.format(args['dataset']), name=args['dataset'])

    # augment: structure embedding
    if args['augment_se'] > 0 and args['model'] != 'ebdgnn' and args['model'] != 'ebdgnn2':
        dataset = augment_se(dataset, args)

    # augment graph
    if args['augment_sp'] > 0 and args['model'] != 'ebdgnn' and args['model'] != 'ebdgnn2':
        dataset = se_fe(dataset, args)
        dataset.data.sebd = torch.tensor(dataset.data.sebd.copy()).float().to(device)
        dataset.data.febd = torch.tensor(dataset.data.febd.copy()).float().to(device)

    # ebdgnn or ebdgnn2
    if args['model'] == 'ebdgnn':
        dataset = se_fe(dataset, args)
        dataset.data.sebd = torch.tensor(dataset.data.sebd.copy()).float().to(device)
        dataset.data.febd = torch.tensor(dataset.data.febd.copy()).float().to(device)
        edge_index = dataset.data.edge_index

    # get the model, optimizer and loss fn
    model = get_model(model=args['model'], dataset=dataset, args=args).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    graph = dataset.data

    for time in range(times):
        if verbose:
            print('Training time: {}'.format(time))

        # empty cache
        if device != 'cpu':
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

        if 'ogbn' in args['dataset']:
            total_length = graph.x.size()[0]
            split_idx = dataset.get_idx_split()
            graph.train_mask = torch.zeros(total_length, dtype=bool)
            graph.train_mask[split_idx['train']] = True
            graph.val_mask = torch.zeros(total_length, dtype=bool)
            graph.val_mask[split_idx['valid']] = True
            graph.test_mask = torch.zeros(total_length, dtype=bool)
            graph.test_mask[split_idx['test']] = True
            
        else:
            # Training node settings.
            if args['setting'] == 'GCN':
                split_gen = RandomNodeSplit(split='random')
            elif args['setting'] == 'semi':
                split_gen = RandomNodeSplit(split='train_rest', num_val=0.025, num_test=0.95)
            elif args['setting'] == '48/32/20':
                split_gen = RandomNodeSplit(split='train_rest', num_val=0.32, num_test=0.20)
            elif args['setting'] == '60/20/20':
                split_gen = RandomNodeSplit(split='train_rest', num_val=0.20, num_test=0.20)
            if args['setting'] != 'public':
                graph = split_gen(graph)

        if not is_undirected(graph.edge_index):
            graph.edge_index = to_undirected(graph.edge_index)

        graph = graph.to(device)

        if args['model'] == 'ebdgnn':
            graph.edge_index = edge_index

        best_val_acc = 0.0
        patience = 0
        state = 'pre'

        model.train()

        for epoch in tqdm(range(epochs)) if verbose else range(epochs):

            train(model=model, graph=graph, loss_fn=loss_fn, optimizer=optimizer, weighted=args['weighted'],
                  model_name=args['model'], state=state, fi_type=args['fi_type'], device=args['device'])

            val_acc, _ = infer(model=model, graph=graph, mode='val', weighted=args['weighted'],
                                               model_name=args['model'], state=state, fi_type=args['fi_type'],
                                               device=args['device'])

            if val_acc > best_val_acc:
                patience = 0
                best_val_acc = val_acc
                best_epoch = epoch
                test_acc, best_pred = infer(model=model, graph=graph, mode='test',
                                                        weighted=args['weighted'], model_name=args['model'],
                                                        state=state, fi_type=args['fi_type'], device=args['device'])
            else:
                patience += 1
                if patience > args['early_stopping']:
                    if args['model'] == 'ebdgnn':
                        if epoch > args['pepochs'] + args['bepochs']:
                            break
                    else:
                        break

            if args['model'] == 'ebdgnn' and epoch == args['pepochs']:
                state = 'train'
                similarity = model.node_similarity(best_pred, dataset)
                graph = model.graph_sparse(similarity, graph, args)
            
            if epoch % 100 == 0:
                print("test acc is {} \n".format(test_acc * 100))

        if verbose:
            print('Training done. Test_acc (on best valid_acc = {:.2f}%) = {:.2f}% , best epoch {} \n'.format(
                100 * best_val_acc, 100 * test_acc, best_epoch))
        val_accs.append(100 * val_acc)
        test_accs.append(100 * test_acc)

    val_accs = np.array(val_accs)
    val_mean = np.mean(val_accs)
    val_stddev = np.std(val_accs)

    test_accs = np.array(test_accs)
    test_mean = np.mean(test_accs)
    test_stddev = np.std(test_accs)
    test_best = np.max(test_accs)

    if verbose:
        print('Using device: {}'.format(device))
        print(args)
        print(' ')
        print('Acc: {:.2f}% {:.2f}$ {:.2f}%'.format(test_mean, test_stddev, test_best) + " by %s on %s of %s" % (
        args['model'], args['dataset'], args['gnn_type']))

    return val_mean, val_stddev, test_mean, test_stddev