import time
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
import torch.nn as nn
from torch_geometric.nn import MessagePassing, Linear, MLP, GCNConv, SGConv, GCN2Conv, PointNetConv,GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import APPNP as _APPNP
from torch_geometric.utils import degree, add_self_loops
import math 
from math import log
from torch.nn.parameter import Parameter
from scipy.sparse import csr_matrix, diags, identity, csgraph, hstack,csc_array,csr_array, eye,coo_matrix
from tqdm import tqdm
import torch_geometric.transforms as T


# A single layer linear model (with bias).
class MyLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(MyLinear, self).__init__()
        self.dropout = dropout
        self.linear = Linear(in_channels = in_channels, out_channels = out_channels)

    def forward(self, x, edge_index):
        x = self.linear(x)
        x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
        return x

# An MLP.
class MyMLP(torch.nn.Module):
    def __init__(self, channel_list, dropout):
        super(MyMLP, self).__init__()
        self.mlp = MLP(channel_list = channel_list, dropout = dropout, norm = None)
    
    def forward(self, x, edge_index):
        return self.mlp(x)

class EbdGNN(torch.nn.Module):
    def __init__(self, num_nodes,in_channels1, in_channels2 ,in_channels3, hidden_channels,
                 out_channels,dropout, num_layers,  fi_type='ori',si_type='se', gnn_type='gcn', sw = 0.2, alpha=[],theta=[], device= 'cuda:0'):
        super(EbdGNN, self).__init__()
        self.fi_type = fi_type 
        self.si_type = si_type 
        self.gnn_type = gnn_type
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.sw = sw 
        self.fw = 1. - sw 
        self.device = device
        self.num_nodes=num_nodes
        if fi_type == 'ori':
            self.lin1 = MyLinear(in_channels1, hidden_channels, dropout)
        else:
            self.lin1 = MyLinear(in_channels2, hidden_channels, dropout)
        if si_type =='se':
            self.lin2 = MyLinear(in_channels3, hidden_channels, dropout)
        self.similarity_head = MyLinear(hidden_channels,out_channels,dropout)

        if gnn_type == 'gcn':
            self.backbone = GCN(in_channels = hidden_channels, 
					hidden_channels = hidden_channels, 
					out_channels = out_channels, 
					num_layers = num_layers, 
					dropout = dropout)
        elif gnn_type == 'gat':
            self.backbone = GAT(in_channels = hidden_channels, 
					hidden_channels = hidden_channels, 
					out_channels = out_channels, 
					num_layers = num_layers, 
					dropout = dropout)
        elif gnn_type == 'sgc':
            self.backbone = SGC(in_channels = hidden_channels, 
					hidden_channels = hidden_channels, 
					out_channels = out_channels, 
					K = num_layers, 
					dropout = dropout)
        elif gnn_type == 'appnp':
            K = num_layers
            alpha =alpha
            self.backbone = APPNP(in_channels=hidden_channels, 
                hidden_channels=hidden_channels, 
                out_channels = out_channels , K=K, 
                alpha=alpha, dropout=dropout) 
        elif gnn_type == 'gcn2':
            num_layers=num_layers
            alpha=alpha
            theta = theta
            self.backbone = GCNII(in_channels=hidden_channels, 
                hidden_channels=hidden_channels, out_channels=out_channels, 
                num_layers=num_layers, alpha=alpha, theta=theta, dropout=dropout)
            
    def node_similarity(self, pred, dataset):
        pred = F.softmax(pred, dim=-1)
        edge = dataset.data.edge_index
        nnodes = dataset.data.x.size()[0]
        src, dst = edge[0], edge[1]
        src_pred = pred[src]
        dst_pred = pred[dst]
        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        batch_size = 5000
        similarity = []
        for i in range(0, len(src_pred), batch_size):
            src_batch_pred = src_pred[i:i + batch_size]
            dst_batch_pred = dst_pred[i:i + batch_size]
            similarity.append(F.cosine_similarity(src_batch_pred, dst_batch_pred, dim=-1))
        similarity = torch.cat(similarity)
        similarity_coo = coo_matrix((similarity.cpu().numpy(), (src_np, dst_np)),
                                    shape=(nnodes, nnodes))
        similarity_sum_list = similarity_coo.sum(axis=1).transpose() + similarity_coo.sum(axis=0)
        similarity_sum = torch.tensor(similarity_sum_list).to(self.device).view(-1)
        similarity = similarity * (1. / similarity_sum[src] + 1. / similarity_sum[dst]) / 2
        return similarity
    
    def graph_sparse(self, similairity, graph, args):
        edge = graph.edge_index
        edges_num = edge.shape[1]
        sample_rate = args['sp_rate']
        sample_edges_num = int(edges_num * sample_rate)
        # remove edges from high to low
        degree_norm_sim = similairity
        sorted_dns = torch.sort(degree_norm_sim,descending=True)
        idx =  sorted_dns.indices
        sample_edge_idx = idx[: sample_edges_num]
        edge = edge[:, sample_edge_idx].to(self.device)
        graph.edge_index = edge
        if args['model'] == 'SGC' and args['dataset'] == 'penn':
            graph = T.ToSparseTensor()(graph.to(self.device))
        return  graph  
       
    def forward(self,f,s,edge_index,state='pre'):

        febd = self.lin1(f,edge_index)
        sebd = self.lin2(s, edge_index)
        ebd = self.fw * febd + self.sw * sebd
        ebd = F.relu(ebd)
        if state== 'pre':
            output = self.similarity_head(ebd, edge_index)
            return output
        output = self.backbone(ebd,edge_index)

        return output



class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True):
        super(GCN, self).__init__()

        cached = False
        add_self_loops = False
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x,edge_index):
        # x = data.graph['node_feat']
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, sampling=False, add_self_loops=True):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):

            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops) )
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.elu
        self.sampling = sampling
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x,edge_indexs, adjs=None, x_batch=None):
        if not self.sampling:
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_indexs)
                x = self.bns[i](x)
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x,edge_indexs)
        else:
            x = x_batch
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = self.activation(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x


    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        total_edges = 0
        device = x_all.device
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = self.activation(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

class SGC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, 
                out_channels, K, dropout):
        super(SGC, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.K = K
        self.dropout = dropout

        self.conv = SGConv(in_channels = self.in_channels, out_channels = self.out_channels, K = self.K, cached = True)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
        return x

class APPNP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, 
                out_channels, K, 
                alpha, dropout):
        super(APPNP, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.K = K
        self.alpha = alpha
        self.dropout = dropout

        self.initial = MLP(channel_list = [self.in_channels, self.hidden_channels, self.hidden_channels], dropout = self.dropout, norm = None)
        self.conv = _APPNP(K = self.K, alpha = self.alpha, cached = True)
        self.final = Linear(in_channels = self.hidden_channels, out_channels = self.out_channels, weight_initializer = 'glorot')
    def forward(self, x, edge_index):
        x = self.initial(x)
        x = self.conv(x, edge_index)
        x = self.final(x)
        return x


class GCNII(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, alpha, theta, shared_weights=True,
                 dropout=0.5):
        super(GCNII, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x,edge_index):
        n = x.size()[0]
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            edge_weight = None
            adj_t = edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = self.bns[i](x)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x




