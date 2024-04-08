import torch
from torch_geometric.nn.models import GCN, GAT, GIN, LINKX
from torch_geometric.datasets import Planetoid, CoraFull, Amazon, Coauthor, WebKB, WikipediaNetwork, Actor, DeezerEurope, WikiCS, LINKXDataset
from torch_geometric.datasets import (Yelp, Flickr, Reddit2, Reddit)
from ogb.nodeproppred import NodePropPredDataset
from model import MyLinear, MyMLP, SGC, APPNP, GCNII, MGNN, PointNet,MyGCN,EbdGNN,EbdGNN2
import random
from scipy.sparse import csr_matrix, diags, identity, csgraph, hstack,csc_array,csr_array, eye 
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import svds
import scipy
import scipy.io
from scipy.linalg import  clarkson_woodruff_transform

import numpy as np
from sklearn.cluster import KMeans,SpectralClustering
# from sklearn
import sklearn.preprocessing as preprocessing
from sklearn.decomposition import TruncatedSVD, IncrementalPCA,KernelPCA, LatentDirichletAllocation
from sklearn.manifold import Isomap

from collections import Counter
import torch.nn.functional as F
from tqdm import tqdm
import itertools
import gpustat
from  torch_geometric.data import Data
import os
from numpy import zeros, max, sqrt, isnan, isinf, dot, diag, count_nonzero
from numpy.linalg import svd, linalg
from scipy.linalg import svd as scipy_svd
from scipy.sparse.linalg import svds as scipy_svds
from sklearn.preprocessing import label_binarize
import time as te 
import  pandas as pd
script_dir = os.path.dirname(os.path.realpath(__file__))
def get_model(model : str, dataset, args):
	if model == 'Linear':
		model = MyLinear(in_channels = dataset.data.num_features, out_channels = dataset.num_classes, 
					dropout = args['dropout'])
	elif model == 'MLP':
		model = MyMLP(channel_list = [dataset.data.num_features] + [args['hidden_dim']] * (args['num_layers'] - 1) + [dataset.num_classes],
					dropout = args['dropout'])
	elif model == 'GCN':
		model = GCN(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					num_layers = args['num_layers'], 
					dropout = args['dropout'])
	elif model == 'MyGCN':
		model = MyGCN(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					dropout = args['dropout'])
  
	elif model == 'ebdgnn':
		model = EbdGNN( num_nodes = dataset.data.x.size()[0],in_channels1 = dataset.data.num_features,
						in_channels2 = args['fe_dim'],
					   in_channels3 =  args['se_dim']  , hidden_channels= args['hidden_dim'], 
					   out_channels=dataset.num_classes ,
					   dropout=args['dropout'], num_layers=args['num_layers'],  fi_type=args['fi_type'],si_type=args['si_type'], gnn_type=args['gnn_type'], sw= args['sw'],alpha=args['alpha'],theta=args['theta'],device=args['device'])
	
	elif model == 'ebdgnn2':
		model = EbdGNN2( in_channels1 = dataset.data.num_features, 
						in_channels2 = args['fe_dim'],
					   in_channels3 =  args['se_dim']  , hidden_channels= args['hidden_dim'], 
					   out_channels=dataset.num_classes ,
					   dropout=args['dropout'], num_layers=args['num_layers'],  fi_type=args['fi_type'],si_type=args['si_type'], gnn_type=args['gnn_type'], sw= args['sw'])
	
	elif model == 'SGC':
		model = SGC(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					K = args['num_layers'], 
					dropout = args['dropout'])
	elif model == 'GAT':
		model = GAT(in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'],
					out_channels = dataset.num_classes,
					num_layers = args['num_layers'],
					dropout = args['dropout'])
	elif model == 'APPNP':
		model = APPNP(in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'],
					out_channels = dataset.num_classes,
					K = args['num_layers'],
					alpha = args['alpha'],
					dropout = args['dropout'])
	elif model == 'GCNII':
		model = GCNII(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					num_layers = args['num_layers'], 
					alpha = args['alpha'],
					theta = args['theta'],
					dropout = args['dropout'])
	elif model == 'MGNN':
		model = MGNN(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					num_layers = args['num_layers'], 
					alpha = args['alpha'],
					beta = args['beta'],
					theta = args['theta'],
					dropout = args['dropout'],
					attention_method = args['attention_method'],
					initial = args['initial'])
	elif model == 'LINKX':
		model = LINKX(num_nodes = dataset.data.x.size()[0],
					in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes,
					num_layers = args['num_layers'],
					dropout = args['dropout'])

	elif model == 'GIN':
		model = GIN(in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					num_layers = args['num_layers'],
					dropout = args['dropout'])
	elif model == 'PointNet':
		model = PointNet(in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					num_layers = args['num_layers'],
					dropout = args['dropout'])
	return model



class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.data = {}
        self.label = None
        self.num_classes=0

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.data, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}


        return split_idx
def load_pokec_mat():
    """ requires pokec.mat
    """
    # if not path.exists(f'{DATAPATH}pokec.mat'):
    #     gdown.download(id=dataset_drive_url['pokec'], \
    #         output=f'{DATAPATH}pokec.mat', quiet=False)
    file_path=os.path.join(script_dir,'data','pokec.mat')
    print(file_path)
    fulldata = scipy.io.loadmat(file_path)

    dataset = NCDataset('pokec')
    dataset.num_classes = 2
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    x = torch.tensor(
        fulldata['node_feat'], dtype=torch.float)
    # num_nodes = int(fulldata['num_nodes'])
    label = fulldata['label'].flatten()
    print(label)
    y = torch.tensor(label, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    dataset.data = data

    return dataset
def load_snap_patents_mat(nclass=5):
    file_path = os.path.join(script_dir, 'data', 'snap_patents.mat')
    fulldata = scipy.io.loadmat(file_path)

    dataset = NCDataset('snap_patents')
    dataset.num_classes = nclass
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    x = torch.tensor(
        fulldata['node_feat'].todense(), dtype=torch.float)
    # num_nodes = int(fulldata['num_nodes'])
    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    y= torch.tensor(label, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    dataset.data = data
    return dataset
def load_arxiv_year_dataset(nclass=5):
    filename = 'arxiv-year'
    dataset = NCDataset(filename)
    dataset.num_classes=nclass
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv')
    raw_data = ogb_dataset.graph
    edge_index = torch.as_tensor(raw_data['edge_index'])
    x = torch.as_tensor(raw_data['node_feat'])
    label = even_quantile_labels(raw_data['node_year'].flatten(), nclass, verbose=True)
    # print('######{}'.format(label))
    y = torch.as_tensor(label,dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    dataset.data = data
    # print('######{}'.format(dataset.data.y))
    return dataset
def load_fb100():
    # e.g. filename = Rutgers89 or Cornell5 or Wisconsin87 or Amherst41
    # columns are: student/faculty, gender, major,
    #              second major/minor, dorm/house, year/ high school
    # 0 denotes missing entry
    filepath = os.path.join(script_dir, 'data', 'Penn94.mat')
    mat = scipy.io.loadmat(filepath)
    A = mat['A']
    metadata = mat['local_info']
    return A, metadata
def load_penn_dataset():
    A, metadata = load_fb100()
    filename='penn'
    dataset = NCDataset(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = np.round(metadata).astype(np.int_)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled
    dataset.num_classes = 2
    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    x = torch.tensor(features, dtype=torch.float)

    y = torch.tensor(label)
    num_nodes = metadata.shape[0]
    data = Data(x=x, edge_index=edge_index, y=y)
    dataset.data = data
    return dataset


def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding

    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()

    return label, features

def load_twitch_gamer_dataset(task="mature", normalize=True):
    # if not path.exists(f'{DATAPATH}twitch-gamer_feat.csv'):
    #     gdown.download(id=dataset_drive_url['twitch-gamer_feat'],
    #                    output=f'{DATAPATH}twitch-gamer_feat.csv', quiet=False)
    # if not path.exists(f'{DATAPATH}twitch-gamer_edges.csv'):
    #     gdown.download(id=dataset_drive_url['twitch-gamer_edges'],
    #                    output=f'{DATAPATH}twitch-gamer_edges.csv', quiet=False)

    edge_file=os.path.join(script_dir,'data','twitch-gamer_edges.csv')
    node_file=os.path.join(script_dir,'data','twitch-gamer_feat.csv')
    edges = pd.read_csv(edge_file)
    nodes = pd.read_csv(node_file)
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
    dataset = NCDataset("twitch-gamer")
    dataset.num_classes = 2
    y=label
    x=node_feat
    data = Data(x=x, edge_index=edge_index, y=y)
    dataset.data = data
    return dataset
def get_dataset(root : str, name : str):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root = root, name = name)
        print(dataset.data.x)
    elif name == 'CoraFull':
        dataset = CoraFull(root = root)
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(root = root, name = name)
    elif name in ['CS', 'Physics']:
        dataset = Coauthor(root = root, name = name)
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root = root, name = name)
    elif name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root = root, name = name.lower())
    elif name == 'Actor':
        dataset = Actor(root = root)
    elif name == 'DeezerEurope':
        dataset = DeezerEurope(root = root)
    elif name == 'WikiCS':
        dataset = WikiCS(root = root, is_undirected = True)
    elif name in ['genius']:
        dataset = LINKXDataset(root = root, name = name)
    elif name in [ 'ogbn-arxiv']:
        dataset = load_arxiv_year_dataset()
    elif name == 'reddit':
        dataset = Reddit(f'{root}/Reddit')
    elif name == 'reddit2':
        dataset = Reddit2(f'{root}/Reddit2')
    elif name == 'flickr':
        dataset = Flickr(f'{root}/Flickr')
    elif name == 'yelp':
        dataset = dataset = Yelp(f'{root}/YELP')
    elif name == 'pokec':
        dataset = load_pokec_mat()
    elif name == 'snap-patents':
        dataset = load_snap_patents_mat()
    elif name=='penn':
         dataset=load_penn_dataset()
    elif name=='twitch_gamer':
        dataset=load_twitch_gamer_dataset()
    else:
        raise Exception('Unknown dataset.')

    print(dataset)
    # exit()
    return dataset

def cal_SEC(dataset):
    print("calculating spanning edge centrality...")
    data = dataset.data
    edges = data.edge_index.numpy()
    rows = edges[0]
    cols = edges[1]
    m = rows.shape[0]
    A = csr_matrix( ([1]*2*m, (np.concatenate([rows, cols]) , np.concatenate([cols, rows]))) )

    D = diags(np.array(A.sum(axis=1)).flatten().tolist())

    E = csr_matrix( ([1]*m+[-1]*m, (np.array(list(range(m))+list(range(m))),np.concatenate([rows, cols]))) )

    L = D-A

    s, U = eigsh(L, k=256, which='LM')

    S = diags(np.array(np.sqrt(1.0/s)).flatten().tolist()).todense()

    Z = E.dot(U.dot(S))

    ER = np.linalg.norm(Z, axis=1)

    m = ER.shape[0]

    return ER

def ER_approximation(dataset, k=100):
    data = dataset.data
    edges = data.edge_index.numpy()
    rows = edges[0]
    cols = edges[1]
    m = rows.shape[0]

    E = csr_matrix( ([1.]*m+[-1.]*m, (np.array(list(range(m))+list(range(m))),np.concatenate([rows, cols]))) )
    
    # get the trucated SVD
    U,s,VT = svds(E.transpose(),k)
    app_ER = ((VT.transpose())**2).sum(axis=-1)
    
    return app_ER

def ER_up_approximation(dataset):
    
    edge = dataset.data.edge_index
    edges_num = edge.shape[1]
    nnodes = dataset.data.x.size()[0]
    src, dst = edge[0], edge[1]
    ngh_list = []   
    
    for i in range(nnodes):
        src_ngh = set( dst[torch.where(src==i)[0]].numpy().tolist())
        dst_ngh = set( src[torch.where(dst==i)[0]].numpy().tolist())
        ngh = src_ngh | dst_ngh 
        ngh_list.append(ngh)
    
    te_list = []
    for i in range(edges_num):
        src_ngh = ngh_list[src[i]]
        dst_ngh = ngh_list[dst[i]]
        te_list.append(len( src_ngh & dst_ngh ))
        
    te = torch.tensor(te_list)
    
    ER_up = 2./(2. + te ) 
    
    return ER_up

def ER_low_approximation(dataset):
    
    edge = dataset.data.edge_index
    edges_num = edge.shape[1]
    nnodes = dataset.data.x.size()[0]
    src, dst = edge[0], edge[1]
    node_degree_list = []    
    
    for i in range(nnodes):
        src_ngh = set( dst[torch.where(src==i)[0]].numpy().tolist())
        dst_ngh = set( src[torch.where(dst==i)[0]].numpy().tolist())
        ngh = src_ngh | dst_ngh 
        node_degree_list.append(len(ngh))
    
    node_degree = torch.tensor(node_degree_list)
    ngh_degree_list  = []
    
    # lower bound of effective resistance
    ER_low =  (1./node_degree[src] + 1./node_degree[dst])/2    
    
    return ER_low 

def augment_se(dataset, args):       
    
    se_type = args['se_type']
    k=args['se_dim']
    
    data = dataset.data
    edges = data.edge_index.numpy()
    rows = edges[0]
    cols = edges[1]
    
    A = csr_matrix( ([1.0]*len(rows), (rows , cols)) )
    n = A.shape[0]
    Feat = data.x.numpy()
    Feat = csr_matrix(Feat)
    
    deg = A.sum(axis=-1)
    deg = np.array(deg.flatten().tolist()[0])+1. 
    D =  csr_matrix(( deg , (np.array(list(range(n))), np.array(list(range(n))))), shape=(n,n))
    D1 = csr_matrix(( 1./((deg)**(0.5))  , (np.array(list(range(n))), np.array(list(range(n))))), shape=(n,n))
    L = D - A
    L_norm =  ((D1).dot(A)).dot(D1)
    
    if se_type[-1] == 'l':
        ebd_source = L_norm
    else:
        ebd_source = A  
    
    latent = ebding_function(ebd_source, se_type, k)
    
    Feat = hstack([Feat,latent])
    
    Feat = Feat.tocoo()
    data.x = torch.sparse.FloatTensor(torch.LongTensor([Feat.row.tolist(), Feat.col.tolist()]),
                                        torch.FloatTensor(Feat.data.astype(float))) 
            
    return dataset

def add_edges(dataset):
    data = dataset.data
    X = data.x.numpy()
    X = csr_matrix(X)
    X = preprocessing.normalize(X, norm='l2', axis=1)
    S = X.dot(X.T)
    S.setdiag(0)
    mean = np.mean(S)*10
    S[S<mean]=0
    S = csr_matrix(S)
    S.eliminate_zeros()
    xrows, xcols = S.nonzero()
    SE={}
    for i in range(len(xrows)):
        u = xrows[i]
        v = xcols[i]
        if u>v:
            uv = u
            u = v
            v = uv
        SE[(u,v)] = S[u,v]

    m = data.edge_index.shape[1]
    k = int(0.05*m)
    SE = sorted(SE.items(), key=lambda x: x[1], reverse=True)
    SE = SE[0:k]

    A = np.zeros(shape=(2,len(SE)))
    i=0
    print("adding %d edges"%k)
    for (k, val) in SE:
        (u, v) = k
        A[0,i] = u 
        A[1,i] = v
        i+=1
    
    A = A.astype(int)

    A = torch.from_numpy(A)

    data.edge_index = torch.cat((data.edge_index, A), 1)
    
    return dataset

def graph_sparsifier(dataset, args ):
    
    stype = args['sp_type']
    descending= True
    
    data = dataset.data
    edges_raw = data.edge_index
    edges = data.edge_index.numpy()
    rows = edges[0]
    cols = edges[1]
    m = rows.shape[0]
    n = data.x.size()[0]
    
    sample_rate = args['sp_rate']
    sample_edges_num = int(m * sample_rate) 
    
    if stype== 'ER':
        ER = torch.tensor(ER_approximation(dataset,k=n-1))
        metric = ER 
    
    # the upper_bound of ER tbc...
    elif stype == 'ER_up':
        ER_up =  torch.tensor(ER_up_approximation(dataset)) 
        metric = ER_up 
    
    elif stype == 'ER_low':
        ER_low = torch.tensor(ER_low_approximation(dataset)) 
        metric = ER_low 
    
    # structure cwt embedding similarity degree normed
    elif stype == 'ssd':  
        args['weigh'] =1. 
        ssd = node_similarity(dataset, args) 
        metric = ssd 
    
    # feature cwt embedding similarity degree normed
    elif stype == 'fsd':
        args['weigh'] = 0.  
        fsd = node_similarity(dataset, args) 
        metric = fsd  
    
    elif stype == 'ER_up_fsd':
        weigh = args['weigh']
        ER_up =  torch.tensor(ER_up_approximation(dataset)).cuda()
        args['weigh'] = 0.  
        fsd = node_similarity(dataset, args)         
        ER_up_fsd = weigh*ER_up + (1-weigh)*fsd 
        metric = ER_up_fsd
            
    elif stype == 'ssd_fsd':
        ssd_fsd = node_similarity(dataset, args) 
        metric = ssd_fsd  
        
    if stype == 'rand':     
        idx = [i for i in range(m)]
        random.shuffle(idx)
    else:
        sorted_metric = torch.sort(metric,descending=descending) 
        idx =  sorted_metric.indices
            
    sample_edge_idx = idx[: sample_edges_num]
    edge = edges_raw[:, sample_edge_idx]
    dataset.data.edge_index = edge 
       
    return  dataset 

def ebding_function( ebd_source, ebd_type, ebd_dim,arg_list=[]):
    
    if ebd_type == 'ori':
        ebd = ebd_source
    
    # clarkson_woodruff_transform
    elif ebd_type == 'cwt':
        ebd = clarkson_woodruff_transform(ebd_source.transpose(), ebd_dim).transpose()
        print(ebd.data.shape)
        
    elif ebd_type == 'fd':
        ebd = frequency_direction(ebd_source, ebd_dim)
        
    elif ebd_type == 'svd':
        U, s, VT = svds(ebd_source, k=ebd_dim, which='LM')
        ebd = U 
    
    # matrix factorization
    elif ebd_type == 'pca':
        pca = IncrementalPCA(n_components=ebd_dim) 
        ebd = pca.fit_transform(ebd_source)
        
    elif ebd_type == 'kpca':
        pca = KernelPCA(n_components=ebd_dim) 
        ebd = pca.fit_transform(ebd_source)
    
    # clustering 
    elif ebd_type == 'kmeans': 
        cluster = KMeans(n_clusters = ebd_dim, random_state=42).fit(ebd_source) 
        cluster_c = cluster.cluster_centers_
        ebd = ebd_source.dot(cluster_c.transpose())
    
    elif ebd_type == 'lda':
        lda = LatentDirichletAllocation(n_components=ebd_dim) 
        ebd = lda.fit_transform(ebd_source)
    
    # manifold learning 
    elif ebd_type == 'iso':
        iso = Isomap(n_components=k)
        ebd = iso.fit_transform(ebd_source)
    
    # only for structure
    elif ebd_type == 'ldp': #local degree profile 
        edges = data.edge_index
        edges_num = edges.shape[1]
        nnodes = dataset.data.x.size()[0]
        src, dst = edges[0], edges[1]
        ngh_list = []
        
        #  node degree and node neighbors 
        node_degree_list = []    
        for i in range(nnodes):
            src_ngh = set( dst[torch.where(src==i)[0]].numpy().tolist())
            dst_ngh = set( src[torch.where(dst==i)[0]].numpy().tolist())
            ngh = src_ngh | dst_ngh 
            ngh_list.append(ngh) 
            node_degree_list.append(len(ngh))
        
        node_degree = torch.tensor(node_degree_list)
        ngh_degree_list  = []
        
        for i in range(nnodes):
            ngh = list(ngh_list[i])
            temp = node_degree[ngh].float()
            if float(temp.shape[0]) > 1.:
                temp2 = torch.tensor( [float(temp.shape[0]), float(temp.mean()), float(temp.std()), float(temp.max()), float(temp.min()) ] )
            elif float(temp.shape[0]) == 1.:
                temp2 = torch.tensor( [float(temp.shape[0]), float(temp.mean()), 0., float(temp.max()), float(temp.min()) ] )
            elif float(temp.shape[0]) == 0.:
                temp2 = torch.tensor( [float(temp.shape[0]), 0., 0., 0.,0. ] )
            ngh_degree_list.append(temp2)
        
        ngh_degree = csr_matrix(torch.stack(ngh_degree_list))
        ebd = ngh_degree
    
    # only for structure
    elif ebd_type == 'rwr' or ebd_type == 'rwrk':
        ebd = rwr_clustering(ebd_source, arg_list, clustering_type= ebd_type, k= ebd_dim)


    return ebd 

# get the structure embedding and feature embedding
def se_fe(dataset, args):
    
    se_type = args['se_type']
    se_dim = args['se_dim']
    
    fe_type = args['fe_type']
    fe_dim = args['fe_dim']

    data = dataset.data
    edges = data.edge_index.numpy()
    rows = edges[0]
    cols = edges[1]

    Feat = data.x.numpy()
    nnodes = Feat.shape[0]
    A = csr_matrix( ([1.0]*len(rows), (rows , cols)), (nnodes , nnodes ) )
    

    Feat = csr_matrix(Feat)
    
    m = rows.shape[0]
    n = A.shape[0]

    if se_type == 'rwr' or se_type =='rwrk':
        se_arg_list = [args['rwr_alpha'], args['rwr_x'], args['rwr_subtype'],args['rwr_rate']]
    else:
        se_arg_list = []
    
    sebd = ebding_function( A , se_type, se_dim,se_arg_list) 
    if not isinstance(sebd,np.ndarray):
        sebd = sebd.toarray()
    dataset.data.sebd = sebd 

    febd = ebding_function(Feat, fe_type, fe_dim)
    if not isinstance(febd,np.ndarray):
        febd = febd.toarray()
    dataset.data.febd = febd
    
    return dataset

def node_similarity(dataset, args):
    
    edge = dataset.data.edge_index
    edges_num = edge.shape[1]
    nnodes = dataset.data.x.size()[0]
    src, dst = edge[0], edge[1]
    
    sebd = dataset.data.sebd 
    febd = dataset.data.febd 
    
    src_sebd = sebd[src]
    dst_sebd = sebd[dst]
    src_febd = febd[src] 
    dst_febd = febd[dst] 
    
    # cosine similairity 
    sebd_similarity =  F.cosine_similarity(src_sebd, dst_sebd, dim=-1)
    sebd_similarity_sum_list = []
    
    febd_similarity =  F.cosine_similarity(src_febd, dst_febd, dim=-1)
    febd_similarity_sum_list = []
    
    # sum of neighbor similarity as 
    for i in range(nnodes):
        temp_sum = (sebd_similarity[torch.where(src==i)[0]]).sum() +  (sebd_similarity[torch.where(dst==i)[0]]).sum() 
        sebd_similarity_sum_list.append(temp_sum)
    
    sebd_similarity_sum = torch.tensor(sebd_similarity_sum_list).cuda()
    structure_similarity = sebd_similarity*(1./sebd_similarity_sum[src] + 1./sebd_similarity_sum[dst])/2
    
    for i in range(nnodes):
        temp_sum = (febd_similarity[torch.where(src==i)[0]]).sum() +  (febd_similarity[torch.where(dst==i)[0]]).sum() 
        febd_similarity_sum_list.append(temp_sum)
    
    febd_similarity_sum = torch.tensor(febd_similarity_sum_list).cuda()
    feature_similarity = febd_similarity*(1./febd_similarity_sum[src] + 1./febd_similarity_sum[dst])/2

    node_similarity =  (args['weigh'])*structure_similarity + (1-args['weigh'])*feature_similarity 

    return node_similarity
    

def frequency_direction(ebd_source, ebd_dim):
    
    sketcher = FrequentDirections(ebd_source.shape[1] , ebd_dim )
    
    for i in range(ebd_source.shape[0]): 
        row =  ebd_source[i]    
        sketcher.append(row) 
    
    sketch = sketcher.get()
    
    ebd = ebd_source @ (sketch.transpose())
        
    return ebd 
    
class MatrixSketcherBase:
    def __init__(self, d, ell):
        self.d = d
        self.ell = ell
        self._sketch = zeros((self.ell, self.d))

    # Appending a row vector to sketch
    def append(self, vector):
        pass

    # Convenient looping numpy matrices row by row
    def extend(self, vectors):
        for vector in vectors:
            self.append(vector)

    # returns the sketch matrix
    def get(self):
        return self._sketch

    # Convenience support for the += operator  append
    def __iadd__(self, vector):
        self.append(vector)
        return self


class FrequentDirections(MatrixSketcherBase):
    def __init__(self, d, ell):
        self.class_name = "FrequentDirections"
        self.d = d
        self.ell = ell
        self.m = 2 * self.ell
        self._sketch = zeros((self.m, self.d))
        
        # create a sparse matrix 
        data = np.array([0.])
        self._sketch = csr_matrix( (data , (np.array([0]), np.array([0])) ) , shape=(self.m,self.d)).tolil()
        
        self.nextZeroRow = 0

    def append(self, vector):

        if vector.count_nonzero() == 0:
            return

        if self.nextZeroRow >= self.m:
            self.__rotate__() 


        self._sketch[self.nextZeroRow, :] = vector

    def __rotate__(self):
        
        [_,s,Vt] = scipy_svds(self._sketch, k = self.ell)
        
        s = s[::-1]
        Vt[:len(s), :] = Vt[len(s)-1::-1, :]
        
        if len(s) >= self.ell:
            
            a = s[: self.ell] ** 2 - s[self.ell - 1] ** 2 
            
            sShrunk = sqrt(a)
            
            self._sketch[: self.ell :, :] = dot(diag(sShrunk), Vt[: self.ell, :])
            self._sketch[self.ell :, :] = 0
            self.nextZeroRow = self.ell
        else:
            self._sketch[: len(s), :] = dot(diag(s), Vt[: len(s), :])
            self._sketch[len(s) :, :] = 0
            self.nextZeroRow = len(s)

    def get(self):
        return self._sketch[: self.ell, :]
    

# clustering for dimension reduction
# k is the clustering number and t is the iteraion number 

def rwr_clustering(data, arg_list, clustering_type = 'rwr',  sub_type = 'binary', k=256):

    alpha = arg_list[0]
    t= int(1/alpha)
    x= arg_list[1]
    y = 1.-x 
    t1 = arg_list[3]
    t2 = 1.   
    init_range = 5 *k  
    sub_type = arg_list[2]

    nnodes = data.shape[0]

    # print(data, arg_list)
    # print(data.shape)
    # print(nnodes)

    # choose the initail cluster center 
    ones_vector = np.ones(nnodes,dtype=float)
    degrees = data.dot(ones_vector) 
    degrees_inv = diags( (1./(degrees+1e-10)).tolist())  
    
    if clustering_type == 'rwr':
        # clustering: random walk with restart 
        topk_deg_nodes = np.argpartition(degrees, -init_range)[-init_range:] 
        P = degrees_inv.dot(data)

        # init k clustering centers 
        PC = P[:,topk_deg_nodes] 
        M = PC 
        alpha = arg_list[0]  

        for i in range(t):
            M = (1-alpha)*P.dot(M)+PC

        cluster_sum = M.sum(axis=0).flatten().tolist()[0]
        newcandidates = np.argpartition(cluster_sum, -k)[-k:]
        M = M[:,newcandidates]  

        if sub_type == 'continuous':
            column_sqrt = diags((1./( np.squeeze(np.asarray(M.sum(axis=-1)))  +1e-10)).tolist())
            prob = column_sqrt.dot(M)
            ebd = data.dot(prob)

        elif sub_type == 'binary':
            column_sqrt = diags((1./( np.squeeze(np.asarray(M.sum(axis=-1)))**x  +1e-10)).tolist())
            row_sqrt = diags((1./( np.squeeze(np.asarray(M.sum(axis=0)))**y  +1e-10)).tolist())
            prob = column_sqrt.dot(M).dot(row_sqrt)
            
            center_idx = np.squeeze(np.asarray(prob.argmax(axis=-1)))

            cluster_center = csr_matrix( ([1.]*nnodes, (np.array([i for i in range(nnodes)]), center_idx
                                                        )),
                                        shape=(nnodes,k))
            
            random_flip = diags(np.where(np.random.rand(nnodes)>0.5, 1., -1.).tolist())
            sketching = csr_matrix( ([1.]*nnodes, (np.array([i for i in range(nnodes)]), np.random.randint(0,k,nnodes))), shape=(nnodes,k) )
            sketching = random_flip.dot(sketching)

            ebd = data.dot( ( t1 * random_flip.dot(cluster_center) + t2 * sketching) ) 
        
        elif sub_type == 'random':
            sketching = csr_matrix( ([1.]*nnodes, (np.array([i for i in range(nnodes)]), np.random.randint(0,k,nnodes))), shape=(nnodes,k) )
            random_flip = diags(np.where(np.random.rand(nnodes)>0.5, 1., -1.).tolist())
            sketching = random_flip.dot(sketching)
            print(sketching.sum(axis=0).tolist())
            ebd = data.dot(sketching)
    
    elif clustering_type == 'rwrk':
        # clustering: random walk with restart 
        topk_deg_nodes = np.argpartition(degrees, -init_range)[-init_range:]

        data_deg = data[:, topk_deg_nodes]

        cwt_ebd = clarkson_woodruff_transform(data.transpose(), k).transpose()

        cwt_ebd_deg = cwt_ebd[topk_deg_nodes]

        cluster = KMeans(n_clusters = k, random_state=42).fit(cwt_ebd_deg) 
        labels = cluster.labels_

        center_idx = np.squeeze(np.asarray(labels))

        cluster_center = csr_matrix( ([1.]*init_range, (np.array([i for i in range(init_range)]), center_idx)), shape=(init_range,k))
        
        random_flip = diags(np.where(np.random.rand(init_range)> 0.5, 1., -1.).tolist())

        ebd =  data_deg.dot(random_flip.dot(cluster_center))

        # x = 1. 
        # y= 0. 

        # column_sqrt = diags((1./( np.squeeze(np.asarray(ebd.sum(axis=-1)))**x  +1e-10)).tolist())
        # row_sqrt = diags((1./( np.squeeze(np.asarray(ebd.sum(axis=0)))**y  +1e-10)).tolist())

        # ebd = column_sqrt.dot(ebd).dot(row_sqrt)
    
    return ebd

def rand_train_test_idx(data, train_prop=.6, valid_prop=.2, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(data.y != -1)[0]
    else:
        labeled_nodes = data.y
    # print(labeled_nodes)
    n = labeled_nodes.shape[0]
    # print(n)
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]
    # print(train_indices.size())
    # print(val_indices.size())
    # print(test_indices.size())
    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]
    # print(train_idx.size())
    # print(valid_idx.size())

    return train_idx, valid_idx, test_idx


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int_)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label

        
         
