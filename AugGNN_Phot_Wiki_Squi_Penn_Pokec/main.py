from argparse import ArgumentParser
from run import run 

if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic.
    parser.add_argument('--model', type = str, default = 'MGNN', help = 'Model to use.')
    parser.add_argument('--dataset', type = str, default = 'Cora', help = 'Dataset to use.')

    # Training.
    parser.add_argument('--seed', type = int, default = 42, help = 'Random seed.')
    parser.add_argument('--device', type = str, default = 'cuda', help = 'Training device.')
    parser.add_argument('--times', type = int, default = 10, help = 'Training times.')
    parser.add_argument('--epochs', type = int, default = 1800, help = 'Number of epochs to train.')
    parser.add_argument('--early_stopping', type = int, default = 200, help = 'Early stopping.')
    parser.add_argument('--lr', type = float, default = 1e-2, help = 'Learning rate.')
    parser.add_argument('--weight_decay', type = float, default = 1e-4, help = 'Weight decay.')
    parser.add_argument('--dropout', type = float, default = 0.5, help = 'Dropout rate.')
    parser.add_argument('--setting', type = str, default = '60/20/20', help = 'Training node setting (\'public\', \'GCN\', \'semi\', \'48/32/20\' or \'60/20/20\').')
    
    # Model.
    parser.add_argument('--hidden_dim', type = int, default = 128, help = 'Hidden dimension.')
    parser.add_argument('--num_layers', type = int, default = 2, help = 'Number of convolution layers or propagation hops.')
    parser.add_argument('--alpha', type = float, default = 0.5, help = 'Alpha value for the propagation.')
    parser.add_argument('--beta', type = float, default = 0.5, help = 'Beta value for the propagation.')
    parser.add_argument('--theta', type = float, default = 0.5, help = 'Theta value for the propagation.')
    parser.add_argument('--attention_method', type = str, default = 'concat', help = 'Attention method for the MGNNAttention layer.')
    parser.add_argument('--initial', type = str, default = 'Linear', help = 'Initial embedding method for the MGNN model.')
    
    parser.add_argument('--augment_se', type = int, default = 0, help = '0: no augmentation, 1: expand by the structure')
    parser.add_argument('--se_dim', type = int, default = 128, help = 'the dimension of structural embedding')
    parser.add_argument('--se_type', type = str, default = 'pca' , help = 'the type of structural embedding, ori, kmeans, svd, ldp ...')
    
    parser.add_argument('--augment_sp', type = int, default = 0, help = '0: no augmentation, 1: graph sparsifier')
    parser.add_argument('--sp_rate', type = float, default = 1.0, help = 'the sample rate of sparsifier')
    parser.add_argument('--sp_type', type = str, default =   'rand' , help = 'the type of graph sparsifier, rand, ER, rand_ER')
    parser.add_argument('--weighted', type = bool, default =   False , help = 'weighted adj or not')
    
    # hyper-parameter for ebdgcn
    parser.add_argument('--fe_type', type = str, default = 'pca', help = 'the embedding type of feature')
    parser.add_argument('--fe_dim', type = int, default = 128, help = 'the dimension of feature embedding')
    
    parser.add_argument('--sim_type', type = str, default = 'sep', help = 'type of similarity measurement. sep:\
                        seperation, cat: concat and similarity, mlp: using an mlp which will be used later')
    parser.add_argument('--weigh', type = float, default = 0.5, help = 'a factor to weigh the feature similarity and structure similarity')
    
    parser.add_argument('--fi_type', type = str, default = 'ori' , help = 'the type of feature information')
    parser.add_argument('--si_type', type = str, default = 'se' , help = 'the type of structure information')
    parser.add_argument('--gnn_type', type = str, default = 'gcn' , help = 'the type of backbone gnn')
    parser.add_argument('--sw', type = float, default = 0.2, help = 'a factor to weigh the feature embedding and structure embedding fw = 1.- sw')
    
    
    parser.add_argument('--pepochs', type = int, default = 128, help = 'the pretraining epochs of MLP for graph sparsifier')
    parser.add_argument('--bepochs', type = int, default = 128, help = 'the pretraining epochs of MLP for graph sparsifier')

    parser.add_argument('--rwr_alpha', type = float , default = 0.5, help = 'the alpha of random walk with restart')
    parser.add_argument('--rwr_x', type = float , default = 0.5, help = 'the value of exp of row or column')
    parser.add_argument('--rwr_subtype', type = str , default = 'binary', help = 'the subtype of rwr embedding')
    parser.add_argument('--rwr_rate', type=float, default=1.0, help='the rate of rwr embedding')
    args = vars(parser.parse_args())
    
    run(args = args, verbose = True)
