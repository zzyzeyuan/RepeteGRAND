"""
repitition of GRAND using dgl
2023.3.29
zzy
"""

import numpy as np
import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn=False):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.input_dropout = nn.Dropout(input_droprate)
        self.hidden_dropout = nn.Dropout(hidden_droprate)

        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def reset_prameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, x):

        if self.use_bn:
            x = self.bn1(x)
        x = self.input_dropout(x)
        x = F.relu(self.layer1(x))

        if self.use_bn:
            x = self.bn2(x)
        x = self.hidden_dropout(x)
        x = self.layer2(x)

        return x


def drop_node(feats, drop_rate, training):
    """
    :param feats: 原始特征
    :param drop_rate:  default=0.5
    :param training: True or False
    :return: feats 也即增强后的特征
    """
    n = feats.shape[0]
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
    """
    n =10   drop_rate = 0.5
    torch.FloatTensor(np.ones(n) * drop_rate) -> 
    tensor([0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000])
    """
    if training:
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats
    else:
        feats = feats * (1. - drop_rate)

    return feats


def GRANDConv(graph, feats, order):
    """
    :param graph:  dgl.Graph
        The input graph
    :param drop_feat:  Tensor (n_nodes * feat_dim)
        Node features
    :param order:  int
        Propagation Steps
    :return:
    """

    with graph.local_scope():
        ''' Calculate Symmetric normalized adjacency matrix   \hat{A} 
        quick example:
            g = dgl.graph((torch.tensor([0,1,3,2]), torch.tensor([1,3,2,1])))
            *[out]: Graph(num_nodes=4, num_edges=4,ndata_schemes={}, edata_schemes={})
            g.in_degrees().float()
            *[out]: tensor([0., 2., 1., 1.])
            *degs = g.in_degrees().float().clamp(min=1)
            [out]: tensor([1., 2., 1., 1.])
            
            norm = torch.pow(degs, -0.5).unsqueeze(1)
            *[out]: tensor([1.0000, 0.7071, 1.0000, 1.0000])
            norm.unsqueeze(1)
            *[out]: 
            tensor([[1.0000],
                    [0.7071],
                    [1.0000],
                    [1.0000]])
        '''

        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)

        graph.ndata['norm'] = norm
        # apply_edges() : Update the features of the specified edges by the provided function.
        graph.apply_edges(fn.u_mul_v('norm', 'norm', 'weight'))

        """ Graph Conv"""
        x = feats
        y = 0 + feats

        for i in range(order):
            graph.ndata['h'] = x
            graph.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h'))
            x = graph.ndata.pop('h')
            y.add_(x)

        return y / (order + 1)


class GRAND(nn.Module):
    r"""
       Parameters
       -----------
       in_dim: int
           Input feature size. i.e, the number of dimensions of: math: `H^{(i)}`.
       hid_dim: int
           Hidden feature size.
       n_class: int
           Number of classes.
       S: int
           Number of Augmentation samples
       K: int
           Number of Propagation Steps
       node_dropout: float
           Dropout rate on node features.
       input_dropout: float
           Dropout rate of the input layer of a MLP
       hidden_dropout: float
           Dropout rate of the hidden layer of a MLP
       batchnorm: bool, optional
           If True, use batch normalization.
       """

    def __init__(self, in_dim, hid_dim, n_class,
                 S=1, K=3, node_dropout=0.0,
                 input_dropout=0.0,
                 hidden_dropout=0.0,
                 bn=False
                 ):
        super(GRAND, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.S = S
        self.K = K
        self.n_class = n_class

        self.dropout = node_dropout
        self.node_dropout = nn.Dropout(node_dropout)

        self.mlp = MLP(in_dim, hid_dim, n_class, input_dropout, hidden_dropout, bn)

    def forward(self, graph, feats, training=True):

        X = feats
        S = self.S

        if training:
            outputs_list = []
            for s in range(S):
                drop_feat = drop_node(X, self.dropout, True)  # Drop Node
                feat = GRANDConv(graph, drop_feat, self.K)  # Graph Convolution
                outputs_list.append(torch.log_softmax(self.mlp(feat), dim=-1))
            return outputs_list
        else:
            drop_feat = drop_node(X, self.dropout, False)
            X = GRANDConv(graph, drop_feat, self.K)

            return torch.log_softmax(self.mlp(X), dim=-1)
