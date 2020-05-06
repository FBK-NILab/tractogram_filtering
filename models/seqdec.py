import copy
import math
import os
import sys

import numpy as np
#import pointnet_trials as pnt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch_geometric.transforms as T
from torch.autograd import Variable
#from pointnet_mgf import max_mod
from torch.nn import BatchNorm1d as BN
from torch.nn import Dropout
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_cluster import knn_graph, knn as pyg_knn
from torch_geometric.nn import (DynamicEdgeConv, EdgeConv, GATConv, GCNConv,
                                NNConv, SplineConv, global_max_pool,
                                global_mean_pool, graclus)
from torch_geometric.utils import add_self_loops, normalized_cut


def MLP(channels, batch_norm=True):
    if batch_norm:
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
            for i in range(1, len(channels))
        ])
    else:
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i]), ReLU())
            for i in range(1, len(channels))
        ])

class DECSeq(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, dropout=False, k=5, aggr='max',pool_op='max'):
        super(DECSeq, self).__init__()
        self.conv1 = EdgeConv(MLP([2 * input_size, 64, 64, 64], batch_norm=True), aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128], batch_norm=True), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])
        if pool_op == 'max':
            self.pool = global_max_pool

        if dropout:
            self.mlp = Seq(
                MLP([1024, 512],batch_norm=True), Dropout(0.5), MLP([512, 256],batch_norm=True), Dropout(0.5),
                Lin(256, n_classes))
        else:
            self.mlp = Seq(
                MLP([1024, 512]), MLP([512, 256]),
                Lin(256, n_classes))

    def forward(self, data):
        pos, batch, eidx = data.pos, data.batch, data.edge_index
        x1 = self.conv1(pos, eidx)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return out
