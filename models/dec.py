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
from torch_geometric.nn import (DynamicEdgeConv, EdgeConv, global_max_pool)
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

class DEC(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, aggr='max', k=5, pool_op='max', same_size=False):
        super(DEC, self).__init__()
        self.k = k
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), self.k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), self.k, aggr)
        self.lin1 = MLP([128 + 64, 1024])
        if pool_op == 'max':
            self.pool = global_max_pool

        # self.mlp = Seq(
        #     MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
        #     Lin(256, n_classes))
        self.mlp = Seq(
            MLP([1024, 512]),MLP([512, 256]),
            Lin(256, n_classes))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = self.pool(out, batch)
        out = self.mlp(out)
        return out