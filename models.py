from __future__ import print_function

import copy
import math
import os
import sys
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torch.nn import BatchNorm1d as BN
from torch.nn import Dropout
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq

from torch_cluster import knn_graph
from torch_geometric.nn import (DynamicEdgeConv, EdgeConv, GATConv, GCNConv,
                                NNConv, SplineConv, global_max_pool,
                                global_mean_pool, graclus)
from torch_geometric.utils import add_self_loops, normalized_cut

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class PNptg2(torch.nn.Module):

    def __init__(self,
                 input_size,
                 embedding_size,
                 n_classes,
                 pool_op='max',
                 same_size=False):
        super(PNptg2, self).__init__()
        self.fc_enc = MLP([input_size, 64, 64, 64, 128, 1024])
        if pool_op == 'max':
            self.pool = global_max_pool
        self.fc = MLP([1024, 512, 256, 128, embedding_size])
        #self.fc_enc = MLP([input_size, 64, 64, 64, 128, 1024, 512, 256, embedding_size])
        self.fc_cls = nn.Linear(embedding_size, n_classes)
        self.embedding = None

    def forward(self, gdata):
        x, batch = gdata.x, gdata.batch
        x = self.fc_enc(x)
        emb = self.pool(x,batch)
        self.embedding = x.data
        x = self.fc(emb)
        x = self.fc_cls(x)
        return x

class PNemb(torch.nn.Module):

    def __init__(self, input_size, n_classes):
        super(PNemb, self).__init__()
        self.conv1_0 = nn.Linear(input_size, 64)
        self.conv1_1 = nn.Linear(64, 64)
        self.conv2_0 = nn.Linear(64, 64)
        self.conv2_1 = nn.Linear(64, 128)
        self.conv2_2 = nn.Linear(128, 1024)
        self.conv2_3 = nn.Linear(1024, 512)
        self.conv2_4 = nn.Linear(512, 256)
        self.conv3 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_0(x))
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv2_0(x))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = F.relu(self.conv2_4(x))
        x = self.conv3(x)
        return x

class GCNemb(torch.nn.Module):
    def __init__(self, input_size, n_classes):
        super(GCNemb, self).__init__()
        self.conv1_0 = GCNConv(input_size, 64, improved=False)
        self.conv1_1 = GCNConv(64, 64, improved=False)
        self.conv2_0 = GCNConv(64, 64, improved=False)
        self.conv2_1 = GCNConv(64, 128, improved=False)
        self.conv2_2 = GCNConv(128,1024, improved=False)
        self.conv2_3 = GCNConv(1024, 512, improved=False)
        self.conv2_4 = GCNConv(512, 256, improved=False)
        self.conv3 = GCNConv(256, n_classes, improved=False)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index,num_nodes=x.size(0))
        x = F.relu(self.conv1_0(x, edge_index))
        x = F.relu(self.conv1_1(x, edge_index))
        x = F.relu(self.conv2_0(x, edge_index))
        x = F.relu(self.conv2_1(x, edge_index))
        x = F.relu(self.conv2_2(x, edge_index))
        x = F.relu(self.conv2_3(x, edge_index))
        x = F.relu(self.conv2_4(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

class GCNConvNet(torch.nn.Module):
    def __init__(self,
                input_size,
                embedding_size,
                n_classes,
                batch_size=1,
                pool_op='max',
                same_size=False):
        super(GCNConvNet, self).__init__()
        self.gcn = GCNemb(input_size, embedding_size)
        self.fc = torch.nn.Linear(embedding_size, n_classes)
        if pool_op == 'max':
            self.pool = global_max_pool
        self.bs = batch_size
        self.emb_size = embedding_size
        self.same_size = same_size
        self.embedding = None

    def forward(self, gdata):
        x, edge_index, batch = gdata.x, gdata.edge_index, gdata.batch
        x = self.gcn(x,edge_index)
        emb = self.pool(x, batch)
        x = emb.view(-1, self.emb_size)
        self.embedding = x.data
        x = self.fc(F.relu(x))
        return x

class DEC(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, aggr='max', k=5, pool_op='max', same_size=False):
        super(DEC, self).__init__()
        self.k = k
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), self.k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), self.k, aggr)
        self.lin1 = MLP([128 + 64, 1024])
        if pool_op == 'max':
            self.pool = global_max_pool

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, n_classes))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = self.pool(out, batch)
        out = self.mlp(out)
        return out

class BiLSTM(torch.nn.Module):
    def __init__(self, input_size, n_classes=2, embedding_size=128, hidden_size=256):
        super(BiLSTM, self).__init__()
        self.emb_size = embedding_size
        self.h_size = hidden_size
        self.mlp = MLP([input_size, embedding_size])
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            bidirectional=True, batch_first=True)
        self.lin = Seq(
            MLP([hidden_size, 256]), Dropout(0.5),
            MLP([256, 128]), Dropout(0.5),
            nn.Linear(128, n_classes))

    def init_hidden(self):
        return (torch.randn(2, 2, self.h_size),
                torch.randn(2, 2, self.h_size))

    def forward(self, data):
        # expected input has fixed size objects in batches
        bs = data.batch.max() + 1
        # embedding of th single points
        x = self.mlp(data.x)
        x = x.view(bs, -1, x.size(1))

        # hn = hidden state at time step n (final)
        # hn : (num_layers * num_directions, batch, h_size)
        _, (hn, cn) = self.lstm(x)

        # summing up the two hidden states of the two directions
        emb = hn.sum(0)

        # classify
        x = self.lin(emb)
        return x

class DECSeq(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, dropout=True, k=5, aggr='max',pool_op='max'):
        super(DECSeq, self).__init__()
        self.conv1 = EdgeConv(MLP([2 * 3, 64, 64, 64]), aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])
        if pool_op == 'max':
            self.pool = global_max_pool

        if dropout:
            self.mlp = Seq(
                MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
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
        #out = self.pool(out, batch)
        out = global_max_pool(out,batch)
        out = self.mlp(out)
        return out

def ST_loss(pn_model, gamma=0.001):
    A = pn_model.trans  # BxKxK
    A_t = A.transpose(2, 1).contiguous()
    AA_t = torch.bmm(A, A_t)
    I = torch.eye(A.shape[1]).cuda()  # KxK
    return gamma * (torch.norm(AA_t - I)**2)
