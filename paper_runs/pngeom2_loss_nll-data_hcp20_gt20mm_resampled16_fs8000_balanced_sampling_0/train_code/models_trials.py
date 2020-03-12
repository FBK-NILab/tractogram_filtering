from __future__ import print_function

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

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class DynamicEdgeConvCosine(DynamicEdgeConv):

    def forward(self, x, batch=None):
        """"""
        edge_index = knn_graph(x,
                               self.k,
                               batch,
                               loop=False,
                               flow=self.flow,
                               cosine=True)
        return super(DynamicEdgeConv, self).forward(x, edge_index)


class PNptg2(torch.nn.Module):

    def __init__(self,
                 input_size,
                 embedding_size,
                 n_classes,
                 pool_op='max',
                 same_size=False):
        super(PNptg2, self).__init__()
        self.fc_enc = MLP([input_size, 64, 64, 64, 128, 1024], batch_norm=False)
        if pool_op == 'max':
            self.pool = global_max_pool
        self.fc = MLP([1024, 512, 256, 128, embedding_size], batch_norm=False)
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

class PNptg(torch.nn.Module):

    def __init__(self,
                 input_size,
                 embedding_size,
                 n_classes,
                 batch_size=1,
                 pool_op='max',
                 same_size=False):
        super(PNptg, self).__init__()
        self.pn = PNemb(input_size, embedding_size)
        self.fc = torch.nn.Linear(embedding_size, n_classes)
        if pool_op == 'max':
            self.pool = global_max_pool
        self.bs = batch_size
        self.emb_size = embedding_size
        self.same_size = same_size
        self.embedding = None

    def forward(self, gdata):
        x, batch = gdata.x, gdata.batch
        x = self.pn(x)
        emb = self.pool(x,batch)
        # x = emb.view(-1, 1024)
        self.embedding = x.data
        x = self.fc(F.relu(x))
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
        self.bn1_0 = BN(64)
        self.conv1_1 = GCNConv(64, 64, improved=False)
        self.bn1_1 = BN(64)
        self.conv2_0 = GCNConv(64, 64, improved=False)
        self.bn2_0 = BN(64)
        self.conv2_1 = GCNConv(64, 128, improved=False)
        self.bn2_1 = BN(128)
        self.conv2_2 = GCNConv(128,1024, improved=False)
        self.bn2_2 = BN(1024)
        self.conv2_3 = GCNConv(1024, 512, improved=False)
        self.bn2_3 = BN(512)
        self.conv2_4 = GCNConv(512, 256, improved=False)
        self.bn2_4 = BN(256)
        self.conv3 = GCNConv(256, n_classes, improved=False)
        self.bn3_0 = BN(n_classes)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index,num_nodes=x.size(0))
        x = F.relu(self.bn1_0(self.conv1_0(x, edge_index)))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.bn1_1(self.conv1_1(x, edge_index)))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_0(self.conv2_0(x, edge_index)))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_1(self.conv2_1(x, edge_index)))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_2(self.conv2_2(x, edge_index)))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_3(self.conv2_3(x, edge_index)))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_4(self.conv2_4(x, edge_index)))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.bn3_0(self.conv3(x, edge_index)))
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
        x = self.fc(x)
        return x

class GCN(GCNConv):
    def forward(self, data):
        return super(GCN, self).forward(data.x, data.edge_index)

class GCNConvNetBN(torch.nn.Module):
    def __init__(self,
                input_size,
                embedding_size,
                n_classes,
                pool_op='max',
                same_size=False):
        super(GCNConvNetBN, self).__init__()
        channels = [input_size, 64, 64, 64, 128, 1024, 512, 256, embedding_size]
        self.gcn = Seq(*[
            Seq(GCN(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
            for i in range(1, len(channels))])
        print(self.gcn)
        self.fc = torch.nn.Linear(embedding_size, n_classes)
        if pool_op == 'max':
            self.pool = global_max_pool
        self.emb_size = embedding_size
        self.same_size = same_size
        self.embedding = None

    def forward(self, gdata):
        x, edge_index, batch = gdata.x, gdata.edge_index, gdata.batch
        edge_index, _ = add_self_loops(edge_index,num_nodes=x.size(0))
        x = self.gcn(gdata)
        emb = self.pool(x, batch)
        x = emb.view(-1, self.emb_size)
        self.embedding = x.data
        x = self.fc(x)
        return x

class NNC1(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, pool_op='max', same_size=False):
        super(NNC, self).__init__()
        nn1 = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, input_size*32))
        self.conv1_0 = NNConv(input_size, 32, nn1)
        nn2 = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32*32))
        self.conv1_1 = NNConv(32, 32, nn2)
        nn3 = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32,32*64))
        self.conv2_0 = NNConv(32,64, nn3)
        nn4 = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 64*64))
        self.conv2_1 = NNConv(64,64, nn4)
        nn5 = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32,64*embedding_size))
        self.conv3 = NNConv(64, embedding_size, nn5)

        self.fc = torch.nn.Linear(embedding_size, n_classes)
        if pool_op == 'max':
            self.pool = global_max_pool
        self.bs = batch_size
        self.emb_size = embedding_size
        self.same_size = same_size
        self.embedding = None

    def forward(self, data):
        x = F.relu(self.conv1_0(data.x, data.edge_index, data.edge_attr))
        x = F.relu(self.conv1_1(x, data.edge_index, data.edge_attr))
        x = F.relu(self.conv2_0(x, data.edge_index, data.edge_attr))
        x = F.relu(self.conv2_1(x, data.edge_index, data.edge_attr))
        x = self.conv3(x, data.edge_index, data.edge_attr)
        emb = self.pool(x, data.batch)
        x = emb.view(-1, self.emb_size)
        self.embedding = x.data
        x = self.fc(F.relu(x))
        return x

class NNemb(torch.nn.Module):
    def __init__(self, input_size, n_classes):
        super(NNemb, self).__init__()
        nn1 = nn.Sequential(nn.Linear(1,64), nn.ReLU(), nn.Linear(64,128))
        self.conv1 = NNConv(input_size,64,nn1)
        nn2 = nn.Sequential(nn.Linear(1,64), nn.ReLU(), nn.Linear(64,256))
        self.conv2 = NNConv(64,n_classes,nn2)

    def forward(self,x,edge_index,edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return x

class NNConvNet(torch.nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 n_classes,
                 batch_size=1,
                 pool_op='max',
                 same_size=False):
        super(NNConvNet, self).__init__()
        self.nnc = NNemb(input_size, embedding_size)
        self.fc = torch.nn.Linear(embedding_size, n_classes)
        if pool_op == 'max':
            self.pool = global_max_pool
        self.bs = batch_size
        self.emb_size = embedding_size
        self.same_size = same_size
        self.embedding = None

    def forward(self, gdata):
        x, edge_index, edge_attr, batch = gdata.x, gdata.edge_index, gdata.edge_attr, gdata.batch
        x = self.nnc(x,edge_index,edge_attr)
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


class DECSeq7(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, aggr='max', k=5, pool_op='max', same_size=False):
        super(DECSeq5, self).__init__()
        self.conv1 = EdgeConv(MLP([2 * 3, 64, 64, 64]), aggr)
        self.conv2 = EdgeConv(MLP([2 * 64, 128]), aggr)
        self.lin1 = MLP([128 + 64, 1024])
        if pool_op == 'max':
            self.pool = global_max_pool

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, n_classes))

    def forward(self, data):
        pos, batch, eidx = data.pos, data.batch, data.edge_index
        x1 = self.conv1(pos, eidx)
        x2 = self.conv2(x1, eidx)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = self.pool(out, batch)
        out = self.mlp(out)
        return out


class DECSeq6(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, fov=1, aggr='max', k=5, pool_op='max', same_size=False):
        super(DECSeq6, self).__init__()
        self.fov = fov
        self.bn0 = nn.BatchNorm1d(32)
        self.conv0 = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=fov),nn.ReLU())
        #self.bn0, nn.ReLU())
        self.conv1 = DynamicEdgeConv(MLP([2 * 32, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])
        if pool_op == 'max':
            self.pool = global_max_pool

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, n_classes))

    def forward(self, data):
        x, batch, eidx = data.pos, data.batch, data.edge_index
        batch_size = batch.max().item() + 1 if batch is not None else 1
        # inverting the labels in the second half of edgde_index
        # in order to account for the flipped streamlines (in the batch size)
        # eidx[:, :eidx.size(1) // 2] = eidx[:, eidx.size(1) // 2:].flip(1)
        eidx = eidx[:, :eidx.size(1) // 2]
        x = x.view(batch_size, -1, x.size(1))
        x = x.permute(0,2,1).contiguous()
        x = self.conv0(x)
        x0 = x.permute(0,2,1).contiguous().view(-1, x.size(1))
        batch = torch.arange(batch_size).repeat_interleave(data.lengths -
                                                  (self.fov - 1)).cuda()
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = self.pool(out, batch)
        out = self.mlp(out)
        return out



class DECSeq5(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, batch_size=1, k=4, aggr='max',pool_op='max', same_size=False):
        super(DECSeq5, self).__init__()
        self.k = k
        self.conv1 = EdgeConv(MLP([2 * 3, 64, 64, 64]), aggr)
        self.conv2 = EdgeConv(MLP([2 * 64, 128]), aggr)
        self.lin1 = MLP([128 + 64, 1024])
        if pool_op == 'max':
            self.pool = global_max_pool

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, n_classes))

    def forward(self, data):
        pos, batch, eidx = data.pos, data.batch, data.edge_index
        x1 = self.conv1(pos, eidx)
        x2 = self.conv2(x1, eidx)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = self.pool(out, batch)
        out = self.mlp(out)
        return out

class DECSeqSelf(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, aggr='max', k=5, pool_op='max', same_size=False):
        super(DECSeqSelf, self).__init__()
        self.conv1 = EdgeConv(MLP([2 * 3, 64, 64, 64]), aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])
        if pool_op == 'max':
            self.pool = global_max_pool

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, n_classes))

    def forward(self, data):
        pos, batch, eidx = data.pos, data.batch, data.edge_index
        eidx, _ = add_self_loops(eidx,num_nodes=pos.size(0))
        x1 = self.conv1(pos, eidx)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = self.pool(out, batch)
        out = self.mlp(out)
        return out

class DECSeq(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, dropout=True, k=5, aggr='max',pool_op='max'):
        super(DECSeq, self).__init__()
        # self.bn0 = BN(input_size)
        # self.bn1 = BN(64)
        # self.bn2 = BN(128)
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
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return out

class DECSeqAdj(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, dropout=True, k=5, aggr='max',pool_op='max'):
        super(DECSeq, self).__init__()
        # self.bn0 = BN(input_size)
        # self.bn1 = BN(64)
        # self.bn2 = BN(128)
        self.conv1 = EdgeConv(MLP([2 * 3, 64, 64, 64]), aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.mlp_adj = Seq(
                MLP([128 + 64, 64]), Lin(64, 1))
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
        x = torch.cat([x1, x2], dim=1)
        self.adj_emb = (self.mlp_adj(x))
        out = self.lin1(x)
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return out

class DECSeqGlob(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, dropout=True, k=5, aggr='max',pool_op='max', k_global=25):
        super(DECSeqGlob, self).__init__()
        self.k_global = k_global
        self.conv1 = EdgeConv(MLP([2 * 3, 64, 64, 64]), aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])
        if pool_op == 'max':
            self.pool = global_max_pool
        if dropout:
            self.mlp = Seq(
                MLP([1024, 512]), Dropout(0.5), MLP([512, 256]),
                Dropout(0.5), MLP([256, 32]))
        else:
            self.mlp = Seq(
                MLP([1024, 512]), MLP([512, 256]), MLP([256, 32]))
        self.lin = Lin(32, n_classes)
        # self.conv_glob = EdgeConv(MLP([2 * 32, 32]), aggr)
        self.conv_glob = GATConv(32, 32, heads=4, dropout=0.5, concat=False)

    def forward(self, data): #glob_gat.long().cuda()_simple
        pos, batch, eidx = data.pos, data.batch, data.edge_index
        x1 = self.conv1(pos, eidx)
        x2 = self.conv2(x1, batch)
        x = self.lin1(torch.cat([x1, x2], dim=1))
        x = global_max_pool(x, batch)
        x = self.mlp(x)

        # import ipdb; ipdb.set_trace()
        if self.training:
            glob_batch = data.ori_batch.clone()
        else:
            glob_batch = torch.zeros(x.size(0)).long().cuda()
        eidx_glob = knn_graph(pos.view(-1, data.lengths * 3),
                              self.k_global,
                              batch=glob_batch,
                              loop=True)
        x_glob = self.conv_glob(x, eidx_glob)
        out = self.lin(x_glob)
        return out

    # def forward(self, data): #glob_dec
    #     pos, batch, eidx = data.pos, data.batch, data.edge_index
    #     x1 = self.conv1(pos, eidx)
    #     x2 = self.conv2(x1, batch)
    #     x = self.lin1(torch.cat([x1, x2], dim=1))
    #     x = global_max_pool(x, batch)
    #     x = self.mlp(x)
    #     pseudo_pred = self.lin(x)
    #     # retreive streamlines with low confidence
    #     pseudo_pred = F.softmax(pseudo_pred, dim=-1)
    #     low_conf_idxs = torch.eq(pseudo_pred[:, 0] > 0.4,
    #                              pseudo_pred[:, 0] < 0.6).nonzero().squeeze()
    #     # import ipdb; ipdb.set_trace()
    #     if self.training:
    #         glob_batch = data.ori_batch.clone()
    #         # select max 1000 streamlines
    #         if low_conf_idxs.size(0) > 1000:
    #             print('exceeding the max low confidence streamlines %d' %
    #                   low_conf_idxs.size(0))
    #     #         idxs = torch.randint(low_conf_idxs.size(0),(1000,))
    #     #         low_conf_idxs = low_conf_idxs[idxs]
    #     #         # low_conf_x = torch.index_select(low_conf_x, 0,idxs)
    #     # glob_batch = glob_batch[low_conf_idxs]
    #     else:
    #         glob_batch = torch.zeros(x.size(0)).long().cuda()
    #     low_conf_x = x[low_conf_idxs].clone()
    #     eidx_glob = pyg_knn(x,
    #                     low_conf_x,
    #                     self.k_global + 1,
    #                     batch_x=glob_batch,
    #                     batch_y=glob_batch[low_conf_idxs])
    #     eidx_glob[0] = low_conf_idxs[eidx_glob[0]]
    #     x_glob = self.conv_glob(x, eidx_glob)
    #     x[low_conf_idxs] = x_glob[low_conf_idxs]

    #     out = self.lin(x)
    #     return out

    # def forward(self, data): # gat_glob
    #     pos, batch, eidx = data.pos, data.batch, data.edge_index
    #     x1 = self.conv1(pos, eidx)
    #     x2 = self.conv2(x1, batch)
    #     x = self.lin1(torch.cat([x1, x2], dim=1))
    #     x = global_max_pool(x, batch)
    #     x = self.mlp(x)
    #     pseudo_pred = self.lin(x)
    #     # retreive streamlines with low confidence
    #     pseudo_pred = F.softmax(pseudo_pred, dim=-1)
    #     low_conf_idxs = torch.eq(pseudo_pred[:, 0] > 0.4,
    #                              pseudo_pred[:, 0] < 0.6).nonzero().squeeze()
    #     # import ipdb; ipdb.set_trace()
    #     glob_batch = data.ori_batch.clone()
    #     if self.training:
    #         # select max 1000 streamlines
    #         if low_conf_idxs.size(0) > 1000:
    #             print('exceeding the max low confidence streamlines %d' %
    #                   low_conf_idxs.size(0))
    #     #         idxs = torch.randint(low_conf_idxs.size(0),(1000,))
    #     #         low_conf_idxs = low_conf_idxs[idxs]
    #     #         # low_conf_x = torch.index_select(low_conf_x, 0,idxs)
    #     # glob_batch = glob_batch[low_conf_idxs]
    #     low_conf_x = x[low_conf_idxs].clone()
    #     eidx_glob = pyg_knn(x,
    #                     low_conf_x,
    #                     self.k_global + 1,
    #                     batch_x=glob_batch,
    #                     batch_y=glob_batch[low_conf_idxs])
    #     eidx_glob[0] = low_conf_idxs[eidx_glob[0]]
    #     eidx_glob, _ = add_self_loops(eidx_glob, num_nodes=x.size(0))

    #     x_glob = self.conv_glob(x, eidx_glob)

    #     out = self.lin(x_glob)
    #     return out

class DECSeqKnn(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, dropout=True, k=5, aggr='max',pool_op='max', k_global=5):
        super(DECSeqKnn, self).__init__()
        self.k_global = k_global
        self.conv1 = EdgeConv(MLP([2 * 3, 64, 64, 64]), aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])
        if pool_op == 'max':
            self.pool = global_max_pool
        if dropout:
            self.mlp = Seq(
                MLP([1024, 512]), Dropout(0.5), MLP([512, 256]))
        else:
            self.mlp = Seq(
                MLP([1024, 512]), MLP([512, 256]))
        self.lin_global = Lin(256, k_global)
        self.lin2 = Lin(256, n_classes)

    def forward(self, data):
        pos, batch, eidx = data.pos, data.batch, data.edge_index
        x1 = self.conv1(pos, eidx)
        x2 = self.conv2(x1, batch)
        x = self.lin1(torch.cat([x1, x2], dim=1))
        x = global_max_pool(x, batch)
        x = self.mlp(x)
        out_knn = knn_graph(x, self.k_global+1, batch=None, loop=True)[0]
        # assuming k_global < min streamline length
        out_knn = x[out_knn.view(-1, self.k_global+1)].mean(1)
        # pseudo_class = F.log_softmax(out_knn)
        out = self.lin2(out_knn)
        return out

class DECSeqCos(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, aggr='max', k=5, pool_op='max', same_size=False):
        super(DECSeqCos, self).__init__()
        self.conv1 = EdgeConv(MLP([2 * 3, 64, 64, 64]), aggr)
        self.conv2 = DynamicEdgeConvCosine(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])
        if pool_op == 'max':
            self.pool = global_max_pool

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, n_classes))

    def forward(self, data):
        pos, batch, eidx = data.pos, data.batch, data.edge_index
        x1 = self.conv1(pos, eidx)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = self.pool(out, batch)
        out = self.mlp(out)
        return out

class DECSeq2(torch.nn.Module):

    def __init__(self,
                 input_size,
                 embedding_size,
                 n_classes,
                 fov=4,
                 k=5,
                 aggr='max',
                 bn=True):
        super(DECSeq2, self).__init__()
        pad = int(fov / 2)
        self.pad = pad
        self.bn1 = nn.BatchNorm1d(64)
        self.conv1 = nn.Sequential(
            nn.Conv1d(2 * input_size, 64, kernel_size=fov, padding=pad),
            self.bn1, nn.ReLU())
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = Seq(MLP([1024, 512]), Dropout(0.5), MLP([512, 256]),
                       Dropout(0.5), Lin(256, n_classes))

    def forward(self, data):
        x, batch, eidx = data.pos, data.batch, data.edge_index
        print('size eidx:',eidx.shape)
        print('size batch:',batch.shape)
        #print('eidx size:',eidx.shape)
        #print('size x:', x.shape)
        n_pts = x.size(0)
        batch_size = batch.max().item() + 1 if batch is not None else 1
        print('batch size:',batch_size)

        # inverting the labels in the second half of edgde_index
        # in order to account for the flipped streamlines
        eidx[:, :eidx.size(1) // 2] = eidx[:, eidx.size(1) // 2:].flip(1)
        # enlarged filter convolution
        x = torch.cat([x[eidx[1]] - x[eidx[0]], x[eidx[0]]], dim=1)
        x = x.view(batch_size * 2, -1, x.size(1))
        print('x after view:',x.shape)
        x = x.permute(0,2,1).contiguous()
        # after the prevoius steps the number of objects in x changed
        # from n_pts to n_edges*2.
        x = self.conv1(x)
        print('size x:',x.shape)
        # keep max between the two direction
        x = x.unsqueeze(0)
        x = torch.max(torch.cat([x[:, :batch_size],
                                 x[:, batch_size:].flip(1).flip(3)],
                                dim=0),
                      dim=0,
                      keepdim=False)[0]
        x1 = x.permute(0, 2, 1).contiguous().view(-1, x.size(1))
        print('size x:', x.shape)
        print('size x1:', x1.shape)
        # update the batch to refer to edges rather than points,
        # hence, delete one object from each batch
        batch = torch.arange(batch_size).repeat_interleave(data.lengths-1).cuda()
        print('size batch:',batch.shape)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return out


class DECSeq3(torch.nn.Module):

    def __init__(self,
                 input_size,
                 embedding_size,
                 n_classes,
                 fov=1,
                 k=5,
                 aggr='max',
                 bn=True):
        super(DECSeq3, self).__init__()
        pad = int(fov / 2)
        self.pad = pad
        self.bn1fw = self.bn1bw = nn.BatchNorm1d(64)
        self.conv1fw = nn.Sequential(
            nn.Conv1d(2 * input_size, 64, kernel_size=fov, padding=pad),
            self.bn1fw, nn.ReLU())
        self.conv1bw = nn.Sequential(
            nn.Conv1d(2 * input_size, 64, kernel_size=fov, padding=pad),
            self.bn1bw, nn.ReLU())
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = Seq(MLP([1024, 512]), Dropout(0.5), MLP([512, 256]),
                       Dropout(0.5), Lin(256, n_classes))

    def forward(self, data):
        x, batch, eidx = data.pos, data.batch, data.edge_index
        #print('eidx size:',eidx.shape)
        #print('size x:', x.shape)
        n_pts = x.size(0)
        batch_size = batch.max().item() + 1 if batch is not None else 1

        # inverting the labels in the second half of edgde_index
        # in order to account for the flipped streamlines
        eidx[:, :eidx.size(1) // 2] = eidx[:, eidx.size(1) // 2:].flip(1)
        # enlarged filter convolution
        x = torch.cat([x[eidx[1]] - x[eidx[0]], x[eidx[0]]], dim=1)
        x = x.view(batch_size * 2, -1, x.size(1))
        x = x.permute(0,2,1).contiguous()
        # after the prevoius steps the number of objects in x changed
        # from n_pts to n_edges*2.

        # one conv learns one direction and the other learn the opposite
        x_fw = self.conv1fw(x[:batch_size])
        x_bw = self.conv1bw(x[batch_size:])
        # the two embedded directions are summed up into a unique element
        x = x_fw + x_bw.flip(0).flip(2)

        x1 = x.permute(0, 2, 1).contiguous().view(-1, x.size(1))
        print('x1 size:',x1.shape)
        # update the batch to refer to edges rather than points,
        # hence, delete one object from each batch
        batch = torch.arange(batch_size).repeat_interleave(data.lengths-1).cuda()
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return out

class DGCNNSeq(nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, batch_size=1,k=5, fov=1, dropout=0.5):
        super(DGCNNSeq, self).__init__()
        self.bs = batch_size
        #self.fov = fov
        self.k = k
        self.input_size = input_size

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(embedding_size)

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=fov, bias=False), self.bn1,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(64 * 2, 64, kernel_size=fov, bias=False), self.bn2,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(64 * 2, 128, kernel_size=fov, bias=False), self.bn3,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv1d(128 * 2, 256, kernel_size=fov, bias=False), self.bn4,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, embedding_size, kernel_size=fov, bias=False),
            self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(embedding_size * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, n_classes)

    def forward(self, data):
        x = data.x.reshape(self.bs, -1, self.input_size)
        x = x.permute(0,2,1).contiguous()
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

def ST_loss(pn_model, gamma=0.001):
    A = pn_model.trans  # BxKxK
    A_t = A.transpose(2, 1).contiguous()
    AA_t = torch.bmm(A, A_t)
    I = torch.eye(A.shape[1]).cuda()  # KxK
    return gamma * (torch.norm(AA_t - I)**2)

class GAT(torch.nn.Module):
    def __init__(self, input_size, n_classes, k_graph=False):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_size, 64, heads=4, dropout=0.)
        self.conv2 = GATConv(256, 256, heads=3, dropout=0.)
        #self.conv3 = GATConv(1024, 256, heads=4, dropout=0., concat=False)
        self.pool = global_max_pool
        self.mlp = Seq(MLP([1024, 512]), Dropout(0.2), MLP([512, 256]),
                       Dropout(0.2), Lin(256, n_classes))
        #self.mlp = Lin(256,n_classes)
        self.k_graph = k_graph

    def forward(self, data):
        x = data.x
        if self.k_graph:
            data.edge_index = knn_graph(data.x, k_graph)
        x1 = F.elu(self.conv1(x, data.edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        x2 = self.conv2(x1, data.edge_index)
        #x = self.conv3(x, data.edge_index)
        x = self.pool(torch.cat([x1,x2],dim=1), data.batch)
        x = self.mlp(x)
        return x
