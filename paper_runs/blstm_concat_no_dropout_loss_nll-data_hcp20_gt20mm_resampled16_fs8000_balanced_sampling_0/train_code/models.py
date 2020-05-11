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


class PNptg2(torch.nn.Module):

    def __init__(self,
                 input_size,
                 embedding_size,
                 n_classes,
                 pool_op='max',
                 same_size=False):
        super(PNptg2, self).__init__()
        self.fc_enc = MLP([input_size, 64, 64, 64, 128, 1024], batch_norm=True)
        if pool_op == 'max':
            self.pool = global_max_pool
        self.fc = MLP([1024, 512, 256, 128, embedding_size], batch_norm=True)
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
        channels = [input_size, 64, 64, 64, 128, 1024]
        self.convs = nn.ModuleList()
        batch_norm = True
        for i in range(1, len(channels)):
            if batch_norm:
                self.convs.append(
                    Seq(GCN(channels[i - 1], channels[i]), ReLU(),
                        BN(channels[i])))
            else:
                self.convs.append(
                    Seq(GCN(channels[i - 1], channels[i]), ReLU()))

        if pool_op == 'max':
            self.pool = global_max_pool
        self.mlp = MLP([1024, 512, 256, embedding_size])
        self.fc = torch.nn.Linear(embedding_size, n_classes)
        self.emb_size = embedding_size
        self.same_size = same_size
        self.embedding = None
        print(self)

    def forward(self, gdata):
        x, edge_index, batch = gdata.x, gdata.edge_index, gdata.batch
        edge_index, _ = add_self_loops(edge_index,num_nodes=x.size(0))
        data = gdata.clone()
        for gcn in self.convs:
            data.x = gcn(data)
        x = self.pool(data.x, batch)
        #x = emb.view(-1, self.emb_size)
        x = self.mlp(x)
        self.embedding = x.data
        x = self.fc(x)
        return x

class NNC(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, pool_op='max', same_size=False):
        super(NNC, self).__init__()
        nn1 = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, input_size*32))
        self.conv1_0 = NNConv(input_size, 32, nn1, aggr='max')
        nn3 = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32,32*64))
        self.conv2_0 = NNConv(32,64, nn3, aggr='max')
        nn4 = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32,64*embedding_size))
        self.conv3 = NNConv(64, embedding_size, nn4, aggr='max')

        self.fc = torch.nn.Linear(embedding_size, n_classes)
        if pool_op == 'max':
            self.pool = global_max_pool
        self.bs = batch_size
        self.emb_size = embedding_size
        self.same_size = same_size
        self.embedding = None

    def forward(self, data):
        #print('edge attr:',data.edge_attr)
        x = F.relu(self.conv1_0(data.x, data.edge_index, data.edge_attr))
        x = F.relu(self.conv2_0(x, data.edge_index, data.edge_attr))
        x = self.conv3(x, data.edge_index, data.edge_attr)
        emb = self.pool(x, data.batch)
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


class BiLSTM(torch.nn.Module):
    def __init__(self, input_size, n_classes=2, embedding_size=128, hidden_size=256, dropout=True):
        super(BiLSTM, self).__init__()
        self.emb_size = embedding_size
        self.h_size = hidden_size
        self.mlp = MLP([input_size, embedding_size])
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            bidirectional=True, batch_first=True)

        if dropout:
            self.lin = Seq(MLP([hidden_size*2, 128]), Dropout(0.5),
                           MLP([128, 40]), Dropout(0.5),
                           nn.Linear(40, n_classes))
        else:
            self.lin = Seq(MLP([hidden_size*2, 128]), MLP([128, 40]),
                           nn.Linear(40, n_classes))

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
        # emb = hn.sum(0)
        emb = torch.cat([hn[0], hn[1]], dim=1) 

        # classify
        x = self.lin(emb)
        return x


class ECnet(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, aggr='max', k=5, pool_op='max', same_size=False):
        super(ECnet, self).__init__()
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


class DECSeq(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, dropout=True, k=5, aggr='max',pool_op='max'):
        super(DECSeq, self).__init__()
        # self.bn0 = BN(input_size)
        # self.bn1 = BN(64)
        # self.bn2 = BN(128)
        self.conv1 = EdgeConv(MLP([2 * 3, 64, 64, 64], batch_norm=True), aggr)
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
        self.lin = Lin(256, n_classes)
        # self.conv_glob = EdgeConv(MLP([2 * 32, 32]), aggr)
        self.conv_glob = GATConv(32, 32, heads=8, dropout=0.5, concat=True)

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
        # eidx_glob = knn_graph(x,
                              self.k_global,
                            #   batch=glob_batch,
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
