from __future__ import print_function
import os
import sys
import copy
import math
import numpy as np
#import pointnet_trials as pnt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch_geometric.transforms as T
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import DynamicEdgeConv, GCNConv, NNConv, graclus, EdgeConv, GATConv, SplineConv
#from pointnet_mgf import max_mod
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.utils import add_self_loops

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
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

class PNptg(torch.nn.Module):

    def __init__(self,
                 input_size,
                 embedding_size,
                 n_classes,
                 batch_size=1,
                 pool_op=global_max_pool,
                 same_size=False):
        super(PNptg, self).__init__()
        self.pn = PNemb(input_size, embedding_size)
        self.fc = torch.nn.Linear(embedding_size, n_classes)
        self.pool = pool_op
        self.bs = batch_size
        self.emb_size = embedding_size
        self.same_size = same_size
        self.embedding = None

    def forward(self, gdata):
        x, batch = gdata.x, gdata.batch
        x = self.pn(x)
        emb = self.pool(x,batch)
        x = emb.view(-1, self.emb_size)
        self.embedding = x.data
        x = self.fc(F.relu(x))
        return x

class PNbatch(torch.nn.Module):

    def __init__(self,
                 input_size,
                 embedding_size,
                 n_classes,
                 pool_op=torch.max,
                 batch_size=1,
                 same_size=False):
        super(PNbatch, self).__init__()
        self.pn = PNemb(input_size, embedding_size)
        self.fc = torch.nn.Linear(embedding_size, n_classes)
        self.pool = pool_op
        self.bs = batch_size
        self.emb_size = embedding_size
        self.same_size = same_size
        self.embedding = None

    def forward(self, gdata):
        x = gdata.x
        edges = gdata.edge_index
        lengths = gdata.lengths
        #t0 = time.time()
        x = self.pn(x)
        #print(x.shape)
        #print('t gcn: %f' % (time.time()-t0))
        #t0 = time.time()
        if not self.same_size:
            emb = torch.empty((len(lengths), self.emb_size)).cuda()
            for i, g in enumerate(torch.split(x, lengths.tolist(), dim=0)):
                desc = self.pool(g, 0, keepdim=True)
                if len(desc) == 2:
                    desc = desc[0]
                #emb = torch.cat((x, desc), 0)
                emb[i, :] = desc
        else:
            emb = self.pool(x.view(-1, lengths, self.emb_size), 1, keepdim=True)
            print(emb.shape)
            if len(emb) == 2:
                emb = emb[0]
        x = emb.view(-1, self.emb_size)
        #print(x.shape)
        self.embedding = x.data
        #print('t batch reshaping: %f' % (time.time()-t0))
        x = self.fc(F.relu(x))
        #print(x.shape)
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


class PointNetPyg(torch.nn.Module):

    def __init__(self,
                 input_size,
                 task='seg',
                 n_classes=2,
                 pool_op='max',
                 batch_size=1,
                 same_size=False):
        super(PointNetPyg, self).__init__()
        self.pn = PNemb(input_size, embedding_size)
        self.fc = torch.nn.Linear(embedding_size, n_classes)
        self.pool = pool_op
        self.bs = batch_size
        self.emb_size = embedding_size
        self.same_size = same_size
        self.embedding = None

    def forward(self, gdata):
        x = gdata.x
        edges = gdata.edge_index
        lengths = gdata.lengths
        #t0 = time.time()
        x = self.pn(x)
        #print('t gcn: %f' % (time.time()-t0))
        #t0 = time.time()
        if not self.same_size:
            emb = torch.empty((len(lengths), self.emb_size)).cuda()
            for i, g in enumerate(torch.split(x, lengths.tolist(), dim=0)):
                desc = self.pool(g, 0, keepdim=True)
                if len(desc) == 2:
                    desc = desc[0]
                #emb = torch.cat((x, desc), 0)
                emb[i, :] = desc
        else:
            emb = self.pool(x.view(-1, lengths, self.emb_size), 1, keepdim=True)
            if len(emb) == 2:
                emb = emb[0]
        x = emb.view(self.bs, -1, self.emb_size)
        self.embedding = x.data
        #print('t batch reshaping: %f' % (time.time()-t0))
        x = self.fc(F.relu(x))
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
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1_1(x, edge_index))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2_0(x, edge_index))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2_1(x, edge_index))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2_2(x, edge_index))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2_3(x, edge_index))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2_4(x, edge_index))
        #x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x

class GCNConvNet(torch.nn.Module):
    def __init__(self,
                input_size,
                embedding_size,
                n_classes,
                batch_size=1,
                pool_op=global_max_pool,
                same_size=False):
        super(GCNConvNet, self).__init__()
        self.gcn = GCNemb(input_size, embedding_size)
        self.fc = torch.nn.Linear(embedding_size, n_classes)
        self.pool = pool_op
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

class NNC1(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, batch_size=1, pool_op=global_max_pool, same_size=False):
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
        self.pool = pool_op
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

class NNC(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, batch_size=1, pool_op=global_max_pool, same_size=False):
        super(NNC, self).__init__()
        nn1 = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, input_size*32))
        self.conv1_0 = NNConv(input_size, 32, nn1, aggr='max')
        nn3 = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32,32*64))
        self.conv2_0 = NNConv(32,64, nn3, aggr='max')
        nn4 = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32,64*embedding_size))
        self.conv3 = NNConv(64, embedding_size, nn4, aggr='max')

        self.fc = torch.nn.Linear(embedding_size, n_classes)
        self.pool = pool_op
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
                 pool_op=global_max_pool,
                 same_size=False):
        super(NNConvNet, self).__init__()
        self.nnc = NNemb(input_size, embedding_size)
        self.fc = torch.nn.Linear(embedding_size, n_classes)
        self.pool = pool_op
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
    def __init__(self, input_size, embedding_size, n_classes, batch_size=1, k=5, aggr='max',pool_op=global_max_pool, same_size=False):
        super(DEC, self).__init__()
        self.k = k
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), self.k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), self.k, aggr)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, n_classes))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return out


class BiLSTM(torch.nn.Module):
    def __init__(self, input_size, n_classes=2, embedding_size=64, hidden_size=256):
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
    def __init__(self, input_size, embedding_size, n_classes, batch_size=1, k=5, aggr='max',pool_op=global_max_pool, same_size=False):
        super(DECSeq5, self).__init__()
        self.conv1 = EdgeConv(MLP([2 * 3, 64, 64, 64]), aggr)
        self.conv2 = EdgeConv(MLP([2 * 64, 128]), aggr)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, n_classes))

    def forward(self, data):
        pos, batch, eidx = data.pos, data.batch, data.edge_index
        x1 = self.conv1(pos, eidx)
        x2 = self.conv2(x1, eidx)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return out


class DECSeq6(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, fov=1, k=5, aggr='max',pool_op=global_max_pool, same_size=False):
        super(DECSeq6, self).__init__()
        self.fov = fov
        self.bn0 = nn.BatchNorm1d(32)
        self.conv0 = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=fov),nn.ReLU())
            #self.bn0, nn.ReLU())
        self.conv1 = DynamicEdgeConv(MLP([2 * 32, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])

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
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return out



class DECSeq5(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, batch_size=1, k=4, aggr='max',pool_op=global_max_pool, same_size=False):
        super(DECSeq5, self).__init__()
        self.k = k
        self.conv1 = EdgeConv(MLP([2 * 3, 64, 64, 64]), aggr)
        self.conv2 = EdgeConv(MLP([2 * 64, 128]), aggr)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, n_classes))

    def forward(self, data):
        pos, batch, eidx = data.pos, data.batch, data.edge_index
        x1 = self.conv1(pos, eidx)
        x2 = self.conv2(x1, eidx)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return out


class DECSeq(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, batch_size=1, k=5, aggr='max',pool_op=global_max_pool, same_size=False):
        super(DECSeq, self).__init__()
        self.k = k
        self.conv1 = EdgeConv(MLP([2 * 3, 64, 64, 64]), aggr)
        print('k:',self.k)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), self.k, aggr)
        print('new k:',self.k)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, n_classes))

    def forward(self, data):
        pos, batch, eidx = data.pos, data.batch, data.edge_index
        x1 = self.conv1(pos, eidx)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
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
                 pool_op=global_max_pool,
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
                 pool_op=global_max_pool,
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
