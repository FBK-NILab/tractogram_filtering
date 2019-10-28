from __future__ import print_function

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
from torch_geometric.nn import GCNConv, NNConv, graclus
from pointnet_mgf import max_mod


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
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
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
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


class STN3d(nn.Module):

    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(
            torch.from_numpy(
                np.array([1, 0, 0, 0, 1, 0, 0, 0,
                          1]).astype(np.float32))).view(1,
                                                        9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):

    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(
            torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(
                1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):

    def __init__(
            self,
            input_size,
            task='cls',
            spatial_tn=True,
            gf_size=1024,
            gf_op='max',
            bn=True,
            simple=False,
    ):

        super(PointNetfeat, self).__init__()

        if gf_op == 'max':
            self.gf_op = max_mod

        self.conv1 = torch.nn.Conv1d(input_size, 64, 1)
        self.bn1 = nn.BatchNorm1d(64) if bn else Identity()
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64) if bn else Identity()
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64) if bn else Identity()

        if spatial_tn > 0:
            self.stn = STN3d()
            if spatial_tn == 2:
                self.fstn = STNkd(k=64)

        if not simple:
            self.conv4 = torch.nn.Conv1d(64, 128, 1)
            self.conv5 = torch.nn.Conv1d(128, 1024, 1)
            self.bn4 = nn.BatchNorm1d(128) if bn else Identity()
            self.bn5 = nn.BatchNorm1d(1024) if bn else Identity()

        self.task = task
        self.spatial_tn = spatial_tn

        self.gf = None
        self.gf_size = gf_size
        self.simple = simple

    def forward(self, x):
        n_pts = x.size()[2]

        if self.spatial_tn > 0:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        if self.spatial_tn == 2:
            trans = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)

        pointfeat = x

        if not self.simple:
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))

        x = self.gf_op(x, 2, keepdim=True)
        x = x.view(-1, self.gf_size)
        if self.task == 'cls':
            return x, trans
        else:
            self.gf = x.view(-1).data.tolist()
            x = x.view(-1, self.gf_size, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans


class PointNetCls(nn.Module):

    def __init__(self,
                 input_size,
                 cl=2,
                 gf_type='max',
                 bn=True,
                 simple=False,
                 st=False,
                 dropout=True):
        super(PointNetCls, self).__init__()

        self.feat = PointNetfeat(input_size,
                                 task='cls',
                                 spatial_tn=st,
                                 gf_op=gf_type,
                                 bn=bn,
                                 simple=simple)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, cl)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp = dropout
        self.trans = None

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        if self.dp: x = F.dropout(x, p=0.3)
        x = F.relu(self.bn2(self.fc2(x)))
        if self.dp: x = F.dropout(x, p=0.3)
        x = self.fc3(x)
        self.trans = trans
        return x

class GCNConvNet(torch.nn.Module):
    def __init__(self,
                input_size,
                n_classes=2):
        super(GCNConvNet, self).__init__()
        self.conv1_0 = GCNConv(input_size, 64)
        self.conv1_1 = GCNConv(64, 64)
        self.conv2_0 = GCNConv(64, 64)
        self.conv2_1 = GCNConv(64, 128)
        self.conv2_2 = GCNConv(128, 1024)
        self.conv2_3 = GCNConv(1024, 512)
        self.conv2_4 = GCNConv(512, 256)
        self.conv3 = GCNConv(256, n_classes)
    
    def forward(self, gdata):
        x, edge_index = gdata.x, gdata.edge_index
        x = F.relu(self.conv1_0(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1_1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2_0(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2_1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2_2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2_3(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2_4(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x
        
        
class NNConvNet(torch.nn.Module):
    def __init__(self,
                 input_size,
                 n_classes=2)
        super(NNConvNet, self).__init__()
        nn1 = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(), nn.Linear(64, 64))
        self.conv1 = NNConv(input_size, 64, nn1, aggr='mean')
        
        nn2 = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(), nn.Linear(64, 10124))
        self.conv2 = NNConv(32, 64, nn2, aggr='mean')
        
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, n_classes)
        
    def forward(self, gdata)
        x, edge_index, edge_attr = gdata.x, gdata.edge_index, gdata.edge_attr
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x
        
def ST_loss(pn_model, gamma=0.001):
    A = pn_model.trans  # BxKxK
    A_t = A.transpose(2, 1).contiguous()
    AA_t = torch.bmm(A, A_t)
    I = torch.eye(A.shape[1]).cuda()  # KxK
    return gamma * (torch.norm(AA_t - I)**2)
