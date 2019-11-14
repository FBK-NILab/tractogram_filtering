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
#from pointnet_mgf import max_mod

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
        self.conv1_0 = GCNConv(input_size, 64)
        self.conv1_1 = GCNConv(64, 64)
        self.conv2_0 = GCNConv(64, 64)
        self.conv2_1 = GCNConv(64, 128)
        self.conv2_2 = GCNConv(128,1024)
        self.conv2_3 = GCNConv(1024, 512)
        self.conv2_4 = GCNConv(512, 256)
        self.conv3 = GCNConv(256, n_classes)
        
    def forward(self, x, edge_index):
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
    
class NNC_working(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, batch_size=1, pool_op=global_max_pool, same_size=False):
        super(NNC, self).__init__()
        nn1 = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, input_size*embedding_size))
        self.conv = NNConv(input_size, embedding_size, nn1)
        self.fc = torch.nn.Linear(embedding_size, n_classes)
        self.pool = pool_op
        self.bs = batch_size
        self.emb_size = embedding_size
        self.same_size = same_size
        self.embedding = None
        
    def forward(self, data):
        print('data:',data)
        x = F.relu(self.conv(data.x, data.edge_index, data.edge_attr))
        print('shape of x:',x.shape)
        emb = self.pool(x, data.batch)
        x = emb.view(-1, self.emb_size)
        self.embedding = x.data
        x = self.fc(F.relu(x))
        return x        
    
class NNC(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, batch_size=1, pool_op=global_max_pool, same_size=False):
        super(NNC, self).__init__()
        nn1 = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, input_size*embedding_size))
        self.conv1_0 = NNConv(input_size, 64, nn1)
        nn2 = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64,64))
        self.conv1_1 = NNConv(64, 64, nn1)
        #self.conv2_0 = NNConv(64,64, nn1)
        #self.conv2_1 = NNConv(64, 128, nn1)
        #self.conv2_2 = NNConv(128, 1024, nn1)
        #self.conv2_3 = NNConv(1024, 512, nn1)
        #self.conv2_4 = NNConv(512, 256, nn1)
        #self.conv3 = NNConv(256, embedding_size)
        
        
        
        self.fc = torch.nn.Linear(embedding_size, n_classes)
        self.pool = pool_op
        self.bs = batch_size
        self.emb_size = embedding_size
        self.same_size = same_size
        self.embedding = None
        
    def forward(self, data):
        x = F.relu(self.conv1_0(data.x, data.edge_index, data.edge_attr))
        x = self.conv1_1(data.x, data.edge_index, data.edge_attr)
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
        
def ST_loss(pn_model, gamma=0.001):
    A = pn_model.trans  # BxKxK
    A_t = A.transpose(2, 1).contiguous()
    AA_t = torch.bmm(A, A_t)
    I = torch.eye(A.shape[1]).cuda()  # KxK
    return gamma * (torch.norm(AA_t - I)**2)
