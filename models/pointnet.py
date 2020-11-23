import copy
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch_geometric.transforms as T
from torch.autograd import Variable
from torch.nn import BatchNorm1d as BN
from torch.nn import Dropout
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn import global_max_pool

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