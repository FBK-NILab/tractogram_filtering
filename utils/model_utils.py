import os
import sys
import torch
import torchviz
from models import (MLP, DEC, BiLSTM, DECSeq, PNptg2)

def get_model(cfg):

    num_classes = int(cfg['n_classes'])
    input_size = int(cfg['data_dim'])

    if cfg['model'] == 'blstm':
        classifier = BiLSTM(input_size,
                            n_classes=num_classes,
                            embedding_size=128,
                            hidden_size=256,
                            dropout=cfg['dropout'])
    if cfg['model'] == 'sdec':
        classifier = DECSeq(
            input_size,
            int(cfg['embedding_size']),
            num_classes,
            #dropout=cfg['dropout'],
            k=int(cfg['k']),
            aggr='max',
            pool_op=cfg['pool_op'])
    if cfg['model'] == 'dec':
        classifier = DEC(
            input_size,
            int(cfg['embedding_size']),
            num_classes,
            #dropout=cfg['dropout'],
            k=int(cfg['k']),
            aggr='max',
            pool_op=cfg['pool_op'])
    elif cfg['model'] == 'pn_geom':
        classifier = PNptg2(input_size,
                           int(cfg['embedding_size']),
                           num_classes,
                           same_size=cfg['same_size'])
    return classifier

def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()

def count_parameters(model):
    print([p.size() for p in model.parameters()])
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_net_graph(classifier, loss, logdir):
    from torchviz import make_dot, make_dot_from_trace
    g = make_dot(loss, params=dict(classifier.named_parameters()))
    g.view('net_bw_graph')

    print('classifier parameters: %d' % int(count_parameters(classifier)))
    os.system('rm -r runs/%s' % logdir.split('/', 1)[1])
    os.system('rm -r tb_logs/%s' % logdir.split('/', 1)[1])
    sys.exit()
