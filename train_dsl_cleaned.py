import glob
import itertools
import os
import pickle
import sys
from time import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from torch_geometric.data import Batch as gBatch
from torch_geometric.data import DataListLoader as gDataLoader
from torch_geometric.nn import global_max_pool
from torchvision import transforms

import datasets as ds
from models import (DEC, NNC, BiLSTM, DECSeq, DECSeqCos, DECSeqSelf, DECSeq2, DECSeq3,
                    DECSeq5, DECSeq6, DGCNNSeq, GCNConvNet, GCNemb, NNConvNet,
                    NNemb, PNbatch, PNemb, PNptg, PNptg2, PointNetPyg, ST_loss)
from tensorboardX import SummaryWriter


def count_parameters(model):
    print([p.size() for p in model.parameters()])
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(cfg):

    num_classes = int(cfg['n_classes'])
    input_size = int(cfg['data_dim'])

    if cfg['model'] == 'blstm':
        classifier = BiLSTM(input_size,
                            n_classes=num_classes,
                            embedding_size=128,
                            hidden_size=256)
    if cfg['model'] == 'dec_ori':
        classifier = DGCNNSeq(input_size,
                              int(cfg['embedding_size']),
                              num_classes,
                              batch_size=int(cfg['batch_size']),
                              k=5,
                              fov=1,
                              dropout=0.5)
    if cfg['model'] == 'dec':
        classifier = DECSeq(
            input_size,
            int(cfg['embedding_size']),
            num_classes,
            #fov=3,
            batch_size=int(cfg['batch_size']),
            k=int(cfg['k_dec']),
            aggr='max',
            pool_op=global_max_pool,
            same_size=cfg['same_size'])
        #bn=True)
    if cfg['model'] == 'nnc':
        classifier = NNC(input_size,
                         int(cfg['embedding_size']),
                         num_classes,
                         batch_size=int(cfg['batch_size']),
                         pool_op=global_max_pool,
                         same_size=cfg['same_size'])
    if cfg['model'] == 'gcn':
        classifier = GCNConvNet(input_size,
                                int(cfg['embedding_size']),
                                num_classes,
                                batch_size=int(cfg['batch_size']),
                                pool_op=global_max_pool,
                                same_size=cfg['same_size'])
    elif cfg['model'] == 'pn_geom':
        classifier = PNptg2(input_size,
                           int(cfg['embedding_size']),
                           num_classes,
                           batch_size=int(cfg['batch_size']),
                           same_size=cfg['same_size'])
    return classifier


def train_ep(cfg, dataloader, classifier, optimizer, writer, epoch, n_iter):
    '''
    run one epoch of training
    '''

    # set the classifier in train mode
    classifier.train()

    num_classes = int(cfg['n_classes'])
    num_batch = cfg['num_batch']

    ep_loss = 0.
    metrics = initialize_metrics()

    for i_batch, sample_batched in enumerate(dataloader):

        ### reorganize the batch in term of streamlines
        points = get_gbatch_sample(sample_batched, int(cfg['fixed_size']),
                                   cfg['same_size'])
        target = points['y']

        points, target = points.to('cuda'), target.to('cuda')

        ### initialize gradients
        if not cfg['accumulation_interval'] or i_batch == 0:
            optimizer.zero_grad()

        ### forward
        logits = classifier(points)
        ### minimize the loss

        pred = F.log_softmax(logits, dim=-1).view(-1, num_classes)
        pred_choice = pred.data.max(1)[1].int()

        loss = F.nll_loss(pred, target.long())

        ep_loss += loss.item()

        if cfg['print_bwgraph']:
            print_net_graph(classifier, loss, writer.logdir)

        loss.backward()

        if int(cfg['accumulation_interval']) % (i_batch + 1) == 0:
            optimizer.step()
            optimizer.zero_grad
        elif not cfg['accumulation_interval']:
            optimizer.step()

        ### compute performance
        update_metrics(metrics, pred_choice, target)

        print('[%d: %d/%d] train loss: %f acc: %f' \
            % (epoch, i_batch, num_batch, loss.item(), metrics['acc'][-1]))

        n_iter += 1

    ep_loss = ep_loss / (i_batch + 1)
    writer.add_scalar('train/epoch_loss', ep_loss, epoch)

    log_avg_metrics(writer, metrics, 'train', epoch)

    return ep_loss, n_iter


def val_ep(cfg, val_dataloader, classifier, writer, epoch, best_epoch,
           best_score):
    '''
    run the validation phase when called
    '''
    best = False
    num_classes = int(cfg['n_classes'])

    # set classifier in eval mode
    classifier.eval()

    with torch.no_grad():
        print('\n\n')

        metrics_val = initialize_metrics()
        ep_loss = 0.

        for i, data in enumerate(val_dataloader):
            points = get_gbatch_sample(data, int(cfg['fixed_size']),
                                       cfg['same_size'])
            target = points['y']

            points, target = points.to('cuda'), target.to('cuda')

            ### forward
            logits = classifier(points)

            pred = F.log_softmax(logits, dim=-1).view(-1, num_classes)
            pred_choice = pred.data.max(1)[1].int()

            loss = F.nll_loss(pred, target.long())
            ep_loss += loss.item()

            print('val min / max class pred %d / %d' %
                  (pred_choice.min().item(), pred_choice.max().item()))
            print('# class pred ', len(torch.unique(pred_choice)))

            ### compute performance
            update_metrics(metrics_val, pred_choice, target)

            print('VALIDATION [%d: %d/%d] val loss: %f acc: %f' %
                  ((epoch, i, len(val_dataloader), loss.item(),
                    metrics_val['acc'][-1])))

        writer.add_scalar('val/loss', ep_loss / i, epoch)
        log_avg_metrics(writer, metrics_val, 'val', epoch)
        epoch_score = torch.tensor(metrics_val['acc']).mean().item()
        print('VALIDATION ACCURACY: %f' % epoch_score)
        print('\n\n')

        if epoch_score > best_score:
            best_score = epoch_score
            best_epoch = epoch
            best = True

        if cfg['save_model']:
            dump_model(cfg,
                       classifier,
                       writer.logdir,
                       epoch,
                       epoch_score,
                       best=best)

        return best_epoch, best_score


def train(cfg):

    batch_size = int(cfg['batch_size'])
    n_epochs = int(cfg['n_epochs'])
    sample_size = int(cfg['fixed_size'])
    cfg['loss'] = cfg['loss'].split(' ')

    #### DATA LOADING
    trans_train = []
    trans_val = []
    if cfg['rnd_sampling']:
        trans_train.append(ds.RndSampling(sample_size, maintain_prop=False))
        trans_val.append(ds.RndSampling(sample_size, maintain_prop=False))

    dataset, dataloader = get_dataset(cfg, trans=trans_train)
    val_dataset, val_dataloader = get_dataset(cfg, trans=trans_val, train=False)

    # summary for tensorboard
    writer = create_tb_logger(cfg)

    #### BUILD THE MODEL
    classifier = get_model(cfg)

    #### SET THE TRAINING
    optimizer = get_optimizer(cfg, classifier)

    lr_scheduler = get_lr_scheduler(cfg, optimizer)

    classifier.cuda()

    num_batch = len(dataset) / batch_size
    print('num of batches per epoch: %d' % num_batch)
    cfg['num_batch'] = num_batch

    n_iter = 0
    best_pred = 0
    best_epoch = 0
    current_lr = float(cfg['learning_rate'])
    for epoch in range(n_epochs + 1):

        # update bn decay
        if cfg['bn_decay'] and epoch != 0 and epoch % int(
                cfg['bn_decay_step']) == 0:
            update_bn_decay(cfg, classifier, epoch)

        loss, n_iter = train_ep(cfg, dataloader, classifier, optimizer, writer,
                                epoch, n_iter)

        ### validation during training
        if epoch % int(cfg['val_freq']) == 0 and cfg['val_in_train']:
            best_epoch, best_pred = val_ep(cfg, val_dataloader, classifier,
                                           writer, epoch, best_epoch, best_pred)

        # update lr
        if cfg['lr_type'] == 'step' and current_lr >= float(cfg['min_lr']):
            lr_scheduler.step()
        if cfg['lr_type'] == 'plateau':
            lr_scheduler.step(loss)

        current_lr = get_lr(optimizer)
        writer.add_scalar('train/lr', current_lr, epoch)

    writer.close()


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()


def create_tb_logger(cfg):
    if cfg['experiment_name'] != 'default':
        for ext in range(100):
            exp_name = cfg['experiment_name'] + '_%d' % ext
            logdir = 'runs/%s' % exp_name
            if not os.path.exists(logdir):
                writer = SummaryWriter(logdir=logdir)
                break
        else:
            writer = SummaryWriter()

    tb_log_name = glob.glob('%s/events*' % logdir)[0].rsplit('/', 1)[1]
    tb_log_dir = 'tb_logs/%s' % exp_name
    os.system('mkdir -p %s' % tb_log_dir)
    os.system('ln -sr %s/%s %s/%s ' %
              (logdir, tb_log_name, tb_log_dir, tb_log_name))

    os.system('cp main_dsl_config.py %s/config.txt' % (writer.logdir))

    return writer


def get_dataset(cfg, trans, train=True):
    if not train:
        sub_list = cfg['sub_list_val']
        batch_size = 1
        shuffling = False
    else:
        sub_list = cfg['sub_list_train']
        batch_size = int(cfg['batch_size'])
        shuffling = cfg['shuffling']

    dataset = ds.HCP20Dataset(sub_list,
                              cfg['dataset_dir'],
                              transform=transforms.Compose(trans),
                              return_edges=cfg['return_edges'],
                              load_one_full_subj=False)

    dataloader = gDataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffling,
                             num_workers=int(cfg['n_workers']),
                             pin_memory=True)

    print("Dataset %s loaded, found %d samples" %
          (cfg['dataset'], len(dataset)))
    return dataset, dataloader


def get_optimizer(cfg, classifier):
    if cfg['optimizer'] == 'sgd_momentum':
        return optim.SGD(classifier.parameters(),
                         lr=float(cfg['learning_rate']),
                         momentum=float(cfg['momentum']),
                         weight_decay=float(cfg['weight_decay']))
    elif cfg['optimizer'] == 'adam':
        return optim.Adam(classifier.parameters(),
                          lr=float(cfg['learning_rate']),
                          weight_decay=float(cfg['weight_decay']))
    else:
        sys.exit('wrong or uknown optimizer')


def get_lr_scheduler(cfg, optimizer):
    if cfg['lr_type'] == 'step':
        return optim.lr_scheduler.StepLR(optimizer,
                                         int(cfg['lr_ep_step']),
                                         gamma=float(cfg['lr_gamma']))
    elif cfg['lr_type'] == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(cfg['lr_gamma']),
            patience=int(cfg['patience']),
            threshold=0.0001,
            min_lr=float(cfg['min_lr']))
    else:
        return None


def get_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        return float(param_group['lr'])


def update_bn_decay(cfg, classifier, epoch):
    # inspired by pointnet charlesq34 implementation
    bnd_0 = float(cfg['bn_decay_init'])
    bnd_gamma = float(cfg['bn_decay_gamma'])
    bnd_step = int(cfg['bn_decay_step'])

    bn_momentum = bnd_0 * bnd_gamma**(epoch / bnd_step)
    bn_momentum = 1 - min(0.99, 1 - bn_momentum)
    print('updated bn momentum to %f' % bn_momentum)
    for module in classifier.modules():
        if type(module) == torch.nn.BatchNorm1d:
            module.momentum = bn_momentum


def get_gbatch_sample(sample, sample_size, same_size, return_name=False):
    data_list = []
    name_list = []
    for i, d in enumerate(sample):
        if 'bvec' in d['points'].keys:
            d['points'].bvec += sample_size * i
        data_list.append(d['points'])
        name_list.append(d['name'])
    points = gBatch().from_data_list(data_list)
    if 'bvec' in points.keys:
        #points.batch = points.bvec.copy()
        points.batch = points.bvec.clone()
        del points.bvec
    if same_size:
        points['lengths'] = points['lengths'][0].item()

    if return_name:
        return points, name_list
    return points


def print_net_graph(classifier, loss, logdir):
    from torchviz import make_dot, make_dot_from_trace
    g = make_dot(loss, params=dict(classifier.named_parameters()))
    g.view('net_bw_graph')

    print('classifier parameters: %d' % int(count_parameters(classifier)))
    os.system('rm -r runs/%s' % logdir.split('/', 1)[1])
    os.system('rm -r tb_logs/%s' % logdir.split('/', 1)[1])
    sys.exit()


def initialize_metrics():
    metrics = {}
    metrics['acc'] = []
    metrics['iou'] = []
    metrics['prec'] = []
    metrics['recall'] = []

    return metrics


def update_metrics(metrics, prediction, target):
    prediction = prediction.data.int().cpu()
    target = target.data.int().cpu()

    correct = prediction.eq(target).sum().item()
    acc = correct / float(target.size(0))

    tp = torch.mul(prediction, target).sum().item() + 0.00001
    fp = prediction.gt(target).sum().item()
    fn = prediction.lt(target).sum().item()
    tn = correct - tp

    iou = float(tp) / (tp + fp + fn)
    prec = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)

    metrics['prec'].append(prec)
    metrics['recall'].append(recall)
    metrics['acc'].append(acc)
    metrics['iou'].append(iou)


def log_avg_metrics(writer, metrics, prefix, epoch):
    for k, v in metrics.items():
        if type(v) == list:
            v = torch.tensor(v)
        writer.add_scalar('%s/epoch_%s' % (prefix, k), v.mean().item(), epoch)


def dump_model(cfg, model, logdir, epoch, score, best=False):
    prefix = ''
    if best:
        prefix = 'best_'

    modeldir = os.path.join(logdir, cfg['model_dir'])
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    else:
        os.system('rm %s/%smodel*.pth' % (modeldir, prefix))
    torch.save(model.state_dict(),
               '%s/%smodel_ep-%d_score-%f.pth' %
                    (modeldir, prefix, epoch, score))
