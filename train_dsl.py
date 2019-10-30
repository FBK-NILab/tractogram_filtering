import glob
import itertools
import os
import pickle
import sys
from time import time

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch_geometric.data import Batch as gBatch
from torch_geometric.data import DataListLoader as gDataLoader
from torchvision import transforms

import datasets as ds
#import lovasz_losses as L
from cluster_losses import CenterLoss, CSimLoss, DistLoss, GFLoss, HLoss
#from gcn import GCN, GCNbatch, GCNemb, GCNseg
#from pointnet import PNbatch, PointNetCls, PointNetDenseCls, ST_loss
#from pointnet_mgf import (OnlyFC, PointNetClsMultiFeat, PointNetDenseClsLocal,
                          #PointNetDenseClsMultiFeat,
                          #PointNetDenseClsMultiFeatMultiLayer, mean_mod)
from models import PNemb, PNbatch, PointNetPyg, GCNConvNet, NNConvNet, ST_loss
from tensorboardX import SummaryWriter
#from twounit_net import TwoUnitNet
from utils import get_spaced_colors
from visdom import Visdom

def get_model(cfg):

    num_classes = int(cfg['n_classes'])
    input_size = int(cfg['data_dim'])
    n_gf = int(cfg['num_gf'])
    
    if cfg['model'] == 'gcn_ori':
        classifier = GCN(input_size,
                                num_classes)
    elif cfg['model'] == 'pn_geom':
        classifier = PNbatch(input_size,
                                int(cfg['embedding_size']),
                                num_classes,
                                pool_op=torch.max,
                                batch_size=int(cfg['batch_size']),
                                same_size=cfg['same_size'])
    return classifier

def train_iter(cfg, dataloader, classifier, optimizer, writer, epoch, n_iter, cluster_loss_fn):

    num_classes = int(cfg['n_classes'])
    batch_size = int(cfg['batch_size'])
    n_epochs = int(cfg['n_epochs'])
    sample_size = int(cfg['fixed_size'])
    n_gf = int(cfg['num_gf'])
    input_size = int(cfg['data_dim'])
    num_batch = cfg['num_batch']
    alfa = float(cfg['alfa_loss'])

    ep_loss = 0.
    ep_seg_loss = 0.
    ep_cluster_loss = 0.
    mean_acc = torch.tensor([])
    mean_iou = torch.tensor([])
    mean_prec = torch.tensor([])
    mean_recall = torch.tensor([])
 
    for i_batch, sample_batched in enumerate(dataloader): 
        #print(i_batch[:2],sample_batched[:2])

        ### get batch
        if 'graph' not in cfg['dataset']:
            data_list = []
            name_list = []
            target_list = []
            for d in sample_batched:
              data_list.append(d['points'])
              name_list.append(d['name'])
              target_list.append(d['gt'])
              #points = sample_batched['points']
              #target = sample_batched['gt']
            points, target = Variable(data_list), Variable(target_list)
            #points, target = Variable(points), Variable(target)
            points, target = points.cuda(), target.cuda()

        else:
            #data_list = []
            #name_list = []
            #for d in sample_batched:
                #data_list.append(d['points'])
                #name_list.append(d['name'])
            #print(data_list)
            points = gBatch().from_data_list(sample_batched['points'])
            target = points['y']
            if cfg['same_size']:
                points['lengths'] = points['lengths'][0].item()
            sample_batched = {'points': points, 'gt': target, 'name': sample_batched['name']}
            points, target = points.to('cuda'), target.to('cuda')

        ### visualize embedding of the input
        if cfg['viz_emb_input'] and n_iter == 0:
            writer.add_embedding(points[0], metadata=target[0], global_step=n_iter)

        ### initialize gradients
        if not cfg['accumulation_interval'] or i_batch == 0:
            optimizer.zero_grad()

        ### state that the model will run in train mode
        classifier = classifier.train()

        ### forward
        if cfg['multi_loss']:
            logits, gf = classifier(points)
        else:
            logits = classifier(points)

        ### minimize the loss
        if len(cfg['loss']) == 2:
            if epoch <= int(cfg['switch_loss_epoch']):
                loss_type = cfg['loss'][0]
            else:
                loss_type = cfg['loss'][1]
        else:
            loss_type = cfg['loss'][0]

        if loss_type == 'nll':
            pred = F.log_softmax(logits, dim=-1)
            pred_choice = pred.data.max(1)[1].int()

            if cfg['nll_w']:
                ce_w = torch.tensor([1.5e-2] + [1.]*(num_classes-1)).cuda()
            else:
                ce_w = torch.tensor([1.]*num_classes).cuda()
            loss = F.nll_loss(pred, target.long(), weight=ce_w)
        elif loss_type == 'LLh':
            pred_choice = (logits.data>0).int()
            loss = L.lovasz_hinge(logits.view(batch_size, sample_size, 1),
                                target.view(batch_size, sample_size, 1),
                                per_image=False)
        elif loss_type == 'LLm':
            pred = F.softmax(logits, dim=-1)
            pred_choice = pred.data.max(1)[1].int()
            loss = L.lovasz_softmax_flat(pred, target, op=cfg['llm_op'],
                                        only_present=cfg['multi_category'])

        if cfg['spatial_tn'] and 'pointnet' in cfg['model']:
            loss = loss + ST_loss(classifier)

        if cfg['multi_loss']:
            loss_cluster = cluster_loss_fn(gf, classifier.feat.mf.f, classifier.feat.mf.p, sample_size, p=2)
            loss_seg = loss
            loss = loss_seg + alfa * loss_cluster
            print('loss_seg: %f\tloss_cluster: %f' %
                  (loss_seg, loss_cluster*alfa))
            ep_seg_loss += loss_seg
            ep_cluster_loss += loss_cluster

        ep_loss += loss
        
        if int(cfg['accumulation_interval']) % (i_batch+1) == 0:
            optimizer.step()
            optimizer.zero_grad
        elif not cfg['accumulation_interval']:
            optimizer.step()

        ### compute performance
        correct = pred_choice.eq(target.data.int()).sum()
        acc = correct.item()/float(target.size(0))
              
        tp = torch.mul(pred_choice.data, target.data.int()).sum().item()+0.00001
        fp = pred_choice.gt(target.data.int()).sum().item()
        fn = pred_choice.lt(target.data.int()).sum().item()
        tn = correct.item() - tp
        iou = float(tp)/(tp+fp+fn)
        prec = float(tp)/(tp+fp)
        recall = float(tp)/(tp+fn)

        print('[%d: %d/%d] train loss: %f acc: %f iou: %f' \
              % (epoch, i_batch, num_batch, loss.item(), acc, iou))

        mean_prec = torch.cat((mean_prec, torch.tensor([prec])), 0)
        mean_recall = torch.cat((mean_recall, torch.tensor([recall])), 0)
        mean_acc = torch.cat((mean_acc, torch.tensor([acc])), 0)
        mean_iou = torch.cat((mean_iou, torch.tensor([iou])), 0)
        n_iter += 1
       
    writer.add_scalar('train/epoch_loss', ep_loss / (i_batch+1), epoch)
    if cfg['multi_loss']:
        writer.add_scalar('train/epoch_seg_loss', ep_seg_loss / (i_batch+1), epoch)
        writer.add_scalar('train/epoch_cluster_loss', ep_cluster_loss / (i_batch+1), epoch)

    return mean_acc, mean_prec, mean_iou, mean_recall, ep_loss / (i_batch+1), n_iter

def val_iter(cfg, val_dataloader, classifier, optimizer, writer, epoch, cluster_loss_fn, best_epoch, best_pred, logdir):

    num_classes = int(cfg['n_classes'])
    batch_size = int(cfg['batch_size'])
    n_epochs = int(cfg['n_epochs'])
    sample_size = int(cfg['fixed_size'])
    n_gf = int(cfg['num_gf'])
    input_size = int(cfg['data_dim'])
    num_batch = cfg['num_batch']
    alfa = float(cfg['alfa_loss'])
    ep_loss = 0.

    with torch.no_grad():
        pred_buffer = {}
        sm_buffer = {}
        sm2_buffer = {}
        gf_buffer = {}
        print('\n\n')
        mean_val_acc = torch.tensor([])
        mean_val_iou = torch.tensor([])
        mean_val_prec = torch.tensor([])
        mean_val_recall = torch.tensor([])
        mean_val_iou_c = torch.tensor([])
        
        for j, data in enumerate(val_dataloader):
            if 'graph' in cfg['dataset']:
                points = data['points']
                target = data['gt']
                points, target = Variable(points), Variable(target)
                points, target = points.cuda(), target.cuda()
            else:
                data_list = []
                name_list = []
                for d in data:
                    data_list.append(d['points'])
                    name_list.append(d['name'])
                points = gBatch().from_data_list(data_list)
                target = points['y']
                if cfg['same_size']:
                    points['lengths'] = points['lengths'][0].item()
                data = {'points': points, 'gt': target, 'name': name_list}
                points, target = points.to('cuda'), target.to('cuda')
           
            sample_name = data['name'][0]
                       
            classifier = classifier.eval()
            if cfg['multi_loss']:
                logits, gf = classifier(points)
            else:
                logits = classifier(points)
            
            if len(cfg['loss']) == 2:
                if epoch <= int(cfg['switch_loss_epoch']):
                    loss_type = cfg['loss'][0]
                else:
                    loss_type = cfg['loss'][1]
            else:
                loss_type = cfg['loss'][0]

            if loss_type == 'nll':
                pred = F.log_softmax(logits, dim=-1)
                probas = torch.exp(pred.data)
                pred_choice = pred.data.max(1)[1].int()
                if cfg['nll_w']:
                    ce_w = torch.tensor([1.5e-2] + [1.]*(num_classes-1)).cuda()
                else:
                    ce_w = torch.tensor([1.]*num_classes).cuda()
                loss_seg = F.nll_loss(pred, target.long(), weight=ce_w)
            elif loss_type == 'LLh':
                pred_choice = (logits.data>0).int()
                loss_seg = L.lovasz_hinge(logits.view(batch_size, sample_size, 1),
                                    target.view(batch_size, sample_size, 1),
                                    per_image=False)
            elif loss_type == 'LLm':
                pred = F.softmax(logits, dim=-1)
                probas = pred.data
                pred_choice = pred.data.max(1)[1].int()
                loss_seg = L.lovasz_softmax_flat(pred, target,
                                op=cfg['llm_op'],
                                only_present=cfg['multi_category'])

            if cfg['multi_loss']:
                loss_cluster = 0.5
                loss = loss_seg + alfa * loss_cluster
            else:
                loss = loss_seg

            ep_loss += loss
            print('val max class pred ', pred_choice.max().item())
            print('val min class pred ', pred_choice.min().item())
            print('# class pred ', len(np.unique(pred_choice.cpu().numpy())))
            correct = pred_choice.eq(target.data.int()).cpu().sum()
            acc = correct.item()/float(target.size(0))
           
            tp = torch.mul(pred_choice.data, target.data.int()).cpu().sum().item()+0.00001
            fp = pred_choice.gt(target.data.int()).cpu().sum().item()
            fn = pred_choice.lt(target.data.int()).cpu().sum().item()
            tn = correct.item() - tp
            iou = torch.tensor([float(tp)/(tp+fp+fn)])
            prec = torch.tensor([float(tp)/(tp+fp)])
            recall = torch.tensor([float(tp)/(tp+fn)])

            print('VALIDATION [%d: %d/%d] val loss: %f acc: %f iou: %f' % (epoch, j, len(val_dataloader), loss, acc, iou))

            mean_val_prec = torch.cat((mean_val_prec, prec), 0)
            mean_val_recall = torch.cat((mean_val_recall, recall), 0)
            mean_val_iou = torch.cat((mean_val_iou, iou), 0)
            mean_val_acc = torch.cat((mean_val_acc, torch.tensor([acc])), 0)

            if cfg['save_pred']:
                sl_idx = np.where(pred_choice.data.cpu().view(-1).numpy() == 1)[0]
                pred_buffer[sample_name] = sl_idx.tolist()
            if cfg['save_softmax_out']:
                if cfg['model'] in 'pointnet_mgfml':
                    if sample_name not in sm_buffer.keys():
                        sm_buffer[sample_name] = []
                    if classifier.feat.multi_feat > 1:
                        sm_buffer[sample_name].append(
                            classifier.feat.mf.softmax_out.cpu().numpy())
                        
        macro_iou = torch.mean(mean_val_iou)
        macro_prec = torch.mean(mean_val_prec)
        macro_recall = torch.mean(mean_val_recall)
        macro_iou_c = torch.mean(mean_val_iou_c)

        epoch_iou = macro_iou.item()

        writer.add_scalar('val/epoch_acc',
                    torch.mean(mean_val_acc).item(), epoch)
        writer.add_scalar('val/epoch_iou',
                    epoch_iou, epoch)
        writer.add_scalar('val/epoch_prec',macro_prec.item(), epoch)
        writer.add_scalar('val/epoch_recall',macro_recall.item(), epoch)
        writer.add_scalar('val/epoch_iou_c',
                    macro_iou_c.item(), epoch)
        writer.add_scalar('val/loss', ep_loss / j, epoch)
        print('VALIDATION ACCURACY: %f' % torch.mean(mean_val_acc).item())
        print('VALIDATION IOU: %f' % epoch_iou)
        print('VALIDATION IOUC: %f' % macro_iou_c.item())
        print('\n\n')

        if epoch_iou > best_pred:
            best_pred = epoch_iou
            best_epoch = epoch 
            
            if cfg['save_model']: 
                modeldir = os.path.join(logdir, cfg['model_dir'])
                if not os.path.exists(modeldir):
                    os.makedirs(modeldir)
                else:
                    os.system('rm %s/best_model*.pth' % modeldir)
                torch.save(classifier.state_dict(), '%s/best_model_iou-%f_ep-%d.pth' % (modeldir, best_pred, epoch))
        return best_epoch, best_pred, ep_loss


def train(cfg):

    num_classes = int(cfg['n_classes'])
    batch_size = int(cfg['batch_size'])
    n_epochs = int(cfg['n_epochs'])
    sample_size = int(cfg['fixed_size'])
    cfg['loss'] = cfg['loss'].split(' ')
    n_gf = int(cfg['num_gf'])
    input_size = int(cfg['data_dim'])


    #### DATA LOADING
    trans_train = []
    trans_val = []
    if cfg['rnd_sampling']:
        trans_train.append(ds.RndSampling(sample_size,maintain_prop=False))
        trans_val.append(ds.RndSampling(sample_size,maintain_prop=False))
    if cfg['standardization']:
        trans_train.append(ds.SampleStandardization())
        trans_val.append(ds.SampleStandardization())

    if cfg['dataset'] == 'hcp20_graph':
      dataset = ds.HCP20Dataset(cfg['sub_list_train'],
                                cfg['dataset_dir'],
                                act=cfg['act'],
                                transform=transforms.Compose(trans_train))    
    elif cfg['dataset'] == 'left_ifof_ss_sl_graph':
        dataset = ds.LeftIFOFSupersetGraphDataset(cfg['sub_list_train'],
                                cfg['dataset_dir'],
                                transform=transforms.Compose(trans_train),
                                same_size=cfg['same_size'])
    elif cfg['dataset'] == 'tractseg_500k':
        dataset = ds.Tractseg500kDataset(cfg['sub_list_train'],
                                cfg['dataset_dir'],
                                act=cfg['act'],
                                #fold_size=int(cfg['fold_size']),
                                transform=transforms.Compose(trans_train))
    if 'graph' in cfg['dataset']:
        DL = gDataLoader
    else:
        DL = DataLoader

    dataloader = DL(dataset, batch_size=batch_size,
                                    shuffle=cfg['shuffling'],
                                    num_workers=int(cfg['n_workers']),
                                    pin_memory=True)

    print("Dataset %s loaded, found %d samples" % (cfg['dataset'], len(dataset)))
    if cfg['val_in_train']:
        if cfg['dataset'] == 'hcp20_graph':
            val_dataset = ds.HCP20Dataset(cfg['sub_list_val'], 
                                          cfg['val_dataset_dir'], 
                                          act=cfg['act'],
                                          transform=transforms.Compose(trans_val))
        elif cfg['dataset'] == 'tractseg_500k':
            val_dataset = ds.Tractseg500kDataset(
                                    cfg['sub_list_val'],
                                    cfg['val_dataset_dir'],
                                    act=cfg['act'],
                                    #fold_size=int(cfg['fold_size']),
                                    transform=transforms.Compose(trans_val))
        elif cfg['dataset'] == 'left_ifof_ss_sl_graph':
            val_dataset = ds.LeftIFOFSupersetGraphDataset(cfg['sub_list_val'],
                                cfg['dataset_dir'],
                                transform=transforms.Compose(trans_val),
                                same_size=cfg['same_size'])
        
        val_dataloader = DL(val_dataset, batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=True)
        print("Validation dataset loaded, found %d samples" % (len(val_dataset)))

    # summary for tensorboard
    if cfg['experiment_name'] != 'default':
        for ext in range(100):
            exp_name = cfg['experiment_name'] + '_%d' % ext
            logdir = 'runs/%s' % exp_name
            if not os.path.exists(logdir):
                writer = SummaryWriter(logdir=logdir)
                break
    else:
        writer = SummaryWriter()

    tb_log_name = glob.glob('%s/events*' % logdir)[0].rsplit('/',1)[1]
    tb_log_dir = 'tb_logs/%s' % exp_name
    os.system('mkdir -p %s' % tb_log_dir)
    os.system('ln -sr %s/%s %s/%s ' %
                        (logdir, tb_log_name, tb_log_dir, tb_log_name))

    os.system('cp main_dsl_config.py %s/config.txt' % (writer.logdir))

    #### BUILD THE MODEL
    classifier = get_model(cfg)
    
    #### SET THE TRAINING
    if cfg['optimizer'] == 'sgd_momentum':
        optimizer = optim.SGD(classifier.parameters(),
                        lr=float(cfg['learning_rate']),
                        momentum=float(cfg['momentum']),
                        weight_decay=float(cfg['weight_decay']))
    elif cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(classifier.parameters(),
                        lr=float(cfg['learning_rate']),
                        weight_decay=float(cfg['weight_decay']))

    if cfg['lr_type'] == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                            int(cfg['lr_ep_step']),
                                            gamma=float(cfg['lr_gamma']))
    elif cfg['lr_type'] == 'plateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                            mode='min',
                                            factor=float(cfg['lr_gamma']),
                                            patience=int(cfg['patience']),
                                            threshold=0.0001,
                                            min_lr=float(cfg['min_lr']))
    if cfg['loss'] == 'nll':
        loss_fn = F.nll_loss

    alfa = float(cfg['alfa_loss'])
    if cfg['cluster_loss'] == 'entropy':
        cluster_loss_fn = HLoss()
    elif cfg['cluster_loss'] == 'dist':
        cluster_loss_fn = DistLoss()
    elif cfg['cluster_loss'] == 'csim':
        cluster_loss_fn = CSimLoss()
    elif cfg['cluster_loss'] == 'gfopt':
        cluster_loss_fn = GFLoss()
    elif cfg['cluster_loss'] == 'center':
        cluster_loss_fn = CenterLoss()
    else:
        cluster_loss_fn = None


    classifier.cuda()
    num_batch = len(dataset)/batch_size
    print('num of batches per epoch: %d' % num_batch)
    cfg['num_batch'] = num_batch
    
    n_iter = 0
    best_pred = 0
    best_epoch = 0
    current_lr = float(cfg['learning_rate'])
    for epoch in range(n_epochs+1):

        # bn decay as in pointnet orig
        if cfg['bn_decay'] and epoch % int(cfg['bn_decay_step']) == 0:
            bn_momentum = float(cfg['bn_decay_init']) * float(
                cfg['bn_decay_gamma'])**(epoch / int(cfg['bn_decay_step']))
            bn_momentum = 1 - min(0.99, 1 - bn_momentum)
            print('updated bn momentum to %f' % bn_momentum)
            for module in classifier.modules():
                if type(module) == torch.nn.BatchNorm1d:
                    module.momentum = bn_momentum

        mean_acc, mean_prec, mean_iou, mean_recall, loss, n_iter = train_iter(
                                    cfg,
                                    dataloader,
                                    classifier,
                                    optimizer,
                                    writer,
                                    epoch,
                                    n_iter,
                                    cluster_loss_fn
                                    )

        ### validation during training
        if epoch % int(cfg['val_freq']) == 0 and cfg['val_in_train']:
            best_epoch, best_pred, loss_val = val_iter(cfg,
                    val_dataloader,
                    classifier,
                    optimizer,
                    writer,
                    epoch,
                    cluster_loss_fn,
                    best_epoch,
                    best_pred,
                    logdir
                    )
        if cfg['lr_type'] == 'step' and current_lr >= float(cfg['min_lr']):
            lr_scheduler.step()
        if cfg['lr_type'] == 'plateau':
            lr_scheduler.step(loss_val)
        for i, param_group in enumerate(optimizer.param_groups):
            current_lr = float(param_group['lr'])
        writer.add_scalar('train/lr', current_lr, epoch)

        ### logging
        writer.add_scalar('train/epoch_acc', torch.mean(mean_acc).item(), epoch)
        writer.add_scalar('train/epoch_iou', torch.mean(mean_iou).item(), epoch)
        writer.add_scalar('train/epoch_prec', torch.mean(mean_prec).item(), epoch)
        writer.add_scalar('train/epoch_recall', torch.mean(mean_recall).item(), epoch)

        if cfg['viz_learned_features'] and cfg['model'] >= 'pointnet' and (epoch % 100) == 0:
            if cfg['model'] == 'pointnet':
                globalfeat = classifier.feat.globalfeat.data
                print(globalfeat.max(0))
            else:
                global_features = classifier.feat.globalfeat.data.permute(0,2,1).clone()
                writer.add_embedding(global_features[0],
                    metadata=target.data[:sample_size], tag="gf", global_step=n_iter)

            class_features = classifier.feat.output_conv3.data.permute(0,2,1).clone()
            seg_features = classifier.output_conv3.data.permute(0,2,1).clone()
            print(seg_features.size())
            writer.add_embedding(class_features[0],
                    metadata=target.data[:sample_size], tag="class", global_step=n_iter)
            writer.add_embedding(seg_features[0],
                    metadata=target.data[:sample_size], tag="seg", global_step=n_iter)


    writer.close()

    if best_epoch != n_epochs:
        if cfg['save_model']:
            modeldir = os.path.join(logdir, cfg['model_dir'])
            torch.save(classifier.state_dict(), '%s/model_ep-%d.pth' % (modeldir, epoch))

        if cfg['save_pred']:
            pred_dir = writer.logdir + '/predictions_%d' % epoch
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            for filename, value in pred_buffer.iteritems():
                with open(os.path.join(pred_dir, filename) + '.pkl', 'wb') as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

        if cfg['save_softmax_out']:
            sm_dir = writer.logdir + '/sm_out_%d' % epoch
            if not os.path.exists(sm_dir):
                os.makedirs(sm_dir)
            for filename, value in sm_buffer.iteritems():
                with open(os.path.join(sm_dir, filename) + '_sm_1.pkl', 'wb') as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            for filename, value in sm2_buffer.iteritems():
                with open(os.path.join(sm_dir, filename) + '_sm_2.pkl', 'wb') as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

        if cfg['save_gf']:
            gf_dir = writer.logdir + '/gf_%d' % epoch
            if not os.path.exists(gf_dir):
                os.makedirs(gf_dir)
            i = 0
            for filename, value in gf_buffer.iteritems():
                if i == 3:
                    break
                i += 1
                with open(os.path.join(gf_dir, filename) + '.pkl', 'wb') as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()
