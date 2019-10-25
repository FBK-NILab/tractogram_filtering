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
from models import PNbatch, PointNetCls, ST_loss
from tensorboardX import SummaryWriter
from twounit_net import TwoUnitNet
from utils import get_spaced_colors
from visdom import Visdom


def count_parameters(model):
    print([p.size() for p in model.parameters()])
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_3dscatter(points, labels, n_classes=None):
    plt.switch_backend('agg')

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    points -= points.min(axis=0)
    points = points/points.max()

    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]

    if n_classes is None:
        n_classes = len(np.unique(labels))
    colors = np.array(get_spaced_colors(n_classes))

    for cl in range(n_classes):
        i = np.where(labels == cl)
        if len(i[0]) == 0:
            continue
        ax.scatter(xs[i], ys[i], zs[i], c=colors[cl], s=50, label=cl)
    #ax.scatter(xs, ys, zs, c=colors[labels], s=50)

    plt.axis('off')
    plt.axis('equal')

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)

    ax.view_init(-30., 70.)
    ax.legend()

    #plt.show()

    #print('ax.azim {}'.format(ax.azim))
    #print('ax.elev {}'.format(ax.elev))

    return fig

def plot_heatmap(x, title):
    plt.switch_backend('agg')
    if type(x) != list:
        x = [x]
        title = [title]
    fig, axs = plt.subplots(ncols=len(x))

    if len(x) == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        im = ax.imshow(x[i].data.cpu().numpy(),
                    cmap='bwr',
                    interpolation='nearest',
                    aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.set_title(title[i])

    return fig

def get_model(cfg):

    num_classes = int(cfg['n_classes'])
    input_size = int(cfg['data_dim'])
    n_gf = int(cfg['num_gf'])

    if cfg['model'] == 'pointnet_cls':
        classifier = PointNetCls(input_size,
                                      cl=num_classes,
                                      gf_type=cfg['gf_operation'],
                                      bn=cfg['batch_norm'],
                                      simple=cfg['simple'],
                                      st=int(cfg['spatial_tn']),
                                      dropout=cfg['dropout'])
    elif cfg['model'] == 'pointnet':
        classifier = PointNetDenseCls(input_size,
                                      cl=num_classes,
                                      gf_type=cfg['gf_operation'],
                                      bn=cfg['batch_norm'],
                                      simple=cfg['simple'],
                                      st=int(cfg['spatial_tn']),
                                      multi_category=cfg['multi_category'])
    elif cfg['model'] == 'pointnet_local':
        classifier = PointNetDenseClsLocal(input_size,
                                            k=num_classes,
                                            bn=cfg['batch_norm'],
                                            simple=cfg['simple'])
    elif cfg['model'] == 'pointnet_mgf':
        classifier = PointNetDenseClsMultiFeat(input_size,
                                    k=num_classes,
                                    bn=cfg['batch_norm'],
                                    gf_type=cfg['gf_operation'],
                                    n_feat=n_gf,
                                    soft=cfg['soft_gf'],
                                    dist_gf=cfg['centroids_gf'],
                                    scaling=cfg['feat_scaling'],
                                    simple=cfg['simple'],
                                    simple_size = int(cfg['simple_size']),
                                    direct_clustering=cfg['direct_clustering'],
                                    multi_loss=cfg['multi_loss'],
                                    split_backprop=cfg['split_backprop'],
                                    bary=cfg['barycenter'],
                                    multi_category=cfg['multi_category'])
    elif cfg['model'] == 'pointnet_mgfml':
        classifier = PointNetDenseClsMultiFeatMultiLayer(input_size,
                                    k=num_classes,
                                    bn=cfg['batch_norm'],
                                    dropout=cfg['dropout'],
                                    gf_type=cfg['gf_operation'],
                                    n_feat=n_gf,
                                    dyn_k=cfg['dyn_k'],
                                    soft=cfg['soft_gf'],
                                    dist_gf=cfg['centroids_gf'],
                                    scaling=cfg['feat_scaling'],
                                    simple=cfg['simple'],
                                    simple_size = int(cfg['simple_size']),
                                    bary=cfg['barycenter'],
                                    direct_clustering=cfg['direct_clustering'],
                                    full_cat=cfg['full_concat'],
                                    multi_loss=cfg['multi_loss'],
                                    split_backprop=cfg['split_backprop'],
                                    num_layers=int(cfg['n_layers']),
                                    residual=cfg['residual'],
                                    multi_category=cfg['multi_category'],
                                    direct_pool=cfg['direct_pooling'])
    elif cfg['model'] == 'onlyFC':
        classifier = OnlyFC(k=num_classes)
    elif cfg['model'] == 'pointnet_mgf_cls':
        classifier = PointNetClsMultiFeat(input_size,
                                    k=num_classes,
                                    bn=cfg['batch_norm'],
                                    gf_type=cfg['gf_operation'],
                                    n_feat=n_gf,
                                    soft=cfg['soft_gf'],
                                    dist_gf=cfg['centroids_gf'],
                                    scaling=cfg['feat_scaling'],
                                    simple=cfg['simple'],
                                    simple_size = int(cfg['simple_size']),
                                    direct_clustering=cfg['direct_clustering'],
                                    multi_loss=cfg['multi_loss'],
                                    split_backprop=cfg['split_backprop'],
                                    bary=cfg['barycenter'],
                                    multi_category=cfg['multi_category'])
    elif cfg['model'] == 'gcn':
        classifier = GCNemb(input_size,
                                int(cfg['embedding_size']),
                                num_classes,
                                pool_op=torch.max,
                                batch_size=int(cfg['batch_size']),
                                same_size=cfg['same_size'])
    elif cfg['model'] == 'gcn_ori':
        classifier = GCN(input_size,
                                num_classes)
    elif cfg['model'] == 'pn_geom':
        classifier = PNbatch(input_size,
                                int(cfg['embedding_size']),
                                num_classes,
                                pool_op=torch.max,
                                batch_size=int(cfg['batch_size']),
                                same_size=cfg['same_size'])
    elif cfg['model'] == 'gcn_seg':
        classifier = GCNseg(input_size,
                                num_classes,
                                k=int(cfg['knngraph']),
                                batch_size=int(cfg['batch_size']))
    elif cfg['model'] == '2unit':
        classifier = TwoUnitNet(cfg,
                                input_size,
                                int(cfg['embedding_size']),
                                num_classes)
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
    #if epoch == 500:
    #    classifier.feat.mf.switch = True

    #t0 = time.time()
    for i_batch, sample_batched in enumerate(dataloader):

        ### get batch
        if 'graph' not in cfg['dataset']:
            points = sample_batched['points']
            target = sample_batched['gt']
            #if cfg['model'] == 'pointnet_cls':
            #points = points.view(batch_size*sample_size, -1, input_size)
            #target = target.view(batch_size*sample_size, -1)

            #batch_size = batch_size*sample_size
            #sample_size = points.shape[1]
            points, target = Variable(points), Variable(target)
            points, target = points.cuda(), target.cuda()

        else:
            data_list = []
            name_list = []
            for d in sample_batched:
                data_list.append(d['points'])
                name_list.append(d['name'])
            points = gBatch().from_data_list(data_list)
            target = points['y']
            if cfg['same_size']:
                points['lengths'] = points['lengths'][0].item()
            sample_batched = {'points': points, 'gt': target, 'name': name_list}


            #if (epoch != 0) and (epoch % 20 == 0):
            #    assert(len(dataloader.dataset) % int(cfg['fold_size']) == 0)
            #    folds = len(dataloader.dataset)/int(cfg['fold_size'])
            #    n_fold = (dataloader.dataset.n_fold + 1) % folds
            #    if n_fold != dataloader.dataset.n_fold:
            #        dataloader.dataset.n_fold = n_fold
            #        dataloader.dataset.load_fold()
            points, target = points.to('cuda'), target.to('cuda')

        #print('t loading batch: %f' % (time.time()-t0))

        ### add one-hot labels if multi-category task
        if cfg['multi_category']:
            one_hot_label = Variable(sample_batched['category'])
            classifier.category_vec = one_hot_label.cuda()

        ### visualize embedding of the input
        if cfg['viz_emb_input'] and n_iter == 0:
            writer.add_embedding(points[0], metadata=target[0], global_step=n_iter)

        ### initialize gradients
        if not cfg['accumulation_interval'] or i_batch == 0:
            optimizer.zero_grad()

        ### state that the model will run in train mode
        classifier = classifier.train()

        #torch.onnx.export(classifier, points.transpose(2,1), "model_pointnetmagf_max_1gf.onnx")
        #import hiddenlayer as hl
        #hl.build_graph(classifier, points.transpose(2,1))

        ### forward
        if cfg['multi_loss']:
            logits, gf = classifier(points)
        else:
            logits = classifier(points)

        ### flatten outputs
        if cfg['model'] != 'pointnet_mgf_cls':
            logits = logits.view(-1, num_classes)
        target = target.view(-1,1)[:,0]
        if cfg['verbose'] and i_batch == 0:
            #print("logits size: %s\ntarget size: %s" \
            #                            % (logits.size(), target.size()))
            norm_w_tot = torch.tensor(0.)
            norm_w_tot = norm_w_tot.cuda()
            for name, param in classifier.named_parameters():
                #writer.add_histogram(name, param, n_iter)
                norm_w_tot += torch.norm(param)
            writer.add_scalar('train/total_w_norm', norm_w_tot, n_iter)
            l2_reg = float(cfg['weight_decay']) * norm_w_tot

            #print('total norm: %f' % norm_tot.item())
            #print('L2 regularization term: %f' % l2_reg.item())
            # for name, param in classifier.named_parameters():
            #     if name == 'conv5.weight':
            #         #writer.add_histogram(name, param, n_iter)
            #         print('last layer weights norm: %f', param.data.norm(p=2).item())

        ### visualize the prediction distribution
        if cfg['viz_logits_distribution']:
            dist_cl = []
            for cl in range(target.max().item()+1):
                dist_cl.append(logits.view(-1)[target==cl])


        if cfg['viz_clusters'] and cfg['viz_clusters'] in sample_batched['name'] and np.log2(epoch).is_integer():
            i = sample_batched['name'].index(cfg['viz_clusters'])
            fig = []
            fig.append(plot_3dscatter(points[i,:,:].cpu().numpy(),
                    classifier.feat.mf.softmax_out[i,:,:].max(1)[1].squeeze().int().cpu().numpy(),
                    n_gf))

            rand_pts = torch.randint(1000, (100,)).tolist()
            fig.append(plot_heatmap(
                [classifier.feat.mf.p_soft[i, rand_pts, :], classifier.feat.mf.p[i, rand_pts, :]],
                    ['P_soft','P']))
            fig.append(plot_heatmap(classifier.feat.mf.gf[i], 'GF'))
            fig.append(plot_heatmap(classifier.feat.mf.out[i].transpose(0,1)[rand_pts, :], 'GF_distributed'))
            writer.add_scalar('train/F_norm', classifier.feat.mf.f[i].norm(), epoch)

            writer.add_figure('latent clustering', fig, epoch)

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
            #if cfg['model'] == 'pointnet_mgf_cls':
            #    pred = pred.sum(1) / pred.size(1)
            pred_choice = pred.data.max(1)[1].int()

            if cfg['nll_w']:
                #CLASS_W = torch.tensor([1.]*2).cuda()
                ce_w = torch.tensor([1.5e-2] + [1.]*(num_classes-1)).cuda()
                #if epoch >= int(cfg['switch_loss_epoch']):
                #CLASS_W = torch.tensor([10. * (2** ((epoch-int(cfg['switch_epoch_loss']))/100)) ] + [1.]*67).cuda()
                #CLASS_W = torch.tensor([1.,1.]).cuda()# + [1.]*67).cuda()
            else:
                ce_w = torch.tensor([1.]*num_classes).cuda()
            loss = F.nll_loss(pred, target.long(), weight=ce_w)
        elif loss_type == 'LLh':
            pred_choice = (logits.data>0).int()
            loss = L.lovasz_hinge(logits.view(batch_size, sample_size, 1),
                                target.view(batch_size, sample_size, 1),
                                per_image=False)
            #loss = L.lovasz_hinge_flat(pred.view(-1), target.view(-1))
        elif loss_type == 'LLm':
            pred = F.softmax(logits, dim=-1)
            pred_choice = pred.data.max(1)[1].int()
            loss = L.lovasz_softmax_flat(pred, target, op=cfg['llm_op'],
                                        only_present=cfg['multi_category'])

        if cfg['spatial_tn'] and 'pointnet' in cfg['model']:
            loss = loss + ST_loss(classifier)

        if cfg['multi_loss']:
            # gf vs class loss
            #one_hot = torch.arange(0,num_classes).expand(target.size(0)/sample_size, sample_size,-1).cuda()
            #mask = (one_hot.int() == target.view(-1, sample_size, 1)).float()
            #C = mean_mod(classifier.feat.mf.f.unsqueeze(3) * mask.unsqueeze(2),
            #        1,
            #        keepdim=False)
            #loss_cluster = cluster_loss_fn(C.transpose(1,2), gf, mask)

            loss_cluster = cluster_loss_fn(gf, classifier.feat.mf.f, classifier.feat.mf.p, sample_size, p=2)
            #loss_cluster = torch.norm(gf)
            loss_seg = loss
            loss = loss_seg + alfa * loss_cluster
            print('loss_seg: %f\tloss_cluster: %f' %
                  (loss_seg, loss_cluster*alfa))
            ep_seg_loss += loss_seg
            ep_cluster_loss += loss_cluster

        ep_loss += loss

        if cfg['print_bwgraph']:
            #with torch.onnx.set_training(classifier, False):
            #    trace, _ = torch.jit.get_trace_graph(classifier, args=(points.transpose(2,1),))
            #g = make_dot_from_trace(trace)
            from torchviz import make_dot, make_dot_from_trace
            g = make_dot(loss,
                                    params=dict(classifier.named_parameters()))
            #   g = make_dot(loss,
            #                           params=None)
            g.view('pointnet_mgf')
            print('classifier parameters: %d' % int(count_parameters(classifier)))
            os.system('rm -r runs/%s' % writer.logdir.split('/',1)[1])
            os.system('rm -r tb_logs/%s' % writer.logdir.split('/',1)[1])
            import sys; sys.exit()
        #print('memory allocated in MB: ', torch.cuda.memory_allocated()/2**20)
        #import sys; sys.exit()
        loss.backward()

        ### `clip_grad` helps prevent the exploding gradient problem.
        if float(cfg['clip_gradients']) > 0:
            torch.nn.utils.clip_grad_value_(classifier.parameters(),
                                            float(cfg['clip_gradients']))
        # print gradient norm

        norm_grad_tot = torch.tensor(0.)
        norm_grad_tot = norm_grad_tot.cuda()
        if cfg['verbose'] and i_batch == 0:
            for name, param in classifier.named_parameters():
                if 'weight' in name and param.grad is not None:
                    writer.add_scalar('grad_norm/' + name, torch.norm(param.grad), n_iter)
                    norm_grad_tot += torch.norm(param.grad)
            writer.add_scalar('train/total_grad_norm', norm_grad_tot, n_iter)


        if int(cfg['accumulation_interval']) % (i_batch+1) == 0:
            optimizer.step()
            optimizer.zero_grad
        elif not cfg['accumulation_interval']:
            optimizer.step()

        # t0 = time()
        # print(time() - t0)

        ### compute performance
        #pred_choice = pred.data.round().type_as(target.data)
        correct = pred_choice.eq(target.data.int()).sum()
        acc = correct.item()/float(target.size(0))
        # TODO adapt computation of IoU for multiclass problems
        #print('max class pred ', pred_choice.max().item())
        #print('min class pred ', pred_choice.min().item())
        #print('correct')
        #print(correct.item())
        if num_classes > 2:
            iou, prec, recall = L.iou_multi(pred_choice.data.int().cpu().numpy(),
                            target.data.int().cpu().numpy(), num_classes)
            iou = iou.mean()
            prec  = prec.mean()
            recall  = recall.mean()
            if 'ignore_class' not in cfg.keys():
                cfg['ignore_class'] = -1
            iou_c, prec_c, recall_c = L.iou_multi(pred_choice.data.int().cpu().numpy(),
                            target.data.int().cpu().numpy(),
                            num_classes,
                            ignore=int(cfg['ignore_class']))
            iou_c = iou_c.mean()
            print('[%d: %d/%d] train loss: %f acc: %f iou: %f iouC: %f' \
                    % (epoch, i_batch, num_batch, loss.item(), acc, iou, iou_c))
        else:
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

        ### logging
        # for name, param in classifier.named_parameters():
        #    if name == 'conv1.weight':
        #        writer.add_histogram(name, param, n_iter)


        n_iter += 1
        #t0 = time.time()


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
        #t0 = time.time()
        for j, data in enumerate(val_dataloader):
            if 'graph' not in cfg['dataset']:
                points = data['points']
                target = data['gt']
                #if cfg['model'] == 'pointnet_cls':
                #    points = points.view(batch_size*sample_size, -1, input_size)
                #    target = target.view(batch_size*sample_size, -1)
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
                #if (epoch != 0) and (epoch % 20 == 0):
                #    classifier.n_fold += 1
                points, target = points.to('cuda'), target.to('cuda')
            #print('t load batch: %f' % (time.time()-t0))


            sample_name = data['name'][0]
            ### add one-hot labels if multi-category task
            if cfg['multi_category']:
                one_hot_label = Variable(data['category'])
                classifier.category_vec = one_hot_label.cuda()
            # visualize prediction
#                    sc = vis.scatter(points.data.squeeze()[:,:3],
#                                        target.data.squeeze()+1,
#                                        opts=dict(
#                                            xtickmin=-50,
#                                            xtickmax=50,
#                                            ytickmin=-50,
#                                            ytickmax=50),)
#                    sys.exit()
            classifier = classifier.eval()
            if cfg['multi_loss']:
                logits, gf = classifier(points)
            else:
                logits = classifier(points)
            if cfg['model'] != 'pointnet_mgf_cls':
                logits = logits.view(-1, num_classes)
            target = target.view(-1,1)[:,0]
            #loss = F.nll_loss(pred, target)
            ### TODO: refactoring val in training
            if len(cfg['loss']) == 2:
                if epoch <= int(cfg['switch_loss_epoch']):
                    loss_type = cfg['loss'][0]
                else:
                    loss_type = cfg['loss'][1]
            else:
                loss_type = cfg['loss'][0]

            if loss_type == 'nll':
                pred = F.log_softmax(logits, dim=-1)
                #if cfg['model'] == 'pointnet_mgf_cls':
                #    pred = pred.sum(1) / pred.size(1)
                probas = torch.exp(pred.data)
                pred_choice = pred.data.max(1)[1].int()
                if cfg['nll_w']:
                    #CLASS_W = torch.tensor([1.]*2).cuda()
                    ce_w = torch.tensor([1.5e-2] + [1.]*(num_classes-1)).cuda()
                    #if epoch >= int(cfg['switch_loss_epoch']):
                    #CLASS_W = torch.tensor([10. * (2** ((epoch-int(cfg['switch_epoch_loss']))/100)) ] + [1.]*67).cuda()
                    #CLASS_W = torch.tensor([1.,1.]).cuda()# + [1.]*67).cuda()
                else:
                    ce_w = torch.tensor([1.]*num_classes).cuda()
                loss_seg = F.nll_loss(pred, target.long(), weight=ce_w)
            elif loss_type == 'LLh':
                pred_choice = (logits.data>0).int()
                loss_seg = L.lovasz_hinge(logits.view(batch_size, sample_size, 1),
                                    target.view(batch_size, sample_size, 1),
                                    per_image=False)
                #loss = L.lovasz_hinge_flat(pred.view(-1), target.view(-1))
            elif loss_type == 'LLm':
                pred = F.softmax(logits, dim=-1)
                probas = pred.data
                pred_choice = pred.data.max(1)[1].int()
                loss_seg = L.lovasz_softmax_flat(pred, target,
                                op=cfg['llm_op'],
                                only_present=cfg['multi_category'])

            if cfg['multi_loss']:
                #loss_cluster = cluster_loss_fn(gf, classifier.feat.mf.f.data, classifier.feat.mf.p.data, sample_size, p='Inf')
                #one_hot = torch.arange(0,num_classes).expand(batch_size, sample_size,-1).cuda()
                #mask = (one_hot.int() == target.view(batch_size, sample_size, 1)).float()
                #C = mean_mod(classifier.feat.mf.f.unsqueeze(3) * mask.unsqueeze(2),
                #        1,
                #        keepdim=False)
                #loss_cluster = cluster_loss_fn(C.transpose(1,2), gf, mask)
                #loss_cluster = cluster_loss_fn(gf.squeeze(3))
                #loss_cluster = torch.norm(classifier.feat.mf.f - torch.bmm(
                #    classifier.feat.mf.p, gf))
                loss_cluster = 0.5
                loss = loss_seg + alfa * loss_cluster
            else:
                loss = loss_seg

            ep_loss += loss
            #pred_choice = torch.sigmoid(pred.view(-1,1)).data.round().type_as(target.data)

            print('val max class pred ', pred_choice.max().item())
            print('val min class pred ', pred_choice.min().item())
            print('# class pred ', len(np.unique(pred_choice.cpu().numpy())))
            correct = pred_choice.eq(target.data.int()).cpu().sum()
            acc = correct.item()/float(target.size(0))

            if num_classes > 2:
                iou, prec, recall = L.iou_multi(pred_choice.data.int().cpu().numpy(),
                                target.data.int().cpu().numpy(), num_classes,
                                multi_cat=cfg['multi_category'])
                assert(np.isnan(iou).sum() == 0)
                if cfg['multi_category']:
                    s, n_parts = data['gt_offset']
                    e = s + n_parts
                    iou[0,:s], prec[0,:s], recall[0,:s] = 0., 0., 0.
                    iou[0,e:], prec[0,e:], recall[0,e:] = 0., 0., 0.
                    iou = torch.from_numpy(iou).float()
                    prec = torch.from_numpy(prec).float()
                    recall = torch.from_numpy(recall).float()
                    category = data['category'].squeeze().nonzero().float()
                    iou = torch.cat([iou, category], 1)
                else:
                    iou  = torch.tensor([iou.mean()])
                    prec  = torch.tensor([prec.mean()])
                    recall  = torch.tensor([recall.mean()])
                    if 'ignore_class' not in cfg.keys():
                        cfg['ignore_class'] = -1
                    iou_c, prec_c, recall_c = L.iou_multi(pred_choice.data.int().cpu().numpy(),
                            target.data.int().cpu().numpy(),
                            num_classes,
                            ignore=int(cfg['ignore_class']))
                    iou_c = torch.tensor([iou_c.mean()])
                    mean_val_iou_c = torch.cat((mean_val_iou_c, iou_c), 0)
                    print('VALIDATION [%d: %d/%d] val loss: %f acc: %f iou: %f iouC: %f' \
                        % (epoch, j, len(val_dataloader), loss, acc, iou, iou_c))

                assert(torch.isnan(iou).sum() == 0)
            else:
                tp = torch.mul(pred_choice.data, target.data.int()).cpu().sum().item()+0.00001
                fp = pred_choice.gt(target.data.int()).cpu().sum().item()
                fn = pred_choice.lt(target.data.int()).cpu().sum().item()
                tn = correct.item() - tp
                iou = torch.tensor([float(tp)/(tp+fp+fn)])
                prec = torch.tensor([float(tp)/(tp+fp)])
                recall = torch.tensor([float(tp)/(tp+fn)])

                print('VALIDATION [%d: %d/%d] val loss: %f acc: %f iou: %f' \
                        % (epoch, j, len(val_dataloader), loss, acc, iou))

            mean_val_prec = torch.cat((mean_val_prec, prec), 0)
            mean_val_recall = torch.cat((mean_val_recall, recall), 0)
            mean_val_iou = torch.cat((mean_val_iou, iou), 0)
            mean_val_acc = torch.cat((mean_val_acc, torch.tensor([acc])), 0)

            if cfg['save_pred']:
                if num_classes > 2:
                    for cl in range(num_classes):
                        sl_idx = np.where(pred_choice.data.cpu().view(-1).numpy() == cl)[0]
                        if cl == 0:
                            pred_buffer[sample_name] = []
                        pred_buffer[sample_name].append(sl_idx.tolist())
                else:
                    sl_idx = np.where(pred_choice.data.cpu().view(-1).numpy() == 1)[0]
                    pred_buffer[sample_name] = sl_idx.tolist()
            if cfg['save_softmax_out']:
                if cfg['model'] in 'pointnet_mgfml':
                    if sample_name not in sm_buffer.keys():
                        sm_buffer[sample_name] = []
                    if classifier.feat.multi_feat > 1:
                        sm_buffer[sample_name].append(
                            classifier.feat.mf.softmax_out.cpu().numpy())
                if cfg['model'] == 'pointnet_mgfml':
                    for l in classifier.layers:
                        sm_buffer[sample_name].append(
                                l.mf.softmax_out.cpu().numpy())
                sm2_buffer[sample_name] = probas.cpu().numpy()
            if cfg['save_gf']:
                #   gf_buffer[sample_name] = np.unique(
                #           classifier.feat.globalfeat.data.cpu().squeeze().numpy(), axis = 0)
                gf_buffer[sample_name] = classifier.globalfeat

            #t0 = time.time()


        if cfg['multi_category']:
            macro_iou = torch.ones(num_classes) * -1
            macro_prec = torch.ones(num_classes) * -1
            macro_recall = torch.ones(num_classes) * -1
            micro_iou = torch.ones(int(cfg['multi_category'])) * -1
            micro_prec = torch.ones(int(cfg['multi_category'])) * -1
            micro_recall = torch.ones(int(cfg['multi_category'])) * -1
            s = 0
            for cat,p in enumerate(cfg['num_parts'].split()):
                idx = (mean_val_iou[:,-1] == cat).nonzero()
                if len(idx) == 0:
                    s += int(p)
                    continue
                multicat_iou = mean_val_iou[idx,s:s+int(p)].mean(0)
                multicat_prec = mean_val_prec[idx,s:s+int(p)].mean(0)
                multicat_recall = mean_val_recall[idx,s:s+int(p)].mean(0)
                micro_iou[cat] = multicat_iou.mean()
                micro_prec[cat] = multicat_prec.mean()
                micro_recall[cat] = multicat_recall.mean()
                macro_iou[s:s+int(p)] = multicat_iou
                macro_prec[s:s+int(p)] = multicat_prec
                macro_recall[s:s+int(p)] = multicat_recall
                s += int(p)
            macro_iou = macro_iou[macro_iou != -1.].mean()
            macro_prec = macro_prec[macro_prec != -1.].mean()
            macro_recall = macro_recall[macro_recall != -1.].mean()
            writer.add_text('MultiCategory macro-IoU', str(macro_iou.cpu().numpy().astype(np.float16)), epoch)
            writer.add_text('MultiCategory micro-IoU', str(micro_iou.cpu().numpy().astype(np.float16)), epoch)
            writer.add_text('MultiCategory micro-Precision', str(micro_prec.cpu().numpy().astype(np.float16)), epoch)
            writer.add_text('MultiCategory micro-Recall', str(micro_recall.cpu().numpy().astype(np.float16)), epoch)
        else:
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

            if cfg['save_pred']:
                os.system('rm -r %s/predictions_best*' % writer.logdir)
                pred_dir = writer.logdir + '/predictions_best_%d' % epoch
                if not os.path.exists(pred_dir):
                    os.makedirs(pred_dir)
                for filename, value in pred_buffer.iteritems():
                    with open(os.path.join(pred_dir, filename) + '.pkl', 'wb') as f:
                        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

            if cfg['save_softmax_out']:
                os.system('rm -r %s/sm_out_best*' % writer.logdir)
                sm_dir = writer.logdir + '/sm_out_best_%d' % epoch
                if not os.path.exists(sm_dir):
                    os.makedirs(sm_dir)
                for filename, value in sm_buffer.iteritems():
                    with open(os.path.join(sm_dir, filename) + '_sm_1.pkl', 'wb') as f:
                        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                for filename, value in sm2_buffer.iteritems():
                    with open(os.path.join(sm_dir, filename) + '_sm_2.pkl', 'wb') as f:
                        pickle.dump(value,
                                f,
                                protocol=pickle.HIGHEST_PROTOCOL)

            if cfg['save_gf']:
                os.system('rm -r %s/gf_best*' % writer.logdir)
                gf_dir = writer.logdir + '/gf_best_%d' % epoch
                if not os.path.exists(gf_dir):
                    os.makedirs(gf_dir)
                i = 0
                for filename, value in gf_buffer.iteritems():
                    if i == 3:
                        break
                    i += 1
                    with open(os.path.join(gf_dir, filename) + '.pkl', 'wb') as f:
                        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)


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
        #prop_vect = [np.random.uniform(low=0.1,high=0.6)]
        #prop_vect = [1./num_classes]*num_classes
        #trans_train.append(ds.RndSampling(sample_size,maintain_prop=False, prop_vector=prop_vect))
        #trans_val.append(ds.RndSampling(sample_size*batch_size,maintain_prop=False, prop_vector=prop_vect))
        #trans_val.append(ds.RndSampling(200000,maintain_prop=False, prop_vector=[np.random.uniform(low=0.1,high=0.6)]))
        #trans_val.append(ds.RndSampling(5000,maintain_prop=True))
        #trans_train.append(ds.RndSampling(sample_size,maintain_prop=True))
        #trans_val.append(ds.RndSampling(sample_size*batch_size,maintain_prop=True))
        #trans_val.append(ds.RndSampling(sample_size,maintain_prop=True))
        #trans_val.append(ds.RndSampling(sample_size*batch_size,maintain_prop=False))
        trans_train.append(ds.RndSampling(sample_size,maintain_prop=False))
        trans_val.append(ds.RndSampling(sample_size,maintain_prop=False))
    if cfg['standardization']:
        trans_train.append(ds.SampleStandardization())
        trans_val.append(ds.SampleStandardization())

    if cfg['dataset'] == 'left_ifof_ss_sl':
        dataset = ds.LeftIFOFSupersetDataset(cfg['sub_list_train'],
                                cfg['dataset_dir'],
                                transform=transforms.Compose(trans_train),
                                uniform_size=True)
    elif cfg['dataset'] == 'left_ifof_ss_sl_graph':
        dataset = ds.LeftIFOFSupersetGraphDataset(cfg['sub_list_train'],
                                cfg['dataset_dir'],
                                transform=transforms.Compose(trans_train),
                                same_size=cfg['same_size'])
    elif cfg['dataset'] == 'left_ifof_ss_dr':
        dataset = ds.DRLeftIFOFSupersetDataset(cfg['sub_list_train'],
                                cfg['dataset_dir'],
                                transform=transforms.Compose(trans_train),
                                load_all=cfg['load_all_once'])
    elif cfg['dataset'] == 'left_ifof_emb':
        dataset = ds.EmbDataset(cfg['sub_list_train'],
                                cfg['emb_dataset_dir'],
                                cfg['gt_dataset_dir'],
                                transform=transforms.Compose(trans_train),
                                load_all=cfg['load_all_once'],
                                precompute_graph=cfg['precompute_graph'],
                                k_graph=int(cfg['knngraph']))
    elif cfg['dataset'] == 'tractseg_500k_dr':
        dataset = ds.DRTractsegDataset(cfg['sub_list_train'],
                                cfg['dataset_dir'],
                                transform=transforms.Compose(trans_train),
                                load_all=cfg['load_all_once'])
    elif cfg['dataset'] == 'tractseg_500k':
        dataset = ds.Tractseg500kDataset(cfg['sub_list_train'],
                                cfg['dataset_dir'],
                                act=cfg['act'],
                                #fold_size=int(cfg['fold_size']),
                                transform=transforms.Compose(trans_train))
    elif cfg['dataset'] == 'psb_airplane':
        dataset = ds.PsbAirplaneDataset(cfg['dataset_dir'])
    elif cfg['dataset'] == 'shapes':
        dataset = ds.ShapesDataset(cfg['dataset_dir'],
                                transform=transforms.Compose(trans_train),
                                multi_cat=cfg['multi_category'])
    elif cfg['dataset'] == 'shapenet':
        dataset = ds.ShapeNetCore(cfg['dataset_dir'],
                                transform=transforms.Compose(trans_train),
                                multi_cat=cfg['multi_category'],
                                load_all=cfg['load_all_once'])
    elif cfg['dataset'] == 'modelnet':
        dataset = ds.ModelNetDataset(cfg['dataset_dir'],
                                    split='train_files',
                                    load_all=cfg['load_all_once'])
    elif cfg['dataset'] == 'scanobj':
        dataset = ds.ScanObjNNDataset(cfg['dataset_dir'],
                                        run='trainAll',
                                        variant=cfg['scanobj_variant'],
                                        background=cfg['scanobj_bg'],
                                        transform=transforms.Compose(trans_val),
                                        load_all=cfg['load_all_once'])

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
        if cfg['dataset'] == 'psb_airplane':
            val_dataset = ds.PsbAirplaneDataset(cfg['dataset_dir'], train=False)
        elif cfg['dataset'] == 'shapes':
            val_dataset = ds.ShapesDataset(cfg['dataset_dir'],
                                            train=False,
                                            multi_cat=cfg['multi_category'])
        elif cfg['dataset'] == 'shapenet':
            val_dataset = ds.ShapeNetCore(cfg['dataset_dir'],
                                            train=False,
                                            multi_cat=cfg['multi_category'],
                                            load_all=cfg['load_all_once'])
        elif cfg['dataset'] == 'modelnet':
            val_dataset = ds.ModelNetDataset(cfg['dataset_dir'],
                                            split='test_files',
                                            load_all=cfg['load_all_once'])
        elif cfg['dataset'] == 'tractseg_500k':
            val_dataset = ds.Tractseg500kDataset(
                                    cfg['sub_list_val'],
                                    cfg['val_dataset_dir'],
                                    act=cfg['act'],
                                    #fold_size=int(cfg['fold_size']),
                                    transform=transforms.Compose(trans_val))
        elif cfg['dataset'] == 'scanobj':
            val_dataset = ds.ScanObjNNDataset(cfg['dataset_dir'],
                                        run='test',
                                        variant=cfg['scanobj_variant'],
                                        background=cfg['scanobj_bg'],
                                        load_all=cfg['load_all_once'])
        elif cfg['dataset'] == 'tractseg_500k_dr':
            val_dataset = ds.DRTractsegDataset(
                                    cfg['sub_list_val'],
                                    cfg['val_dataset_dir'],
                                    transform=transforms.Compose(trans_val),
                                    load_all=cfg['load_all_once'])
        elif cfg['dataset'] == 'left_ifof_ss_sl':
            val_dataset = ds.LeftIFOFSupersetDataset(cfg['sub_list_val'],
                                cfg['val_dataset_dir'],
                                transform=transforms.Compose(trans_val))
        elif cfg['dataset'] == 'left_ifof_ss_sl_graph':
            val_dataset = ds.LeftIFOFSupersetGraphDataset(cfg['sub_list_val'],
                                cfg['dataset_dir'],
                                transform=transforms.Compose(trans_val),
                                same_size=cfg['same_size'])
        elif cfg['dataset'] == 'left_ifof_emb':
            val_dataset = ds.EmbDataset(cfg['sub_list_val'],
                                cfg['emb_dataset_dir'],
                                cfg['gt_dataset_dir'],
                                transform=transforms.Compose(trans_val),
                                load_all=cfg['load_all_once'],
                                precompute_graph=cfg['precompute_graph'],
                                k_graph=int(cfg['knngraph']))
        else:
            val_dataset = ds.DRLeftIFOFSupersetDataset(cfg['sub_list_val'],
                                            cfg['val_dataset_dir'],
                                            transform=transforms.Compose(trans_val),
                                            load_all=cfg['load_all_once'])
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

    # initialize visdom
    #vis = Visdom()


    #### BUILD THE MODEL
    classifier = get_model(cfg)
    #writer.add_graph(classifier)

    #classifier.apply(weight_init)

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


    #ipdb.set_trace()
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
                                    cluster_loss_fn,
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
