import sys
from time import time

import numpy as np
import torch
import torch.nn.functional as F

from utils.data.data_utils import (get_dataset, get_gbatch_sample,
                                   get_transforms)
from utils.general_utils import (initialize_metrics, log_avg_metrics,
                                 update_metrics)
from utils.model_utils import get_model, print_net_graph
from utils.train_utils import (compute_loss, create_tb_logger, dump_code,
                               dump_model, get_lr, get_lr_scheduler,
                               get_optimizer, initialize_loss_dict, log_losses,
                               update_bn_decay, set_lr)
from utils.data.transforms import RndSampling

def train_ep(cfg, dataloader, classifier, optimizer, writer, epoch, n_iter):
    '''
    run one epoch of training
    '''

    # set the classifier in train mode
    classifier.train()

    num_classes = int(cfg['n_classes'])
    num_batch = cfg['num_batch']

    ep_loss = 0.
    ep_loss_dict = initialize_loss_dict(cfg)
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
        #logits = classifier(points)
        pred = classifier(points)
        #print(pred,preds.shape)
        ### minimize the loss

        #pred = F.log_softmax(logits, dim=-1).view(-1, num_classes)
        #pred_choice = pred.data.max(1)[1].int()

        #loss = F.mse_loss(pred, target.long())
        target = target.view(-1, num_classes)
        loss = F.mse_loss(pred,target.float())
        
        ep_loss += loss.item()
        running_ep_loss = ep_loss / (i_batch + 1)

        loss.backward()

        if int(cfg['accumulation_interval']) % (i_batch + 1) == 0:
            optimizer.step()
            optimizer.zero_grad
        elif not cfg['accumulation_interval']:
            optimizer.step()

        ### compute performance
        #update_metrics(metrics, pred_choice, target)
        update_metrics(metrics, pred.float(), target.float())
        #running_acc = torch.tensor(metrics['acc']).mean().item()
        print(loss[0])
        print('[%d: %d/%d] train loss: %f mse: %f' \
            % (epoch, i_batch, num_batch, loss.item(), metrics['mse'][-1]))

        n_iter += 1

    ep_loss = ep_loss / (i_batch + 1)
    writer.add_scalar('train/epoch_loss', ep_loss, epoch)
    log_losses(ep_loss_dict, writer, epoch, i_batch + 1)
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
            #logits = classifier(points)
            pred = classifier(points)

            #pred = F.log_softmax(logits, dim=-1).view(-1, num_classes)
            #pred_choice = pred.data.max(1)[1].int()
            target = target.view(-1, num_classes)
            loss = F.mse_loss(pred,target.float())
           
            ep_loss += loss.item()

            #print('val min / max class pred %d / %d' %
            #      (pred_choice.min().item(), pred_choice.max().item()))
            #print('# class pred ', len(torch.unique(pred_choice)))

            ### compute performance
            #update_metrics(metrics_val, pred_choice, target)
            update_metrics(metrics_val, pred.float(), target.float())

            print('VALIDATION [%d: %d/%d] val loss: %f mse: %f' %
                  ((epoch, i, len(val_dataloader), loss.item(),
                    metrics_val['mse'][-1])))

        writer.add_scalar('val/loss', ep_loss / i, epoch)
        log_avg_metrics(writer, metrics_val, 'val', epoch)
        #epoch_score = torch.tensor(metrics_val['acc']).mean().item()
        epoch_score = torch.tensor(metrics_val['mse']).float().mean().item()
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
        trans_train.append(
            RndSampling(sample_size,
                        maintain_prop=False))
                        #prop_vector=[1, 1]))
        trans_val.append(RndSampling(sample_size, maintain_prop=False))

    dataset, dataloader = get_dataset(cfg, trans=trans_train)
    val_dataset, val_dataloader = get_dataset(cfg, trans=trans_val, train=False)
    # summary for tensorboard
    writer = create_tb_logger(cfg)
    dump_code(cfg, writer.logdir)

    #### BUILD THE MODEL
    classifier = get_model(cfg)
    if cfg['verbose']:
        print(classifier)

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
