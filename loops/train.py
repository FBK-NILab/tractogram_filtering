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

        sample_batched = sample_batched.to('cuda')
        target = sample_batched['y']

        ### initialize gradients
        if not cfg['accumulation_interval'] or i_batch == 0:
            optimizer.zero_grad()

        ### forward
        logits = classifier(sample_batched)
        ### minimize the loss

        pred = F.log_softmax(logits, dim=-1).view(-1, num_classes)
        pred_choice = pred.data.max(1)[1].int()

        loss = F.nll_loss(pred, target.long())

        ep_loss += loss.item()
        running_ep_loss = ep_loss / (i_batch + 1)

        loss.backward()

        if int(cfg['accumulation_interval']) % (i_batch + 1) == 0:
            optimizer.step()
            optimizer.zero_grad
        elif not cfg['accumulation_interval']:
            optimizer.step()

        ### compute performance
        update_metrics(metrics, pred_choice, target)
        running_acc = torch.tensor(metrics['acc']).mean().item()

        if cfg['verbose']:
            print('[%d: %d%d] train loss: %f acc: %f' \
                % (epoch, i_batch, num_batch, loss.item(), metrics['acc'][-1]))
        else:
            print_prefix = '\r[%d: %d%d] train loss: ' % (epoch, i_batch,
                                                          num_batch)
            sys.stdout.write('%s %.4f acc: %.4f ep_loss: %.4f ep_acc: %.4f ' \
                % (print_prefix, loss.item(), metrics['acc'][-1], running_ep_loss, running_acc))
            if ep_loss_dict is not None:
                sys.stdout.write(' '.join([
                    '%s=%.4f' % (k, v / (i_batch + 1))
                    for k, v in ep_loss_dict.iteritems()
                ]))

        n_iter += 1

    ep_loss = ep_loss / (i_batch + 1)
    writer.add_scalar('train/epoch_loss', ep_loss, epoch)
    log_losses(ep_loss_dict, writer, epoch, i_batch + 1)
    log_avg_metrics(writer, metrics, 'train', epoch)

    return ep_loss, n_iter


def val_ep(cfg, val_dataloader, classifier, writer, epoch):
    '''
    run the validation phase when called
    '''
    num_classes = int(cfg['n_classes'])

    # set classifier in eval mode
    classifier.eval()

    with torch.no_grad():
        print('\n\n')

        metrics_val = initialize_metrics()
        ep_loss = 0.

        for i, data in enumerate(dataloader):
            
            data = data.to('cuda')
            target = data['y']

            ### forward
            logits = classifier(data)

            pred = F.log_softmax(logits, dim=-1).view(-1, num_classes)
            pred_choice = pred.data.max(1)[1].int()
            loss = F.nll_loss(pred, target.long())

            ep_loss += loss.item()

            ### compute performance
            update_metrics(metrics_val, pred_choice, target)

            if cfg['verbose']:
                print('val min / max class pred %d / %d' %
                      (pred_choice.min().item(), pred_choice.max().item()))
                print('# class pred ', len(torch.unique(pred_choice)))

                print('VALIDATION [%d: %d%d] val loss: %f acc: %f' %
                      ((epoch, i, len(dataloader), loss.item(),
                      metrics_val['acc'][-1])))
            else:
                sys.stdout.write(
                    '\rVALIDATION [%d: %d/%d] val loss: %f acc: %f' %
                    ((epoch, i, len(dataloader), loss.item(),
                      metrics_val['acc'][-1]))
                )

        writer.add_scalar('val/loss', ep_loss / i, epoch)
        log_avg_metrics(writer, metrics_val, 'val', epoch)
        epoch_score = torch.tensor(metrics_val['acc']).mean().item()
        print('\nVALIDATION ACCURACY: %f' % epoch_score)
        print('\n\n')

        return epoch_score


def train(cfg):

    batch_size = int(cfg['batch_size'])
    n_epochs = int(cfg['n_epochs'])
    sample_size = int(cfg['fixed_size'])
    cfg['loss'] = cfg['loss'].split(' ')

    #### DATA LOADING
    dataset, dataloader = get_dataset(cfg, trans=get_transforms(cfg))
    val_dataset, val_dataloader = get_dataset(cfg, trans=get_transforms(cfg, train=False), train=False)
    
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
    best_score = 0
    current_lr = float(cfg['learning_rate'])
    ep0 = 0

    for epoch in range(ep0, ep0 + n_epochs + 1):

        loss, n_iter = train_ep(cfg, dataloader, classifier, optimizer, writer,
                                epoch, n_iter)

        ### validation during training
        if epoch % int(cfg['val_freq']) == 0 
                or epoch == n_epochs and cfg['val_in_train']:
            best = False
            val_score = val_ep(cfg, val_dataloader, classifier, writer, epoch)
            if val_score >= best_score:
                best = True
                best_score = val_score

        if cfg['save_model']:
            dump_model(cfg,
                       classifier,
                       optimizer,
                       loss,
                       epoch,
                       val_score,
                       writer.logdir,
                       best=best)

        # update bn decay
        if cfg['bn_decay'] and epoch != 0 and epoch % int(
                cfg['bn_decay_step']) == 0:
            update_bn_decay(cfg, classifier, epoch)

        # update lr
        if cfg['lr_type'] == 'step' and current_lr >= float(cfg['min_lr']):
            lr_scheduler.step()
        if cfg['lr_type'] == 'plateau':
            lr_scheduler.step(loss)

        current_lr = get_lr(optimizer)
        writer.add_scalar('train/lr', current_lr, epoch)

    writer.close()