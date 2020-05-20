import glob
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch as gBatch
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from torchvision import transforms

import datasets as ds
from utils.data.data_utils import (get_dataset, get_gbatch_sample,
                                   get_transforms)
from utils.general_utils import (initialize_metrics, log_avg_metrics,
                                 update_metrics)
from utils.model_utils import get_model, print_net_graph
from utils.train_utils import (compute_loss, create_tb_logger, dump_code,
                               dump_model, get_lr, get_lr_scheduler,
                               get_optimizer, initialize_loss_dict, log_losses,
                               update_bn_decay, set_lr)
from utils.data.transforms import RndSampling, TestSampling, SampleStandardization

def test(cfg):
    num_classes = int(cfg['n_classes'])
    sample_size = int(cfg['fixed_size'])
    cfg['loss'] = cfg['loss'].split(' ')
    batch_size = 1
    cfg['batch_size'] = batch_size
    epoch = eval(str(cfg['n_epochs']))
    #n_gf = int(cfg['num_gf'])
    input_size = int(cfg['data_dim'])

    trans_val = []
    if cfg['rnd_sampling']:
        trans_val.append(TestSampling(sample_size))
    if cfg['standardization']:
        trans_val.append(SampleStandardization())

    if cfg['dataset'] == 'hcp20_graph':
        dataset = ds.HCP20Dataset(cfg['sub_list_test'],
                                  cfg['dataset_dir'],
                                  transform=transforms.Compose(trans_val),
                                  with_gt=cfg['with_gt'],
                                  #distance=T.Distance(norm=True,cat=False),
                                  return_edges=True,
                                  split_obj=True,
                                  train=False,
                                  load_one_full_subj=False,
                                  labels_dir=cfg['labels_dir'])

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    print("Validation dataset loaded, found %d samples" % (len(dataset)))

    for ext in range(100):
        logdir = '%s/test_%d' % (cfg['exp_path'], ext)
        if not os.path.exists(logdir):
            break
    writer = SummaryWriter(logdir)
    if cfg['weights_path'] == '':
        cfg['weights_path'] = glob.glob(cfg['exp_path'] + '/models/best*')[0]
        epoch = int(cfg['weights_path'].rsplit('-',1)[1].split('.')[0])
    elif 'ep-' in cfg['weights_path']:
        epoch = int(cfg['weights_path'].rsplit('-',1)[1].split('.')[0])

    tb_log_name = glob.glob('%s/events*' % writer.logdir)[0].rsplit('/',1)[1]
    tb_log_dir = 'tb_logs/%s' % logdir.split('/',1)[1]
    os.system('mkdir -p %s' % tb_log_dir)
    os.system('ln -sr %s/%s %s/%s ' %
                        (writer.logdir, tb_log_name, tb_log_dir, tb_log_name))

    #### BUILD THE MODEL
    classifier = get_model(cfg)

    classifier.cuda()
    classifier.load_state_dict(torch.load(cfg['weights_path']))
    classifier.eval()


    with torch.no_grad():
        pred_buffer = {}
        sm_buffer = {}
        sm2_buffer = {}
        gf_buffer = {}
        emb_buffer = {}
        print('\n\n')
        #mean_val_acc = torch.tensor([])
        #mean_val_iou = torch.tensor([])
        #mean_val_prec = torch.tensor([])
        #mean_val_recall = torch.tensor([])
        mean_val_mse = torch.tensor([])
        mean_val_mae = torch.tensor([])

        if 'split_obj' in dir(dataset) and dataset.split_obj:
            split_obj = True
        else:
            split_obj = False
            dataset.transform = []

        if split_obj:
            consumed = False
        else:
            consumed = True
        j = 0
        visualized = 0
        new_obj_read = True
        sls_count = 1
        while j < len(dataset):
        #while sls_count <= len(dataset):
            data = dataset[j]

            if split_obj:
                if new_obj_read:
                    #obj_pred_choice = torch.zeros(data['obj_full_size'], dtype=torch.int).cuda()
                    #obj_target = torch.zeros(data['obj_full_size'], dtype=torch.int).cuda()
                    obj_pred_choice = torch.zeros(data['obj_full_size'], dtype=torch.float32).cuda()
                    obj_target = torch.zeros(data['obj_full_size'], dtype=torch.float32).cuda()
                    new_obj_read = False

                if len(dataset.remaining[j]) == 0:
                    consumed = True

            sample_name = data['name'] if type(data['name']) == str else data['name'][0]

            #print(points)
            #if len(points.shape()) == 2:
                #points = points.unsqueeze(0)
            #print(data)
            points = gBatch().from_data_list([data['points']])
            #points = data['points']
            if 'bvec' in points.keys:
                points.batch = points.bvec.clone()
                del points.bvec            
            if cfg['with_gt']:
                target = points['y']
                target = target.to('cuda')
            if cfg['same_size']:
                points['lengths'] = points['lengths'][0].item()
            #if cfg['model'] == 'pointnet_cls':
                #points = points.view(len(data['obj_idxs']), -1, input_size)
            points = points.to('cuda')

            pred = classifier(points)
            
            #logits = classifier(points)
            #logits = logits.view(-1, num_classes)
            
            #pred = F.log_softmax(logits, dim=-1).view(-1, num_classes)
            #pred_choice = pred.data.max(1)[1].int()

            if split_obj:
                obj_pred_choice[data['obj_idxs']] = pred.view(-1)
                #obj_pred_choice[data['obj_idxs']] = pred_choice
                obj_target[data['obj_idxs']] = target.float()
                #print(obj_pred_choice)
                #print(obj_target)
                #obj_target[data['obj_idxs']] = target.int()
                #if cfg['save_embedding']:
                #    obj_embedding[data['obj_idxs']] = classifier.embedding.squeeze()
            else:
                obj_data = points
                obj_pred_choice = pred_choice
                obj_target = target
                if cfg['save_embedding']:
                    obj_embedding = classifier.embedding.squeeze()

            if cfg['with_gt'] and consumed:
                print('val max class pred ', obj_pred_choice.max().item())
                print('val min class pred ', obj_pred_choice.min().item())
                print('val max class target ', obj_target.max().item())
                print('val min class target ', obj_target.min().item())
                #obj_pred_choice = obj_pred_choice.view(-1,1)
                #obj_target = obj_target.view(-1,1)
               
                mae = torch.mean(abs(obj_target.data.cpu() - obj_pred_choice.data.cpu())).item()
                mse = torch.mean((obj_target.data.cpu() - obj_pred_choice.data.cpu())**2).item()
                #correct = obj_pred_choice.eq(obj_target.data.int()).cpu().sum()
                #acc = correct.item()/float(obj_target.size(0))
                #tp = torch.mul(obj_pred_choice.data, obj_target.data.int()).cpu().sum().item()+0.00001
                #fp = obj_pred_choice.gt(obj_target.data.int()).cpu().sum().item()
                #fn = obj_pred_choice.lt(obj_target.data.int()).cpu().sum().item()
                #tn = correct.item() - tp
                #iou = torch.tensor([float(tp)/(tp+fp+fn)])
                #prec = torch.tensor([float(tp)/(tp+fp)])
                #recall = torch.tensor([float(tp)/(tp+fn)])

                mean_val_mae = torch.cat((mean_val_mae, torch.tensor([mae])), 0)
                mean_val_mse = torch.cat((mean_val_mse, torch.tensor([mse])), 0)
                #mean_val_prec = torch.cat((mean_val_prec, prec), 0)
                #mean_val_recall = torch.cat((mean_val_recall, recall), 0)
                #mean_val_iou = torch.cat((mean_val_iou, iou), 0)
                #mean_val_acc = torch.cat((mean_val_acc, torch.tensor([acc])), 0)
                print('VALIDATION [%d: %d/%d] val mse: %f val mae: %f' \
                        % (epoch, j, len(dataset), mse, mae))

            if cfg['save_pred'] and consumed:
                print('buffering prediction %s' % sample_name)
                sl_idx = np.where(obj_pred.data.cpu().view(-1).numpy() == 1)[0]
                pred_buffer[sample_name] = sl_idx.tolist()

            if consumed:
                print(j)
                j += 1
                if split_obj:
                    consumed = False
                    new_obj_read = True

        #macro_iou = torch.mean(mean_val_iou)
        #macro_prec = torch.mean(mean_val_prec)
        #macro_recall = torch.mean(mean_val_recall)

        #epoch_iou = macro_iou.item()

    if cfg['save_pred']:
        #os.system('rm -r %s/predictions_test*' % writer.logdir)
        pred_dir = writer.logdir + '/predictions_test_%d' % epoch
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        print('saving files')
        for filename, value in pred_buffer.items():
            with open(os.path.join(pred_dir, filename) + '.pkl', 'wb') as f:
                pickle.dump(
                    value, f, protocol=pickle.HIGHEST_PROTOCOL)

    if cfg['with_gt']:
        print('TEST MSE: %f' % torch.mean(mean_val_mse).item())
        print('TEST MAE: %f' % torch.mean(mean_val_mae).item())
        #print('TEST ACCURACY: %f' % torch.mean(mean_val_acc).item())
        #print('TEST PRECISION: %f' % macro_prec.item())
        #print('TEST RECALL: %f' % macro_recall.item())
        #print('TEST IOU: %f' % macro_iou.item())
        #mean_val_dsc = mean_val_prec * mean_val_recall * 2 / (mean_val_prec + mean_val_recall)
        final_scores_file = writer.logdir + '/final_scores_test_%d.txt' % epoch
        scores_file = writer.logdir + '/scores_test_%d.txt' % epoch
        print('saving scores')
        with open(scores_file, 'w') as f: 
          f.write('mse\n')
          f.writelines('%f\n' % v for v in mean_val_mse.tolist())
          f.write('mae\n')
          f.writelines('%f\n' % v for v in mean_val_mae.tolist())
          #f.write('acc\n')
          #f.writelines('%f\n' % v for v in  mean_val_acc.tolist())
          #f.write('prec\n')
          #f.writelines('%f\n' % v for v in  mean_val_prec.tolist())
          #f.write('recall\n')
          #f.writelines('%f\n' % v for v in  mean_val_recall.tolist())
          #f.write('dsc\n')
          #f.writelines('%f\n' % v for v in  mean_val_dsc.tolist())
          #f.write('iou\n')
          #f.writelines('%f\n' % v for v in  mean_val_iou.tolist())
        with open(final_scores_file, 'w') as f:
          f.write('mse\n')
          f.write('%f\n' % mean_val_mse.mean())
          f.write('%f\n' % mean_val_mse.std())
          f.write('mae\n')
          f.write('%f\n' % mean_val_mae.mean())
          f.write('%f\n' % mean_val_mae.std())
          #f.write('acc\n')
          #f.write('%f\n' % mean_val_acc.mean())
          #f.write('%f\n' % mean_val_acc.std())
          #f.write('prec\n')
          #f.write('%f\n' % mean_val_prec.mean())
          #f.write('%f\n' % mean_val_prec.std())
          #f.write('recall\n')
          #f.write('%f\n' % mean_val_recall.mean())
          #f.write('%f\n' % mean_val_recall.std())
          #f.write('dsc\n')
          #f.write('%f\n' % mean_val_dsc.mean())
          #f.write('%f\n' % mean_val_dsc.std())
          #f.write('iou\n')
          #f.write('%f\n' % mean_val_iou.mean())
          #f.write('%f\n' % mean_val_iou.std())

    print('\n\n')
