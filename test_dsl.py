import glob
import os
import pickle
import sys
from time import time
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch as gBatch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from visdom import Visdom

import datasets as ds
import lovasz_losses as L
from cluster_losses import CSimLoss, DistLoss, HLoss
from tensorboardX import SummaryWriter
from torchvision import transforms
from train_dsl import get_model
import torch_geometric.transforms as T
from nilab.load_trk import load_streamlines
from dipy.tracking.streamline import length

def get_ncolors(n):
    random.seed(10)
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append([r,g,b])
    return np.array(ret, dtype=np.uint8)

def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    colors = [
        "FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF", "000000",
        "800000", "008000", "000080", "808000", "800080", "008080", "808080",
        "C00000", "00C000", "0000C0", "C0C000", "C000C0", "00C0C0", "C0C0C0",
        "400000", "004000", "000040", "404000", "400040", "004040", "404040",
        "200000", "002000", "000020", "202000", "200020", "002020", "202020",
        "600000", "006000", "000060", "606000", "600060", "006060", "606060",
        "A00000", "00A000", "0000A0", "A0A000", "A000A0", "00A0A0", "A0A0A0",
        "E00000", "00E000", "0000E0", "E0E000", "E000E0", "00E0E0", "E0E0E0",
        ]

    colors = colors[:n]
    colors = [[int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)] for i in colors]
    return np.array(colors, dtype=np.uint8)

def test(cfg):
    num_classes = int(cfg['n_classes'])
    sample_size = int(cfg['fixed_size'])
    cfg['loss'] = cfg['loss'].split(' ')
    batch_size = 1
    cfg['batch_size'] = batch_size
    epoch = eval(cfg['n_epochs'])
    n_gf = int(cfg['num_gf'])
    input_size = int(cfg['data_dim'])

    trans_val = []
    if cfg['rnd_sampling']:
        trans_val.append(ds.TestSampling(sample_size))
    if cfg['standardization']:
        trans_val.append(ds.SampleStandardization())


    if cfg['dataset'] == 'left_ifof_ss_sl':
        dataset = ds.LeftIFOFSupersetDataset(cfg['sub_list_test'],
                                cfg['dataset_dir'],
                                transform=transforms.Compose(trans_val),
                                uniform_size=True,
                                train=False,
                                split_obj=True,
                                with_gt=cfg['with_gt'])
    elif cfg['dataset'] == 'hcp20_graph':
        dataset = ds.HCP20Dataset(cfg['sub_list_test'],
                                  cfg['dataset_dir'],
                                  act=cfg['act'],
                                  transform=transforms.Compose(trans_val),
                                  with_gt=cfg['with_gt'],
                                  #distance=T.Distance(norm=True,cat=False),
                                  return_edges=False,
                                  split_obj=True,
                                  train=False)
    elif cfg['dataset'] == 'left_ifof_ss_sl_graph':
        dataset = ds.LeftIFOFSupersetGraphDataset(cfg['sub_list_test'],
                                cfg['dataset_dir'],
                                transform=transforms.Compose(trans_val),
                                train=False,
                                split_obj=True,
                                with_gt=cfg['with_gt'])
    elif cfg['dataset'] == 'left_ifof_emb':
        dataset = ds.EmbDataset(cfg['sub_list_test'],
                                cfg['emb_dataset_dir'],
                                cfg['gt_dataset_dir'],
                                transform=transforms.Compose(trans_val),
                                load_all=cfg['load_all_once'],
                                precompute_graph=cfg['precompute_graph'],
                                k_graph=int(cfg['knngraph']))
    elif cfg['dataset'] == 'psb_airplane':
        dataset = ds.PsbAirplaneDataset(cfg['dataset_dir'], train=False)
    elif cfg['dataset'] == 'shapes':
        dataset = ds.ShapesDataset(cfg['dataset_dir'],
                                   train=False,
                                   multi_cat=cfg['multi_category'])
    elif cfg['dataset'] == 'shapenet':
        dataset = ds.ShapeNetCore(cfg['dataset_dir'],
                                    train=False,
                                    multi_cat=cfg['multi_category'])
    elif cfg['dataset'] == 'modelnet':
        dataset = ds.ModelNetDataset(cfg['dataset_dir'],
                                        split=cfg['mn40_split'],
                                        fold_size=int(cfg['mn40_fold_size']),
                                        load_all=cfg['load_all_once'])
    elif cfg['dataset'] == 'scanobj':
        dataset = ds.ScanObjNNDataset(cfg['dataset_dir'],
                                        run='test',
                                        variant=cfg['scanobj_variant'],
                                        background=cfg['scanobj_bg'],
                                        load_all=cfg['load_all_once'])
    else:
        dataset = ds.DRLeftIFOFSupersetDataset(cfg['sub_list_test'],
                                               cfg['val_dataset_dir'],
                                               transform=transforms.Compose(trans_val),
                                               with_gt=cfg['with_gt'])

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
        mean_val_acc = torch.tensor([])
        mean_val_iou = torch.tensor([])
        mean_val_prec = torch.tensor([])
        mean_val_recall = torch.tensor([])

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
            data = dataset[j]

            if split_obj:
                if new_obj_read:
                    obj_pred_choice = torch.zeros(data['obj_full_size'], dtype=torch.int).cuda()
                    obj_target = torch.zeros(data['obj_full_size'], dtype=torch.int).cuda()
                    new_obj_read = False
                    if cfg['save_embedding']:
                        obj_embedding = torch.empty((data['obj_full_size'], int(cfg['embedding_size']))).cuda()

                if len(dataset.remaining[j]) == 0:
                    consumed = True
                    
            sample_name = data['name'] if type(data['name']) == str else data['name'][0]
             
            #print(points)
            #if len(points.shape()) == 2:
                #points = points.unsqueeze(0)
            if 'graph' not in cfg['dataset']:
                points = data['points']
                if cfg['with_gt']:
                    target = data['gt']
                    target = target.to('cuda')
                    target = target.view(-1, 1)[:, 0]
            else:
                points = gBatch().from_data_list([data['points']])
                if 'bvec' in points.keys:
                    points.batch = points.bvec.clone()
                    del points.bvec
                if cfg['with_gt']:
                    target = points['y']
                    target = target.to('cuda')
                    target = target.view(-1, 1)[:, 0]
                if cfg['same_size']:
                    points['lengths'] = points['lengths'][0].item()
            #if cfg['model'] == 'pointnet_cls':
                #points = points.view(len(data['obj_idxs']), -1, input_size)
            points = points.to('cuda')
            #print('streamline number:',sls_count)
            #sls_count+=1
            #print('lengths:',points['lengths'].item())
            ### add one-hot labels if multi-category task
            #new_k = points['lengths'].item()*(5/16)
            #print('new k:',new_k,'rounded k:',int(round(new_k)))
            #classifier.conv2.k = int(round(new_k))
            if cfg['multi_category']:
                one_hot_label = Variable(data['category'])
                classifier.category_vec = one_hot_label.cuda()

            if cfg['multi_loss']:
                logits, gf = classifier(points)
            else:
                logits = classifier(points)
            logits = logits.view(-1, num_classes)

            if len(cfg['loss']) == 2:
                if epoch <= int(cfg['switch_loss_epoch']):
                    loss_type = cfg['loss'][0]
                else:
                    loss_type = cfg['loss'][1]
            else:
                loss_type = cfg['loss'][0]

                if loss_type == 'nll':
                    pred = F.log_softmax(logits, dim=-1)
                    pred = pred.view(-1,num_classes)
                    probas = torch.exp(pred.data)
                    pred_choice = pred.data.max(1)[1].int()
                    if cfg['with_gt']:
                        loss_seg = F.nll_loss(pred, target.long())
                elif loss_type == 'LLh':
                    pred_choice = (logits.data > 0).int()
                    if cfg['with_gt']:
                        loss_seg = L.lovasz_hinge(logits.view(batch_size, sample_size, 1),
                                                target.view(
                            batch_size, sample_size, 1),
                            per_image=False)
                    #loss = L.lovasz_hinge_flat(pred.view(-1), target.view(-1))
                elif loss_type == 'LLm':
                    pred = F.softmax(logits, dim=-1)
                    probas = pred.data
                    pred_choice = pred.data.max(1)[1].int()
                    if cfg['with_gt']:
                        loss = L.lovasz_softmax_flat(pred, target,
                                                    op=cfg['llm_op'],
                                                    only_present=cfg['multi_category'])
            #print('pred:',pred)
            #print('pred shape:',pred.shape)
            #print('pred choice:',pred_choice)
            #print('pred choice shape:',pred_choice.shape)
            if visualized < int(cfg['viz_clusters']):
                visualized += 1
                colors = torch.from_numpy(get_spaced_colors(n_gf))
                sm_out = classifier.feat.mf.softmax_out[0,:,:].max(1)[1].squeeze().int()
                writer.add_mesh('latent clustering', points, colors[sm_out.tolist()].unsqueeze(0))
                if 'bg' in data.keys():
                    bg_msk = data['bg']*-1
                    writer.add_mesh('bg_mask', points, colors[bg_msk.tolist()].unsqueeze(0))


            if split_obj:
                obj_pred_choice[data['obj_idxs']] = pred_choice
                obj_target[data['obj_idxs']] = target.int()
                if cfg['save_embedding']:
                    obj_embedding[data['obj_idxs']] = classifier.embedding.squeeze()
            else:
                obj_data = points
                obj_pred_choice = pred_choice
                obj_target = target
                if cfg['save_embedding']:
                    obj_embedding = classifier.embedding.squeeze()

            if cfg['with_gt'] and consumed:
                if cfg['multi_loss']:
                    loss_cluster = cluster_loss_fn(gf.squeeze(3))
                    loss = loss_seg + alfa * loss_cluster

                    #pred_choice = torch.sigmoid(pred.view(-1,1)).data.round().type_as(target.data)
                #print('points:',points['streamlines'])
                #print('points shape:',points['streamlines'].shape)
                #print('streamlines:',
                data_dir = cfg['dataset_dir']
                streamlines, head, leng, idxs = load_streamlines(data['dir']+'/'+data['name']+'.trk')
                #print('tract:',len(streamlines))
                #print('pred:',obj_pred_choice)
                #print('taget:',obj_target)
                #print('pred shape:',obj_pred_choice.shape)
                #print('target shape:',obj_target.shape)
                print('val max class red ', obj_pred_choice.max().item())
                print('val min class pred ', obj_pred_choice.min().item())
                y_pred = obj_pred_choice.cpu().numpy()
                np.save(data['dir']+'/y_pred_blstm_more_pars',y_pred)
                y_test = obj_target.cpu().numpy()
                np.save(data['dir']+'/y_test_blstm_more_pars',y_test)
                np.save(data['dir']+'/streamlines_blstm_more_pars',streamlines)
                correct = obj_pred_choice.eq(obj_target.data.int()).cpu().sum()
                acc = correct.item()/float(obj_target.size(0))

                if num_classes > 2:
                    iou, prec, recall = L.iou_multi(obj_pred_choice.data.int().cpu().numpy(),
                                            obj_target.data.int().cpu().numpy(), num_classes,
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
                    assert(torch.isnan(iou).sum() == 0)

                else:
                    tp = torch.mul(obj_pred_choice.data, obj_target.data.int()).cpu().sum().item()+0.00001
                    fp = obj_pred_choice.gt(obj_target.data.int()).cpu().sum().item()
                    fn = obj_pred_choice.lt(obj_target.data.int()).cpu().sum().item()
                    tn = correct.item() - tp
                    iou = torch.tensor([float(tp)/(tp+fp+fn)])
                    prec = torch.tensor([float(tp)/(tp+fp)])
                    recall = torch.tensor([float(tp)/(tp+fn)])

                mean_val_prec = torch.cat((mean_val_prec, prec), 0)
                mean_val_recall = torch.cat((mean_val_recall, recall), 0)
                mean_val_iou = torch.cat((mean_val_iou, iou), 0)
                mean_val_acc = torch.cat((mean_val_acc, torch.tensor([acc])), 0)
                print('VALIDATION [%d: %d/%d] val accuracy: %f' \
                        % (epoch, j, len(dataset), acc))

            if cfg['save_pred'] and consumed:
                print('buffering prediction %s' % sample_name)
                if num_classes > 2:
                    for cl in range(num_classes):
                        sl_idx = np.where(obj_pred.data.cpu().view(-1).numpy() == cl)[0]
                        if cl == 0:
                            pred_buffer[sample_name] = []
                        pred_buffer[sample_name].append(sl_idx.tolist())
                else:
                    sl_idx = np.where(obj_pred.data.cpu().view(-1).numpy() == 1)[0]
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
            #if cfg['save_gf']:
                #   gf_buffer[sample_name] = np.unique(
                #           classifier.feat.globalfeat.data.cpu().squeeze().numpy(), axis = 0)
                #gf_buffer[sample_name] = classifier.globalfeat
            if cfg['save_embedding'] and consumed:
                emb_buffer[sample_name] = obj_embedding


            if consumed:
                print(j)
                j += 1
                if split_obj:
                    consumed = False
                    new_obj_read = True



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

        epoch_iou = macro_iou.item()

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

    if cfg['save_softmax_out']:
        os.system('rm -r %s/sm_out_test*' % writer.logdir)
        sm_dir = writer.logdir + '/sm_out_test_%d' % epoch
        if not os.path.exists(sm_dir):
            os.makedirs(sm_dir)
        for filename, value in sm_buffer.iteritems():
            with open(os.path.join(sm_dir, filename) + '_sm_1.pkl', 'wb') as f:
                pickle.dump(
                    value, f, protocol=pickle.HIGHEST_PROTOCOL)
        for filename, value in sm2_buffer.iteritems():
            with open(os.path.join(sm_dir, filename) + '_sm_2.pkl', 'wb') as f:
                pickle.dump(
                    value, f, protocol=pickle.HIGHEST_PROTOCOL)

    if cfg['save_gf']:
        #os.system('rm -r %s/gf_test*' % writer.logdir)
        gf_dir = writer.logdir + '/gf_test_%d' % epoch
        if not os.path.exists(gf_dir):
            os.makedirs(gf_dir)
        i = 0
        for filename, value in gf_buffer.items():
            if i == 3:
                break
            with open(os.path.join(gf_dir, filename) + '.pkl', 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)


    if cfg['save_embedding']:
        print('saving embedding')
        emb_dir = writer.logdir + '/embedding_test_%d' % epoch
        if not os.path.exists(emb_dir):
            os.makedirs(emb_dir)
        for filename, value in emb_buffer.iteritems():
            np.save(os.path.join(emb_dir, filename), value.cpu().numpy())

    if cfg['with_gt']:
        print('TEST ACCURACY: %f' % torch.mean(mean_val_acc).item())
        print('TEST PRECISION: %f' % macro_prec.item())
        print('TEST RECALL: %f' % macro_recall.item())
        print('TEST IOU: %f' % macro_iou.item())
        mean_val_dsc = mean_val_prec * mean_val_recall * 2 / (mean_val_prec + mean_val_recall)
        final_scores_file = writer.logdir + '/final_scores_test_%d.txt' % epoch
        scores_file = writer.logdir + '/scores_test_%d.txt' % epoch
        if not cfg['multi_category']:
            print('saving scores')
            with open(scores_file, 'w') as f:
                f.write('acc\n')
                f.writelines('%f\n' % v for v in  mean_val_acc.tolist())
                f.write('prec\n')
                f.writelines('%f\n' % v for v in  mean_val_prec.tolist())
                f.write('recall\n')
                f.writelines('%f\n' % v for v in  mean_val_recall.tolist())
                f.write('dsc\n')
                f.writelines('%f\n' % v for v in  mean_val_dsc.tolist())
                f.write('iou\n')
                f.writelines('%f\n' % v for v in  mean_val_iou.tolist())
            with open(final_scores_file, 'w') as f:
                f.write('acc\n')
                f.write('%f\n' % mean_val_acc.mean())
                f.write('%f\n' % mean_val_acc.std())
                f.write('prec\n')
                f.write('%f\n' % mean_val_prec.mean())
                f.write('%f\n' % mean_val_prec.std())
                f.write('recall\n')
                f.write('%f\n' % mean_val_recall.mean())
                f.write('%f\n' % mean_val_recall.std())
                f.write('dsc\n')
                f.write('%f\n' % mean_val_dsc.mean())
                f.write('%f\n' % mean_val_dsc.std())
                f.write('iou\n')
                f.write('%f\n' % mean_val_iou.mean())
                f.write('%f\n' % mean_val_iou.std())

    print('\n\n')
