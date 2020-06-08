from __future__ import print_function

import argparse
import configparser
import glob
import json
import os
from time import time

import ants
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.data import Batch as gBatch
from torch_geometric.data import DataListLoader as gDataLoader

from datasets.basic_tract import TractDataset
# from nilab import load_trk as ltrk
from utils.data import selective_loader as sload
from utils.data.data_utils import resample_streamlines, tck2trk, trk2tck
from utils.general_utils import get_cfg_value
from utils.model_utils import get_model

# os.environ["DEVICE"] = torch.device(
#     'cuda' if torch.cuda.is_available() else 'cpu')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tract2standard(t_fn, t1_fn, fixed_fn):
    print('registration using ANTs SyN...')
    fixed = ants.image_read(fixed_fn)
    moving = ants.image_read(t1_fn)
    mytx = ants.registration(fixed=fixed,
                             moving=moving,
                             type_of_transform='SyN')

    print('correcting warp to mrtrix convention...')
    os.system(f'warpinit {fixed_fn} temp/ID_warp[].nii.gz -force')

    for i in range(3):
        temp_warp = ants.image_read(f'temp/ID_warp{i}.nii.gz')
        temp_warp = ants.apply_transforms(fixed=fixed,
                                          moving=temp_warp,
                                          transformlist=mytx['invtransforms'],
                                          whichtoinvert=[True, False])
        ants.image_write(temp_warp, f'temp/mrtrix_warp{i}.nii.gz')

    os.system('warpcorrect temp/mrtrix_warp[].nii.gz ' +
              'temp/mrtrix_warp_cor.nii.gz -force')

    print('applaying warp to tractogram...')
    t_mni_fn = t_fn[:-4] + '_mni.tck'
    os.system(f'tcktransform {t_fn} temp/mrtrix_warp_cor.nii.gz {t_mni_fn} ' +
              '-force -nthreads 0')

    return t_mni_fn


def get_sample(data):
    gdata = gBatch().from_data_list([data['points']])
    gdata = gdata.to(DEVICE)
    gdata.batch = gdata.bvec.clone()
    del gdata.bvec
    gdata['lengths'] = gdata['lengths'][0].item()

    return gdata


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-config', nargs='?', default='run_config.json',
                        help='The tag for the configuration file.')
    args = parser.parse_args()

    ## load config
    t0_global = time()
    print('reading arguments')
    cfg = json.load(open(args.config))
    print(cfg)

    move_tract = cfg['t1'] != ''
    tck_fn = cfg['trk'][:-4] + '.tck'
    trk_fn = 'temp/input/tract_mni_resampled.trk'

    in_dir = 'temp/input'
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)

    ## resample trk to 16points if needed
    if cfg['resample_points']:
        t0 = time()
        print('loading tractogram...')
        streams, lengths = sload.load_selected_streamlines(cfg['trk'])
        streamlines = np.split(streams, np.cumsum(lengths[:-1]))
        print(f'done in {time()-t0} sec')
        t0 = time()
        print('streamlines resampling...')
        streamlines = resample_streamlines(streamlines)
        print(f'done in {time()-t0} sec')
        t0 = time()
        print('saving resampled tractogram...')
        resampled_t = nib.streamlines.Tractogram(streamlines,
                                                 affine_to_rasmm=np.eye(4))
        resampled_fn = tck_fn if move_tract else trk_fn
        nib.streamlines.save(resampled_t, resampled_fn)
        print(f'done in {time()-t0} sec')

    ## compute warp to mni and move tract if needed
    if move_tract:
        if not os.path.exists(tck_fn):
            t0 = time()
            print('convert trk to tck...')
            trk2tck(cfg['trk'])
            print(f'done in {time()-t0} sec')

        t0 = time()
        mni_fn = 'data/standard/MNI152_T1_1mm_brain.nii.gz'
        tck_mni_fn = tract2standard(tck_fn, cfg['t1'], mni_fn)
        print(f'done in {time()-t0} sec')

        t0 = time()
        print('convert warped tck to trk...')
        tck2trk(tck_mni_fn, mni_fn, out_fn=trk_fn)
        print(f'done in {time()-t0} sec')

    if not os.path.exists(trk_fn):
        print('The tractogram loaded is already compatible with the model')
        os.system(f'''ln -sf {cfg['trk']} {trk_fn}''')

    ## run inference
    print('launching inference...')
    exp = 'paper_runs/sdec_nodropout_loss_nll-data_hcp20_gt20mm_resampled16_fs8000_balanced_sampling_1'
    var = 'HCP20'

    cfg_parser = configparser.ConfigParser()
    cfg_parser.read(exp + '/config.txt')

    for name, value in cfg_parser.items('DEFAULT'):
        cfg[name] = get_cfg_value(value)
    for name, value in cfg_parser.items(var):
        cfg[name] = get_cfg_value(value)

    cfg['with_gt'] = False
    cfg['weights_path'] = ''
    cfg['exp_path'] = exp
    cfg['fixed_size'] = 10000

    dataset = TractDataset(trk_fn,
                           transform=None,
                           return_edges=True,
                           split_obj=True)

    dataloader = gDataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)


    classifier = get_model(cfg)

    if DEVICE == 'cuda':
        torch.cuda.set_device(DEVICE)
        torch.cuda.current_device()

    if cfg['weights_path'] == '':
        cfg['weights_path'] = glob.glob(cfg['exp_path'] + '/models/best*')[0]
    state = torch.load(cfg['weights_path'], map_location=DEVICE)

    classifier.load_state_dict(state)
    classifier.to(DEVICE)
    classifier.eval()

    preds = []
    probas = []
    with torch.no_grad():
        j = 0
        i = 0
        while j < len(dataset):
            t0 = time()
            print(f'processing subject {j}...')
            consumed = False
            data = dataset[j]
            obj_pred = np.zeros(data['obj_full_size'])
            obj_proba = np.zeros(data['obj_full_size'])
            obj_cls_embedding = np.zeros((data['obj_full_size'], 256))
            while not consumed:
                points = get_sample(data)
                batch = points.batch

                logits = classifier(points)

                pred = F.log_softmax(logits, dim=-1)
                pred_choice = pred.data.max(1)[1].int()

                obj_pred[data['obj_idxs']] = pred_choice.cpu().numpy()
                obj_proba[data['obj_idxs']] = F.softmax(
                    logits, dim=-1)[:, 0].cpu().numpy()

                if len(dataset.remaining[j]) == 0:
                    consumed = True
                    break
                data = dataset[j]
                i += 1

            preds.append(obj_pred)
            probas.append(obj_proba)

            j += 1
            print(f'done in {time()-t0} sec')

        ## save predictions
        out_dir = 'temp/output'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        print(f'saving predictions...')
        for pred in preds:
            idxs_P = np.where(pred == 1)[0]
            np.savetxt(f'{out_dir}/idxs_plausible.txt', idxs_P)
            idxs_nonP = np.where(pred == 0)[0]
            np.savetxt(f'{out_dir}/idxs_non-plausible.txt', idxs_nonP)
            if cfg['return_trk']:
                hdr = nib.streamlines.load(cfg['trk'], lazy_load=True).header
                streams, lengths = sload.load_selected_streamlines(cfg['trk'])
                streamlines = np.split(streams, np.cumsum(lengths[:-1]))
                streamlines = np.array(streamlines, dtype=np.object)[idxs_P]
                out_t = nib.streamlines.Tractogram(streamlines,
                                                   affine_to_rasmm=np.eye(4))
                out_t_fn = f'''{out_dir}/{cfg['trk'][:-4]}_filtered.trk)'''
                nib.streamlines.save(out_t, out_t_fn, header=hdr)
                print(f'saved {out_t_fn}')
        print(f'End')
        print(f'Duration: {(time()-t0_global)/60} min')

