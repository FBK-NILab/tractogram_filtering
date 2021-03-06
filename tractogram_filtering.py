#!/usr/bin/env python

from __future__ import print_function

import argparse
import configparser
import glob
import json
import os
from os import path as osp
from os.path import basename as osbn
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
from utils.data.transforms import TestSampling, SeqSampling
from utils.general_utils import get_cfg_value
from utils.model_utils import get_model

import subprocess

# os.environ["DEVICE"] = torch.device(
#     'cuda' if torch.cuda.is_available() else 'cpu')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(10)


def get_gpu_free_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are free memory as integers in MB.
    """
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
    ],
                                     encoding='utf-8')
    # Convert lines into a dictionary
    gpu_used_memory = [int(x) for x in result.strip().split('\n')]
    n_gpus = len(gpu_used_memory)
    gpu_free_memory = []
    for i in range(n_gpus):
        tot_mem = torch.cuda.get_device_properties(i).total_memory
        tot_mem = int(tot_mem / 1024**2)
        gpu_free_memory.append(tot_mem - gpu_used_memory[i])

    gpu_free_memory_map = dict(zip(range(n_gpus), gpu_free_memory))
    return gpu_free_memory_map


def get_max_batchsize(curr_device):
    free_mem = int(get_gpu_free_memory_map()[curr_device] / 1024)  # in GB
    if free_mem <= 4:
        return 10000
    elif free_mem <= 8:
        return 20000
    elif free_mem <= 10:
        return 30000
    elif free_mem <= 11:
        return 35000
    elif free_mem >= 12:
        return 40000


def tract2standard(t_fn,
                   t1_fn,
                   fixed_fn,
                   trans_type='SyNRA'):
    print(f'registration using ANTs {trans_type}...')
    fixed = ants.image_read(fixed_fn)
    moving = ants.image_read(t1_fn)

    # this is a workaround to emulate antsRegistrationSyNQuick.sh.
    # Unfortunately it is not possible to equally emulate the script.
    # There are differences in terms of parameters (shrink factor and num of
    # iterations) in the rigid and in the affine registration
    if trans_type == 'SyNRA':
        # values taken from https://github.com/ANTsX/ANTs/blob/952e7918b47385ebfb730f9c844977762b8437f8/Scripts/antsRegistrationSyNQuick.sh#L455
        # Notes:
        # 1. syn_metric and num_of_bins (syn_sampling) are the same as default:
        # "mattes" and 32 respectively
        # 2. the three values that configure the SyN[x,x,x] optimization are
        # respectively grad_step, flow_sigma, and total_sigma
        # 3. syn_iterations correspond to reg_iterations
        # 4. smoothing sigmas and shrink factor are automatically set inside the
        # function. As desired they are set to be: "3x2x1x0vox" and "8x4x2x1"
        # respectively
        mytx = ants.registration(fixed=fixed,
                                moving=moving,
                                type_of_transform=trans_type,
                                reg_iterations=(100,70,50,0),
                                grad_step=0.1,
                                flow_sigma=3,
                                total_sigma=0)
    else:
        mytx = ants.registration(fixed=fixed,
                                moving=moving,
                                type_of_transform=trans_type)

    ants.image_write(mytx['warpedmovout'], f'{tmp_dir}/struct_warped.nii.gz')

    print('correcting warp to mrtrix convention...')
    os.system(f'warpinit {fixed_fn} {tmp_dir}/ID_warp[].nii.gz -force')

    for i in range(3):
        temp_warp = ants.image_read(f'{tmp_dir}/ID_warp{i}.nii.gz')
        temp_warp = ants.apply_transforms(fixed=moving,
                                          moving=temp_warp,
                                          transformlist=mytx['invtransforms'],
                                          whichtoinvert=[True, False],
                                          defaultvalue=2147483647
                                          )
        ants.image_write(temp_warp, f'{tmp_dir}/mrtrix_warp{i}.nii.gz')

    os.system(
        f'warpcorrect {tmp_dir}/mrtrix_warp[].nii.gz ' + 
        f'{tmp_dir}/mrtrix_warp_cor.nii.gz ' + 
        '-marker 2147483647 -tolerance 0.0001 -force'
    )

    print('applaying warp to tractogram...')
    t_mni_fn = t_fn[:-4] + '_mni.tck'
    os.system(
        f'tcktransform {t_fn} {tmp_dir}/mrtrix_warp_cor.nii.gz {t_mni_fn} ' +
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

    script_dir = osp.dirname(osp.realpath(__file__))
    tmp_dir = 'tmp_tractogram_filtering'

    parser = argparse.ArgumentParser()
    parser.add_argument('-config',
                        nargs='?',
                        default=f'{script_dir}/run_config.json',
                        help='The tag for the configuration file.')
    args = parser.parse_args()

    ## load config
    t0_global = time()
    print('reading arguments')
    cfg = json.load(open(args.config))
    print(cfg)

    move_tract = cfg['t1'] or cfg['fa'] != ''
    tck_fn = f'{tmp_dir}/input/tract.tck'
    trk_fn = f'{tmp_dir}/input/tract_mni_resampled.trk'

    in_dir = f'{tmp_dir}/input'
    if not osp.exists(in_dir):
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
        if not osp.exists(tck_fn):
            t0 = time()
            print('convert trk to tck...')
            trk2tck(cfg['trk'], out_fn=tck_fn)
            print(f'done in {time()-t0} sec')

        t0 = time()
        if cfg['fa'] != '':
            struct_fn = cfg['fa']
            mni_fn = f'{script_dir}/data/standard/FSL_HCP1065_FA_1mm.nii.gz'
        else:
            struct_fn = cfg['t1']
            mni_fn = f'{script_dir}/data/standard/MNI152_T1_1mm_brain.nii.gz'

        if cfg['fast_warp']:
            # TODO: uncomment following line when antspy will release precompiled > 0.2.6
            # warp_type = 'antsRegistrationSyNQuick[s]'
            warp_type = 'SyNRA'
        else:
            warp_type = 'SyNCC'

        tck_mni_fn = tract2standard(tck_fn,
                                    struct_fn,
                                    mni_fn,
                                    trans_type=warp_type)
        print(f'done in {time()-t0} sec')

        t0 = time()
        print('convert warped tck to trk...')
        tck2trk(tck_mni_fn, mni_fn, out_fn=trk_fn)
        print(f'done in {time()-t0} sec')

    if not osp.exists(trk_fn):
        print('The tractogram loaded is already compatible with the model')
        os.system(f'''ln -sf {cfg['trk']} {trk_fn}''')

    ## run inference
    print(f'launching inference using {DEVICE}...')
    exp = f'{script_dir}/paper_runs/sdec_nodropout_loss_nll-data_hcp20_gt20mm_resampled16_fs8000_balanced_sampling_1'
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

    # this prevent errors in loading streamlines that miss out of bounds points,
    # removed by tcktranform during warping. Setting same_size to false we lose
    # a bit of loading speed, but we are sure we load correctly streamlines
    cfg['same_size'] = False

    # check available memory to decide how many streams sample
    curr_device = torch.cuda.current_device()
    cfg['fixed_size'] = get_max_batchsize(curr_device)

    dataset = TractDataset(trk_fn,
                           transform=TestSampling(cfg['fixed_size']),
                           return_edges=True,
                           split_obj=True,
                           same_size=cfg['same_size'])

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
        out_dir = f'{tmp_dir}/output'
        if not osp.exists(out_dir):
            os.makedirs(out_dir)

        print(f'saving predictions...')
        for pred in preds:
            idxs_P = np.where(pred == 1)[0]
            np.savetxt(f'{out_dir}/idxs_plausible.txt', idxs_P, fmt='%d')
            idxs_nonP = np.where(pred == 0)[0]
            np.savetxt(f'{out_dir}/idxs_non-plausible.txt',
                       idxs_nonP,
                       fmt='%d')
            if cfg['return_trk']:
                hdr = nib.streamlines.load(cfg['trk'], lazy_load=True).header
                streams, lengths = sload.load_selected_streamlines(cfg['trk'])
                streamlines = np.split(streams, np.cumsum(lengths[:-1]))
                streamlines = np.array(streamlines, dtype=np.object)[idxs_P]
                out_t = nib.streamlines.Tractogram(streamlines,
                                                   affine_to_rasmm=np.eye(4))
                out_t_name = osbn(cfg['trk'])[:-4] + '_filtered.trk'
                out_t_fn = f'''{out_dir}/{out_t_name}'''
                nib.streamlines.save(out_t, out_t_fn, header=hdr)
                print(f'saved {out_t_fn}')
        print(f'End')
        print(f'Duration: {(time()-t0_global)/60} min')
