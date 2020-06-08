import torch
from dipy.tracking.streamlinespeed import set_number_of_points
from nibabel.orientations import aff2axcodes
from nibabel.streamlines.trk import Field
from torch.utils.data.dataloader import DataLoader
from torch_geometric.data import Batch as gBatch
from torch_geometric.data import DataListLoader as gDataLoader
from torchvision import transforms

import datasets as ds

from .transforms import *


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
                              same_size=cfg['same_size'],
                              transform=transforms.Compose(trans),
                              return_edges=cfg['return_edges'],
                              load_one_full_subj=False,
                              labels_dir=cfg['labels_dir'])

    dataloader = gDataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffling,
                             num_workers=int(cfg['n_workers']),
                             pin_memory=True)

    print("Dataset %s loaded, found %d samples" %
          (cfg['dataset'], len(dataset)))
    return dataset, dataloader

def get_transforms(cfg, train=True):
    trans = []
    #if cfg['pc_centering']:
    #    trans.append(PointCloudCentering())
    #if cfg['pc_normalization']:
    #    trans.append(PointCloudNormalization())
    if cfg['rnd_sampling']:
        if train:
            trans.append(RndSampling(cfg['fixed_size'], maintain_prop=False))
        else:
            trans.append(FixedRndSampling(cfg['fixed_size']))
    #if train and cfg['pc_rot']:
    #    trans.append(PointCloudRotation())
    #if train and cfg['pc_jitter']:
    #    trans.append(PointCloudJittering(cfg['pc_jitter']))

    print(trans)
    return trans

def get_gbatch_sample(sample, sample_size, same_size, return_name=False):
    data_list = []
    name_list = []
    ori_batch = []
    for i, d in enumerate(sample):
        if 'bvec' in d['points'].keys:
            d['points'].bvec += sample_size * i
        data_list.append(d['points'])
        name_list.append(d['name'])
        ori_batch.append([i] * sample_size)
    points = gBatch().from_data_list(data_list)
    points.ori_batch = torch.tensor(ori_batch).flatten().long()
    if 'bvec' in points.keys:
        #points.batch = points.bvec.copy()
        points.batch = points.bvec.clone()
        del points.bvec
    if same_size:
        points['lengths'] = points['lengths'][0].item()

    if return_name:
        return points, name_list
    return points


def resample_streamlines(streamlines, n_pts=16):
    resampled = []
    for sl in streamlines:
        resampled.append(set_number_of_points(sl, n_pts))

    return resampled


def tck2trk(tck_fn, nii_fn, out_fn=None):
    nii = nib.load(nii_fn)
    header = {}
    header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
    header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
    header[Field.DIMENSIONS] = nii.shape[:3]
    header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))

    tck = nib.streamlines.load(tck_fn)
    out_fn = tck_fn[:-4] + '.trk' if out_fn is None else out_fn
    nib.streamlines.save(tck.tractogram, out_fn, header=header)


def trk2tck(trk_fn):
    trk = nib.streamlines.load(trk_fn)
    nib.streamlines.save(trk.tractogram, trk_fn[:-4] + '.tck')
