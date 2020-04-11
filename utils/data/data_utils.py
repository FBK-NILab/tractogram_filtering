import torch
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader 
from torch_geometric.data import Batch as gBatch
from torch_geometric.data import DataListLoader as gDataLoader

import datasets as ds
from .transforms import *

def get_dataset(cfg, trans, split_obj=False, train=True):
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
                             shuffle=False,
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
