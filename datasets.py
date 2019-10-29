from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import h5py

from torch.utils.data import Dataset
import csv
import pickle
import nibabel as nib
import glob
import time

from torch_geometric.data import Data as gData, Batch as gBatch
from torch_geometric.data import Dataset as gDataset
from torch_geometric.nn import knn_graph

from selective_loader import load_selected_streamlines,load_selected_streamlines_uniform_size

class HCP20Dataset(gDataset):
    def __init__(self,
                 sub_file,
                 root_dir,
                 act=True,
                 fold_size=None,
                 transform=None):
        """
        Args:
            root_dir (string): root directory of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        with open(sub_file) as f:
            subjects = f.readlines()
        self.subjects = [s.strip() for s in subjects]
        self.transform = transform
        self.fold_size = fold_size
        self.act = act
        self.fold = []
        self.n_fold = 0
        if fold_size is not None:
            self.load_fold()

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        fs = self.fold_size
        if fs is None:
            return self.getitem(idx)

        fs_0 = (self.n_fold * fs)
        idx = fs_0 + (idx % fs)

        return self.data_fold[idx]

    def load_fold(self):
        fs = self.fold_size
        fs_0 = self.n_fold * fs
        t0 = time.time()
        print('Loading fold')
        self.data_fold = [self.getitem(i) for i in range(fs_0, fs_0 + fs)]
        print('time needed: %f' % (time.time()-t0))

    def getitem(self, idx):
        sub = self.subjects[idx]
        sub_dir = os.path.join(self.root_dir, 'sub-%s' % sub)
        T_file = os.path.join(sub_dir, 'sub-%s_var-HCP_full_tract.trk' % (sub))
        label_file = os.path.join(sub_dir, 'sub-%s_var-HCP_labels.pkl' % (sub))
        #T_file = os.path.join(sub_dir, 'All_%s.trk' % (tract_type))
        #label_file = os.path.join(sub_dir, 'All_%s_gt.pkl' % (tract_type))
        T = nib.streamlines.load(T_file, lazy_load=True)
        with open(label_file, 'rb') as f:
            gt = pickle.load(f)

        sample = {'points': np.arange(T.header['nb_streamlines']), 'gt': gt}

        #t0 = time.time()
        if self.transform:
            sample = self.transform(sample)
        #print('time sampling %f' % (time.time()-t0))

        #t0 = time.time()
        streams, slices = load_selected_streamlines(T_file,
                                                    sample['points'].tolist())
        #print('time loading selected streamlines %f' % (time.time()-t0))
        #t0 = time.time()
        sample['points'] = np.split(streams, slices[:-1], axis=0)
        #print('time numpy split %f' % (time.time()-t0))
        ### create graph structure
        # n = len(sample['points'])
        #sample_flat = torch.from_numpy(np.concatenate(sample['points']))
        #l = sample_flat.shape[0]
        #edges = torch.empty((2, 2*l - 2*n), dtype=torch.long)
        #start = 0
        #i0 = 0
        #for sl in sample['points']:
        #    l_sl = len(sl)
        #    end = start + 2*l_sl - 2
        #    edges[:, start:end] = torch.tensor(
        #                        [range(i0, i0+l_sl-1) + range(i0+1,i0+l_sl),
        #                        range(i0+1,i0+l_sl) + range(i0, i0+l_sl-1)])
        #    start = end
        #    i0 += l_sl
        #t0 = time.time()
        data = []
        for i, sl in enumerate(sample['points']):
            l_sl = len(sl)
            sl = torch.from_numpy(sl)
            edges = torch.tensor([list(range(0, l_sl-1)) + list(range(1,l_sl)),
                                list(range(1,l_sl)) + list(range(0, l_sl-1))],
                                dtype=torch.long)
            data.append(gData(x=sl, edge_index=edges, y=sample['gt'][i]))
        #gt = torch.from_numpy(sample['gt'])
        #graph_sample = gData(x=sample_flat, edge_index=edges, y=gt)
        #sample['points'] = gBatch().from_data_list(data)
        sample['points'] = data
        sample['name'] = 'sub-%s_var-HCP_full_tract' %(sub)
        #print('time building graph %f' % (time.time()-t0))
        return sample
