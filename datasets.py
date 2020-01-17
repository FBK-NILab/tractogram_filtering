from __future__ import print_function
from scipy.spatial import distance
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
from torch_geometric.transforms import Distance, AddSelfLoops

from torch_geometric.data import Data as gData, Batch as gBatch
from torch_geometric.data import Dataset as gDataset
from torch_geometric.nn import knn_graph
from torch_geometric.utils import remove_self_loops
from selective_loader import load_selected_streamlines,load_selected_streamlines_uniform_size
from nilab.load_trk import load_streamlines
from dipy.tracking.streamline import length


class HCP20Dataset(gDataset):
    def __init__(self,
                 sub_file,
                 root_dir,
                 #k=4,
                 act=True,
                 fold_size=None,
                 transform=None,
                 distance=None,
                 self_loops=None,
                 with_gt=True,
                 return_edges=False,
                 split_obj=False,
                 train=True,
                 load_one_full_subj=True):
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
        #self.k = k
        self.transform = transform
        self.distance = distance
        self.fold_size = fold_size
        self.act = act
        self.self_loops = self_loops
        self.with_gt = with_gt
        self.return_edges = return_edges
        self.fold = []
        self.n_fold = 0
        self.train = train
        self.load_one_full_subj = load_one_full_subj
        if fold_size is not None:
            self.load_fold()
        if train:
            split_obj=False
        if split_obj:
            self.remaining = [[] for _ in range(len(subjects))]
        self.split_obj = split_obj
        if self.load_one_full_subj:
            print('loading one full subject')
            #sub = subjects[0]
            sub = self.subjects[0]
            print(sub)
            sub_dir = os.path.join(self.root_dir, 'sub-%s' % sub)
            print(sub_dir)
            T_file = os.path.join(sub_dir, 'sub-%s_var-GIN_full_tract.trk' % (sub))
            #T_file = glob.glob('%s/*GIN_full_tract.trk' % sub_dir)[0]
            print(T_file)
            T = nib.streamlines.load(T_file, lazy_load=True)
            hdr = nib.streamlines.load(T_file, lazy_load=True).header
            idxs = np.arange(hdr['nb_streamlines']).tolist()
            streams, lengths = load_selected_streamlines(T_file, idxs)
            cum_lengths = np.concatenate([[0],lengths]).cumsum()
            streamlines = [
                streams[cl:cl + lengths[i]] for i, cl in enumerate(cum_lengths[:-1])
            ]
            if with_gt:
                label_file = os.path.join(sub_dir,
                                          'sub-%s_var-GIN_labels.pkl' % (sub))
                with open(label_file, 'rb') as f:
                    gts = pickle.load(f)
                    gts = torch.tensor(gts).long()
            else:
                gts = None

            assert split_obj == True
            self.remaining[0] = np.arange(len(streamlines)).tolist()
            name = T_file.split('/')[-1].rsplit('.', 1)[0]
            self.full_subj = (streamlines, lengths, gts, name, sub_dir)


    def __len__(self):
        if self.load_one_full_subj:
            return len(self.full_subj[0])
        return len(self.subjects)

    def __getitem__(self, idx):
        if self.load_one_full_subj:
            return self.get_one_streamline()
        fs = self.fold_size
        if fs is None:
            #print(self.subjects[idx])
            #t0 = time.time()
            item = self.getitem(idx)
            #print('get item time: {}'.format(time.time()-t0))
            return item

        fs_0 = (self.n_fold * fs)
        idx = fs_0 + (idx % fs)

        return self.data_fold[idx]

    def get_one_streamline(self):
        while len(self.remaining[0]) > 0:
            idx = self.remaining[0][0]
            print(idx)
            self.remaining[0] = self.remaining[0][1:]

            l = self.full_subj[1][idx]
            stream = self.full_subj[0][idx]
            gt = self.full_subj[2][idx]
        #gt = torch.tensor(self.full_subj[2][idx])

            gsample = self.build_graph_sample(
                stream, [l], gt)

            sample = {'points': gsample, 'gt': gt}
        #sample['points'] = self.build_graph_sample(stream, [l], gt)
            sample['obj_idxs'] = [idx]
            sample['obj_full_size'] = len(self)
            sample['name'] = self.full_subj[3]
            sample['dir'] = self.full_subj[4]

            return sample

    def load_fold(self):
        print('loading fold')
        fs = self.fold_size
        fs_0 = self.n_fold * fs
        #t0 = time.time()
        print('Loading fold')
        self.data_fold = [self.getitem(i) for i in range(fs_0, fs_0 + fs)]
        #print('time needed: %f' % (time.time()-t0))

    def getitem(self, idx):
        sub = self.subjects[idx]
        #t0 = time.time()
        sub_dir = os.path.join(self.root_dir, 'sub-%s' % sub)
        T_file = os.path.join(sub_dir, 'sub-%s_var-GIN_full_tract.trk' % (sub))
        label_sub_dir = os.path.join(self.root_dir.rsplit('/',1)[0], 'merge_shuffle_trk' ,'sub-%s' % sub)
        label_file = os.path.join(label_sub_dir, 'sub-%s_var-GIN_labels.pkl' % (sub))
        T = nib.streamlines.load(T_file, lazy_load=True)
        #print('\tload lazy T %f' % (time.time()-t0))
        #t0 = time.time()
        #gt = np.load(label_file)
        with open(label_file, 'rb') as f:
            gt = pickle.load(f)
        gt = np.array(gt)
        #print('\tload gt %f' % (time.time()-t0))
        if self.split_obj:
            if len(self.remaining[idx]) == 0:
                self.remaining[idx] = set(np.arange(T.header['nb_streamlines']))
            sample = {'points': np.array(list(self.remaining[idx]))}
            if self.with_gt:
                sample['gt'] = gt[list(self.remaining[idx])]
        else:
            #sample = {'points': np.arange(T.header['nb_streamlines'])}
            #if self.with_gt:
            #sample['gt'] = gt
            sample = {'points': np.arange(T.header['nb_streamlines']), 'gt': gt}
        #print(sample['name'])

        #t0 = time.time()
        if self.transform:
            sample = self.transform(sample)
        #print('\ttime sampling %f' % (time.time()-t0))

        if self.split_obj:
            self.remaining[idx] -= set(sample['points'])
            sample['obj_idxs'] = sample['points'].copy()
            sample['obj_full_size'] = T.header['nb_streamlines']
            #sample['streamlines'] = T.streamlines

        sample['name'] = T_file.split('/')[-1].rsplit('.', 1)[0]
        sample['dir'] = sub_dir

        n = len(sample['points'])
        #t0 = time.time()
        uniform_size = False
        if uniform_size:
            streams, l_max = load_selected_streamlines_uniform_size(T_file,
                                                    sample['points'].tolist())
            streams.reshape(n, l_max, -1)
            sample['points'] = torch.from_numpy(streams)
        else:
            #streams, _, lengths, _ = load_streamlines(T_file,
            #                                        idxs=sample['points'].tolist(),
            #                                        container='array_flat')
            streams, lengths = load_selected_streamlines(T_file,
                                                    sample['points'].tolist())
        #print('\ttime loading selected streamlines %f' % (time.time()-t0))

        #t0 = time.time()
        sample['points'] = self.build_graph_sample(streams,
                    lengths,
                    torch.from_numpy(sample['gt']) if self.with_gt else None)
        #sample['tract'] = streamlines
        #print('sample:',sample['points'])
        #print('\ttime building graph %f' % (time.time()-t0))
        return sample

    def build_graph_sample(self, streams, lengths, gt=None):
        #t0 = time.time()
        #print('time numpy split %f' % (time.time()-t0))
        ### create graph structure
        #sls_lengths = torch.from_numpy(sls_lengths)
        lengths = torch.from_numpy(np.array(lengths))
        #print('sls lengths:',sls_lengths)
        batch_vec = torch.arange(len(lengths)).repeat_interleave(lengths)
        batch_slices = torch.cat([torch.tensor([0]), lengths.cumsum(dim=0)])
        slices = batch_slices[1:-1]
        streams = torch.from_numpy(streams)
        l = streams.shape[0]
        graph_sample = gData(x=streams,
                             lengths=lengths,
                             #sls_lengths=sls_lengths,
                             bvec=batch_vec,
                             pos=streams)
        #                     bslices=batch_slices)
        #edges = torch.empty((2, 2*l - 2*n), dtype=torch.long)
        if self.return_edges:
            e1 = set(np.arange(0,l-1)) - set(slices.numpy()-1)
            e2 = set(np.arange(1,l)) - set(slices.numpy())
            edges = torch.tensor([list(e1)+list(e2),list(e2)+list(e1)],
                            dtype=torch.long)
            #print('old edges:', edges)
            #print(edges[0,-1])
            #print(edges[0,:int(edges.shape[1]/2)+2])
            #e1_new = torch.repeat_interleave(edges[0,:int(edges.shape[1]/2)],self.k)
            #e1_new=torch.cat((e1_new,torch.repeat_interleave(edges[0,-1],self.k)))
            #e2_new = torch.tensor([],dtype=torch.long)
            #for i in list(edges[0,:int(edges.shape[1]/2)]):
            #if i == 0:
            #e2_new = torch.cat([e2_new,torch.arange(i+1,self.k+1)],dim=0)
            #if i==1:
            #e2_new = torch.cat([e2_new,torch.cat([torch.tensor([i-1]),torch.arange(i+1,self.k+1)],dim=0)])
            #if i<self.k/2 and i>1:
            #e2_new = torch.cat([e2_new,torch.cat([torch.arange(0,i),torch.arange(i+1,i+(self.k-i)+1)],dim=0)])
            #if i>=self.k/2 and i!=edges[0,int(edges.shape[1]/2)-1]:
            #if i+self.k/2 > edges[0,-1]:
            #e2_new = torch.cat([e2_new,torch.cat([torch.arange(i-(self.k-(edges[0,-1]-i)),i),torch.arange(i+1,edges[0,-1]+1)],dim=0)])
            #else:
            #e = torch.cat([torch.arange(i-self.k/2,i),torch.arange(i+1,i+self.k/2+1)])
            #e = e.long()
            #print(e)
            #e2_new = torch.cat([e2_new,e])
            #e2 = torch.cat([e2,torch.cat([torch.arange(i-self.k/2,i),torch.arange(i+1,i+self.k/2+1)],dim=0)])
            #if i==edges[0,int(edges.shape[1]/2)-1]:
            #e2_new = torch.cat([e2_new,torch.cat([torch.arange(i-1,i-self.k,-1),torch.tensor([i+1])])])
            #e2_new = torch.cat([e2_new,torch.arange(edges[0,-1]-1,(edges[0,-1]-1)-self.k, -1)],dim=0)
            #e1, e2 = e1.cuda(), e2.cuda()
            #edges_new = torch.stack((e1_new,e2_new),0)
            #print('new edges:',edges_new)
            graph_sample['edge_index'] = edges
            num_edges = graph_sample.num_edges
            edge_attr = torch.ones(num_edges,1)
            graph_sample['edge_attr'] = edge_attr
        if self.distance:
            graph_sample = self.distance(graph_sample)
        #if self.self_loops:
        #graph_sample = self.self_loops(graph_sample)
        if gt is not None:
            graph_sample['y'] = gt

        return graph_sample


class RndSampling(object):
    """Random sampling from input object to return a fixed size input object
    Args:
        output_size (int): Desired output size.
        maintain_prop (bool): Default True. Indicates if the random sampling
            must be proportional to the number of examples of each class
    """

    def __init__(self, output_size, maintain_prop=True, prop_vector=[]):
        assert isinstance(output_size, (int))
        assert isinstance(maintain_prop, (bool))
        self.output_size = output_size
        self.maintain_prop = maintain_prop
        self.prop_vector = prop_vector

    def __call__(self, sample):
        np.random.seed(np.uint32(torch.initial_seed()))

        pts, gt = sample['points'], sample['gt']

        n = pts.shape[0]
        if self.maintain_prop:
            n_classes = gt.max() + 1
            remaining = self.output_size
            chosen_idx = []
            for cl in reversed(range(n_classes)):
                if (gt == cl).sum() == 0:
                    continue
                if cl == gt.min():
                    chosen_idx += np.random.choice(
                        np.argwhere(gt == cl).reshape(-1),
                        int(remaining)).reshape(-1).tolist()
                    break
                prop = float(np.sum(gt == cl)) / n
                k = np.round(self.output_size * prop)
                remaining -= k
                chosen_idx += np.random.choice(
                    np.argwhere(gt == cl).reshape(-1),
                    int(k)).reshape(-1).tolist()

            assert (self.output_size == len(chosen_idx))
            chosen_idx = np.array(chosen_idx)
        elif len(self.prop_vector) != 0:
            n_classes = gt.max() + 1
            while len(self.prop_vector) < n_classes:
                self.prop_vector.append(1)
            remaining = self.output_size
            out_size = self.output_size
            chosen_idx = []
            excluded = 0
            for cl in range(n_classes):
                if (gt == cl).sum() == 0:
                    continue
                if cl == gt.max():
                    chosen_idx += np.random.choice(
                        np.argwhere(gt == cl).reshape(-1),
                        int(remaining)).reshape(-1).tolist()
                    break
                if self.prop_vector[cl] != 1:
                    prop = self.prop_vector[cl]
                    excluded += np.sum(gt == cl)
                    #excluded -= (n-excluded)*prop
                    k = np.round(self.output_size * prop)
                    out_size = remaining - k
                else:
                    prop = float(np.sum(gt == cl)) / (n - excluded)
                    k = np.round(out_size * prop)

                remaining -= k
                chosen_idx += np.random.choice(
                    np.argwhere(gt == cl).reshape(-1),
                    int(k)).reshape(-1).tolist()

            assert (self.output_size == len(chosen_idx))
            chosen_idx = np.array(chosen_idx)
        else:
            chosen_idx = np.random.choice(range(n), self.output_size)
        
        out_gt = gt[chosen_idx] if len(gt) > 1 else gt
        return {'points': pts[chosen_idx], 'gt': out_gt}


class TestSampling(object):
    """Random sampling from input object until the object is all sampled
    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        self.output_size = output_size

    def __call__(self, sample):
        sl = sample['points']

        n = sl.shape[0]
        if self.output_size > len(range(n)):
            chosen_idx = range(n)
        else:
            chosen_idx = np.random.choice(range(n), self.output_size).tolist()
        out_sample = {'points': sl[chosen_idx]}

        if 'gt' in sample.keys():
            out_sample['gt'] = sample['gt'][chosen_idx]
        return out_sample
