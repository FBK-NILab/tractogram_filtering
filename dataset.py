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


#def load_h5_item(h5_filename, idx):
   # f = h5py.File(h5_filename)
   # data = f['data'][idx]
   # label = f['label'][idx]
   # return data, label


# Read numpy array data and label from h5_filename
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)



def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train_files.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train_files',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        #print(self.root)
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())
        #print(self.fns)


        self.lengths = []
        for fn in self.fns:
            #print(fn)
            nome_file = os.path.join(root,fn)
            #print(nome_file)
            f = h5py.File(nome_file)
            #self.lengths += len(f['label'][:])
            self.lengths.append(len(f['label'][:]))
        self.lengths = np.array(self.lengths)
        #print(self.lengths[:1].sum())



        
        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])



        self.classes = list(self.cat.keys())
        print(self.classes)


    
    
        

        self.final_data = np.ndarray([])
        self.final_label = np.ndarray([])
        
        for fn in self.fns:
            nome_file = os.path.join(root,fn)
            [data, label] = load_h5(nome_file)
            if not self.final_data.shape:
                self.final_data = data
                self.final_label = label
            else:
                self.final_data = np.concatenate((self.final_data,data))
                self.final_label = np.concatenate((self.final_label,label))

        

    
    def __getitem__(self, idx):
        
         

        '''
        for i, fn in enumerate(self.fns):
            #print(i)
            #print(fn)
            #print(idx)
            #print(self.lengths[:i+1].sum())
            if idx < self.lengths[:i+1].sum(): #if i=0 self.lengths[:i+1].sum() =    :i+1 = items from the beginning through stop-1
                print(self.lengths[:i+1].sum())
                #print('test')
                pts, cls = load_h5_item(fn, 
                                    idx - self.lengths[:i].sum())
                                    #self.lengths[:i].sum() - idx)
                                    #idx)
        '''
        pts = self.final_data[idx]
        cls = self.final_label[idx]    

        pts = torch.from_numpy(pts.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        #sample = {}
        #sample['points'] = pts
        #sample['class'] = cls

        return pts,cls


    def __len__(self):
        return self.lengths.sum()

class bsplineDataset(data.Dataset):
    
    def __init__(self,
                 root,
                 npoints=100,
                 split='train_files',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        self.label_fn = []
        #print(self.root)
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                line = line.strip()
                self.fns.append(os.path.join(root, 'derivatives/bspline_var-n100/sub-%s/bspline-subject-%s.npy' % (line,line) ))
                self.label_fn.append(os.path.join(root, 'derivatives/reduced_tract_labels/sub-%s/labels_%s_reduced.txt' % (line,line) ))
        #print(self.fns)
        #print(self.label_fn)

        self.lengths = []
        self.final_data = np.ndarray([])
        self.final_label = np.ndarray([])
        self.load_subject()
        
    def load_subject(self):
            
        for fn in self.fns:
            # TODO: use labels to retreive the size of the dataset;
            # in this case, you can do this operation directly inside the 
            # function __len__ 
                
            data = np.load(fn)
                
            if not self.final_data.shape:
                self.final_data = data
            else:
                self.final_data = np.concatenate((self.final_data,data))


        for fnl in self.label_fn:

            label = np.loadtxt(fnl)
                
            self.lengths.append(len(label))

            if not self.final_label.shape:
                    self.final_label = label
            else:
                self.final_label = np.concatenate((self.final_label,label))

        self.lengths = np.array(self.lengths)

         

        

         
        self.cat = {}
        with open(os.path.join(self.root, 'bspline_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])



        self.classes = list(self.cat.keys())
        print(self.classes)
    
    def __getitem__(self, idx):
        
        pts = self.final_data[idx]
        cls = self.final_label[idx]    

        pts = torch.from_numpy(pts.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return pts,cls


    def __len__(self):
        

        return self.lengths.sum()

class sl_paddingzeroDataset(data.Dataset):
    
    def __init__(self,
                 root,
                 npoints=100,
                 split='train_files',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        self.label_fn = []
        #print(self.root)
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                line = line.strip()
                self.fns.append(os.path.join(root, 'derivatives/sl_padding_zero/sub-%s/sl-padding-zero-subject-%s.npy' % (line.strip(),line.strip()) ))
                self.label_fn.append(os.path.join(root, 'derivatives/reduced_tract_labels/sub-%s/labels_%s_reduced.txt' % (line.strip(),line.strip()) ))
        #print(self.fns)
        #print(self.label_fn)

        self.lengths = []
        self.final_data = np.ndarray([])
        self.final_label = np.ndarray([])
        self.load_subject()
        
    def load_subject(self):
            
        for fn in self.fns:
            # TODO: use labels to retreive the size of the dataset;
            # in this case, you can do this operation directly inside the 
            # function __len__ 
                
            data = np.load(fn)
                
            if not self.final_data.shape:
                self.final_data = data
            else:
                self.final_data = np.concatenate((self.final_data,data))


        for fnl in self.label_fn:

            label = np.loadtxt(fnl)
                
            self.lengths.append(len(label))

            if not self.final_label.shape:
                    self.final_label = label
            else:
                self.final_label = np.concatenate((self.final_label,label))

        self.lengths = np.array(self.lengths)

         

        

         
        self.cat = {}
        with open(os.path.join(self.root, 'sl_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])



        self.classes = list(self.cat.keys())
        print(self.classes)
    
    def __getitem__(self, idx):
        
        pts = self.final_data[idx]
        cls = self.final_label[idx]    

        pts = torch.from_numpy(pts.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return pts,cls


    def __len__(self):
        

        return self.lengths.sum()


class sl_paddingrandomDataset(data.Dataset):
    
    def __init__(self,
                 root,
                 npoints=100,
                 split='train_files',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        self.label_fn = []
        #print(self.root)
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                line = line.strip()
                self.fns.append(os.path.join(root, 'derivatives/sl_padding_random/sub-%s/sl-padding-random-subject-%s.npy' % (line.strip(),line.strip()) ))
                self.label_fn.append(os.path.join(root, 'derivatives/reduced_tract_labels/sub-%s/labels_%s_reduced.txt' % (line.strip(),line.strip()) ))
        #print(self.fns)
        #print(self.label_fn)

        self.lengths = []
        self.final_data = np.ndarray([])
        self.final_label = np.ndarray([])
        self.load_subject()
        
    def load_subject(self):
            
        for fn in self.fns:
            # TODO: use labels to retreive the size of the dataset;
            # in this case, you can do this operation directly inside the 
            # function __len__ 
                
            data = np.load(fn)
                
            if not self.final_data.shape:
                self.final_data = data
            else:
                self.final_data = np.concatenate((self.final_data,data))


        for fnl in self.label_fn:

            label = np.loadtxt(fnl)
                
            self.lengths.append(len(label))

            if not self.final_label.shape:
                    self.final_label = label
            else:
                self.final_label = np.concatenate((self.final_label,label))

        self.lengths = np.array(self.lengths)

         

        

         
        self.cat = {}
        with open(os.path.join(self.root, 'sl_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])



        self.classes = list(self.cat.keys())
        print(self.classes)
    
    def __getitem__(self, idx):
        
        pts = self.final_data[idx]
        cls = self.final_label[idx]    

        pts = torch.from_numpy(pts.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return pts,cls


    def __len__(self):
        

        return self.lengths.sum()

class sl_paddingfrenetDataset(data.Dataset):
    
    def __init__(self,
                 root,
                 npoints=100,
                 split='train_files',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        self.label_fn = []
        #print(self.root)
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                line = line.strip()
                self.fns.append(os.path.join(root, 'derivatives/sl_padding_frenet/sub-%s/sl-padding-frenet-subject-%s.npy' % (line.strip(),line.strip()) ))
                self.label_fn.append(os.path.join(root, 'derivatives/reduced_tract_labels/sub-%s/labels_%s_reduced.txt' % (line.strip(),line.strip()) ))
        #print(self.fns)
        #print(self.label_fn)

        self.lengths = []
        self.final_data = np.ndarray([])
        self.final_label = np.ndarray([])
        self.load_subject()
        
    def load_subject(self):
            
        for fn in self.fns:
            # TODO: use labels to retreive the size of the dataset;
            # in this case, you can do this operation directly inside the 
            # function __len__ 
                
            data = np.load(fn)
                
            if not self.final_data.shape:
                self.final_data = data
            else:
                self.final_data = np.concatenate((self.final_data,data))


        for fnl in self.label_fn:

            label = np.loadtxt(fnl)
                
            self.lengths.append(len(label))

            if not self.final_label.shape:
                    self.final_label = label
            else:
                self.final_label = np.concatenate((self.final_label,label))

        self.lengths = np.array(self.lengths)

         

        

         
        self.cat = {}
        with open(os.path.join(self.root, 'sl_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])



        self.classes = list(self.cat.keys())
        print(self.classes)
    
    def __getitem__(self, idx):
        
        pts = self.final_data[idx]
        cls = self.final_label[idx]    

        pts = torch.from_numpy(pts.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return pts,cls


    def __len__(self):
        

        return self.lengths.sum()
   
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
        if self.act:
            tract_type = '500000'
        else:
            tract_type = '500000_noACT'

        sub = self.subjects[idx]
        sub_dir = os.path.join(self.root_dir, 'sub-%s' % sub)
        T_file = os.path.join(sub_dir, 'All_%s.trk' % (tract_type))
        label_file = os.path.join(sub_dir, 'All_%s_gt.pkl' % (tract_type))
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
            edges = torch.tensor([range(0, l_sl-1) + range(1,l_sl),
                                range(1,l_sl) + range(0, l_sl-1)],
                                dtype=torch.long)
            data.append(gData(x=sl, edge_index=edges, y=sample['gt'][i]))
        #gt = torch.from_numpy(sample['gt'])
        #graph_sample = gData(x=sample_flat, edge_index=edges, y=gt)
        sample['points'] = gBatch().from_data_list(data)
        sample['name'] = 'sub-%s_var-%s_tract' % (sub, tract_type)
        #print('time building graph %f' % (time.time()-t0))
        return sample


if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    '''
    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(),seg.type())

        d = ShapeNetDataset(root = datapath, classification = True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)
    '''

    if dataset == 'modelnet':
        #print('test')
        #gen_modelnet_id(datapath)
        #print(datapath)

        d = ModelNetDataset(root=datapath)
        #print('test')
        print(len(d))
        print(d[0])

    if dataset == 'bspline':
        #print('test')
        #gen_modelnet_id(datapath)
        #print(datapath)

        d = bsplineDataset(root=datapath)
        #print('test')
        print(len(d))
        print(d[1472115])

