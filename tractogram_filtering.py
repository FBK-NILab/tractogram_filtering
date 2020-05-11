from __future__ import print_function
import os
import numpy as np
import ants
import nibabel as nib
import torch
import torch.nn.functional as F
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes
from nibabel.streamlines.trk import get_affine_trackvis_to_rasmm
from nibabel.affines import apply_affine
from torch_geometric.data import Data as gData, Batch as gBatch
from torch_geometric.data import Dataset as gDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import Sequential as Seq
from torch.nn import ReLU
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear as Lin
from torch_geometric.nn import EdgeConv, DynamicEdgeConv, global_max_pool
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram 
from dipy.io.streamline import save_tractogram
from dipy.tracking.streamline import set_number_of_points

class TractData(gDataset):
    def __init__(self, transform=None,return_edges=True,split_obj=True):
        self.transform = transform
        self.return_edges = return_edges
        if split_obj:
            self.remaining = [[]]
        self.split_obj = split_obj

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        item = self.getitem(idx)
        return item

    def getitem(self,idx):
        #T_file = $new_resampled_name.trk
        T_file = 'sub-105115_var-HCP_full_tract_SUB2mni_resampled.trk'
        T = nib.streamlines.load(T_file, lazy_load=True)
        if self.split_obj:
            if len(self.remaining[idx])==0:
                self.remaining[idx] = set(np.arange(T.header['nb_streamlines']))
            sample = {'points': np.array(list(self.remaining[idx]))}
        if self.transform:
            sample = self.transform(sample)
        if self.split_obj:
            self.remaining[idx] -= set(sample['points'])
            sample['obj_idxs'] = sample['points'].copy()
            sample['obj_full_size'] = T.header['nb_streamlines']
        n = len(sample['points'])
        streams, lengths = load_selected_streamlines_uniform_size(T_file, sample['points'].tolist())
        sample['points'] = build_graph_sample(streams,lengths,gt=None)
        return sample

def build_graph_sample(streams, lengths, gt=None):
    lengths = torch.from_numpy(lengths).long()
    batch_vec = torch.arange(len(lengths)).repeat_interleave(lengths)
    batch_slices = torch.cat([torch.tensor([0]), lengths.cumsum(dim=0)])
    slices = batch_slices[1:-1]
    streams = torch.from_numpy(streams)
    l = streams.shape[0]
    graph_sample = gData(x=streams,lengths=lengths,bvec=batch_vec,pos=streams)
    e1 = set(np.arange(0,l-1))- set(slices.numpy()-1)
    e2 = set(np.arange(1,l)) - set(slices.numpy())
    edges = torch.tensor([list(e1)+list(e2),list(e2)+list(e1)],dtype=torch.long)
    graph_sample['edge_index'] = edges
    num_edges = graph_sample.num_edges
    edge_attr = torch.ones(num_edges,1)
    graph_sample['edge_attr'] = edge_attr
    return graph_sample

def load_selected_streamlines_uniform_size(trk_fn, idxs=None):
    lazy_trk = nib.streamlines.load(trk_fn,lazy_load=True)
    header = lazy_trk.header
    header_size = header['hdr_size']
    nb_streamlines = header['nb_streamlines']
    n_scalars = header['nb_scalars_per_point']
    n_properties = header['nb_properties_per_streamline']
    if idxs is None:
        idxs = np.arange(nb_streamlines)
    length_bytes = 4
    point_size = 3+n_scalars
    point_size = 3+n_scalars
    point_bytes = 4*point_size
    properties_bytes = n_properties*4
    with open(trk_fn,'rb') as f:
        f.seek(header_size)
        l = np.fromfile(f, np.int32, 1)[0]
    lengths = np.array([l] * nb_streamlines).astype(np.int)
    index_bytes = lengths * point_bytes + properties_bytes + length_bytes
    index_bytes = np.concatenate([[length_bytes], index_bytes[:-1]]).cumsum() + header_size
    n_floats = lengths * point_size
    streams = np.empty((lengths[idxs].sum(),3), dtype=np.float32)
    scalars = np.empty((lengths[idxs].sum(), n_scalars), dtype=np.float32) if n_scalars > 0 else None
    j=0
    with open(trk_fn,'rb') as f:
        for idx in idxs:
            f.seek(index_bytes[idx])
            s = np.fromfile(f, np.float32, n_floats[idx])
            s.resize(lengths[idx], point_size)
            if n_scalars > 0:
                scalars[j:j+lengths[idx],:] = s[:,3:]
                s=s[:,:3]
            streams[j:j+lengths[idx],:]=s
            j+= lengths[idx]
        aff = get_affine_trackvis_to_rasmm(lazy_trk.header)
        streams = apply_affine(aff, streams)
    return streams, lengths[idxs]

class TestSampling(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        self.output_size = output_size

    def __call__(self, sample):
        sl = sample['points']
        n = sl.shape[0]
        if self.output_size > len(range(n)):
            chosen_idx = range(n)
        else:
            chosen_idx = np.random.choice(range(n),self.output_size,replace=False).tolist()
        out_sample = {'points': sl[chosen_idx]}
        return out_sample

def MLP(channels, batch_norm=True):
    if batch_norm:
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i])) for i in range(1, len(channels))])

class DECSeq(torch.nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, dropout=True, k=5, aggr='max',pool_op='max'):
        super(DECSeq, self).__init__()
        self.conv1 = EdgeConv(MLP([2*input_size, 64,64,64], batch_norm=True), aggr)
        self.conv2 = DynamicEdgeConv(MLP([2*64,128], batch_norm=True), k, aggr)
        self.lin1 = MLP([128+64, 1024])
        self.pool = global_max_pool
        self.mlp = Seq(MLP([1024, 512]), MLP([512, 256]), Lin(256, n_classes))

    def forward(self, data):
        pos, batch, eidx = data.pos, data.batch, data.edge_index
        x1 = self.conv1(pos,eidx)
        x2 = self.conv2(x1,batch)
        out = self.lin1(torch.cat([x1,x2],dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return out

t1_static= 'MNI_T1_1mm.nii.gz'
#t1_static = $T1_static_padded (MNI_T1_1mm.nii.gz)
#t1_moving= $t1_moving
t1_moving = 't1w_105115.nii.gz'
fixed = ants.image_read(t1_static)
moving = ants.image_read(t1_moving)
mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')
genericAffine = mytx['invtransforms'][0]
invWarp = mytx['invtransforms'][1]
#os.system('warpinit %s $ID_warp[].nii.gz -force' % (t1_static))
os.system('warpinit %s ID_warp[].nii.gz -force' %(t1_static))

#for i in range(3):
#    os.system('WarpImageMultiTransform 3 %s %s -R %s -i %s %s' %
#    ('$ID_warp[].nii.gz'.replace('[]',str(i)), '$mrtrix_warp[].nii.gz'.replace('[]',str(i)),
#    t1_static, genericAffine, invWarp))

for i in range(3):
    os.system('WarpImageMultiTransform 3 %s %s -R %s -i %s %s' %
    ('ID_warp[].nii.gz'.replace('[]',str(i)), 'mrtrix_warp[].nii.gz'.replace('[]',str(i)),
    t1_static, genericAffine, invWarp))

#os.system('warpcorrect $mrtrix_warp[].nii.gz $mrtrix_warp_cor.nii.gz -force')
#trk = nib.streamlines.load($input_trk)
#nib.streamlines.save(trk.tractogram, $output_name.tck)
#os.system('tcktransform $output_name.tck $mrtrix_warp_cor.nii.gz $new_output_name.tck -force -nthreads 0')

os.system('warpcorrect mrtrix_warp[].nii.gz mrtrix_warp_cor.nii.gz -force')
trk = nib.streamlines.load('/home/ruben/Thesis/data/sub-105115_var-HCP_full_tract_SUB.trk')
nib.streamlines.save(trk.tractogram, 'sub-105115_var-HCP_full_tract_SUB.tck')
os.system('tcktransform sub-105115_var-HCP_full_tract_SUB.tck mrtrix_warp_cor.nii.gz sub-105115_var-HCP_full_tract_SUB2mni.tck -force -nthreads 0')

#nii = nib.load($T1_original_MNI)
#header = {}
#header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
#header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
#header[Field.DIMENSIONS] = nii.shape[:3]
#header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))
#tck = nib.streamlines.load($new_output_name.tck)
#nib.streamlines.save(tck.tractogram, $new_output_name.trk, header=header)

nii = nib.load('/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz')
header = {}
header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
header[Field.DIMENSIONS] = nii.shape[:3]
header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))
tck = nib.streamlines.load('sub-105115_var-HCP_full_tract_SUB2mni.tck')
nib.streamlines.save(tck.tractogram, 'sub-105115_var-HCP_full_tract_SUB2mni.trk', header=header)

sft = load_tractogram('sub-105115_var-HCP_full_tract_SUB2mni.trk','same',bbox_valid_check=False)
resampled = []
for sl in sft.streamlines:
    resampled.append(set_number_of_points(sl, 15))
sft_resampled = StatefulTractogram(resampled,'/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz',Space.RASMM)
save_tractogram(sft_resampled,'sub-105115_var-HCP_full_tract_SUB2mni_resampled.trk',bbox_valid_check=False)

trans_val = []
trans_val.append(TestSampling(8000))
dataset = TractData(transform=transforms.Compose(trans_val), return_edges=True, split_obj=True)
dataloader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
classifier = DECSeq(input_size=3,embedding_size=40,n_classes=2,dropout=True,k=5,aggr='max',pool_op='max')
#classifier.cuda()
#classifier.load_state_dict(torch.load($weights.pth))
classifier.load_state_dict(torch.load('/home/ruben/best_model_ep-920_score-0.954500.pth',map_location=torch.device('cpu')))
classifier.eval()
split_obj=True
consumed=False
j=0
visualized=0
new_obj_read=True
while j<len(dataset):
    data=dataset[j]
    if split_obj:
        if new_obj_read:
            obj_pred_choice = torch.zeros(data['obj_full_size'], dtype=torch.int)
            #obj_pred_choice = torch.zeros(data['obj_full_size'], dtype=torch.int).cuda()
            new_obj_read=False
        if len(dataset.remaining[j]) == 0:
            consumed=True
    points = gBatch().from_data_list([data['points']])
    if 'bvec' in points.keys:
        points.batch = points.bvec.clone()
        del points.bvec
    points['lengths'] = points['lengths'][0].item()
    #points = points.to('cuda')
    logits = classifier(points)
    logits = logits.view(-1,2)
    pred = F.log_softmax(logits, dim=-1).view(-1,2)
    pred_choice = pred.data.max(1)[1].int()
    obj_pred_choice[data['obj_idxs']] = pred_choice
    if consumed:
        j+=1
        if split_obj:
            consumed=False
            new_obj_read=True   
#preds = list(obj_pred_choice.cpu().numpy())
preds = list(obj_pred_choice.numpy())
sft = load_tractogram('sub-105115_var-HCP_full_tract_SUB2mni_resampled.trk','same',bbox_valid_check=False)
pl_sls = []
np_sls = []
for i,p in enumerate(preds):
    if p==1:
        pl_sls.append(sft.streamlines[i])
    if p==0:
        np_sls.append(sft.streamlines[i])
sft_pl = StatefulTractogram(pl_sls, '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz', Space.RASMM)
sft_np = StatefulTractogram(np_sls, '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz', Space.RASMM)
save_tractogram(sft_pl, 'sub-105115_var-HCP_full_tract_SUB2mni_resampled_PL.trk', bbox_valid_check=False)
save_tractogram(sft_np, 'sub-105115_var-HCP_full_tract_SUB2mni_resampled_NP.trk', bbox_valid_check=False)
