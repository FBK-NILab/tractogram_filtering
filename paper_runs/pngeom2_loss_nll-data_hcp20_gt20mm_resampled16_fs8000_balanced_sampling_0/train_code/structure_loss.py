import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def batched_cdist_l2(x1, x2):
    # taken from https://github.com/simonepri/PyTorch-BigGraph/blob/b3b1a845e0cc91c750822284a0cadd086dab9413/torchbiggraph/model.py#L601-L610
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res

class OriDistLoss(nn.Module):
    '''
    The loss regulates the structure, A_z, of the latent space representation, 
    z, by reducing the dissimilarity/distance wrt the structure in the original 
    space 
    '''
    def __init__(self):
        super(OriDistLoss, self).__init__()

    def forward(self, x, z, idxs):
        #ut_indexes = np.triu_indices(len(idxs))
        x_dist = batched_cdist_l2(x[:,idxs,:], x[:,idxs,:])
        x_dist = x_dist.triu()
        #x_dist = x_dist[:,ut_indexes]
        z_dist = batched_cdist_l2(z[:,idxs,:], z[:,idxs,:])
        #z_dist = z_dist[:,ut_indexes]
        z_dist = z_dist.triu()

        #return F.l1_loss(x_dist, z_dist)
        return (((x_dist - z_dist)**2).sum()).sqrt()
