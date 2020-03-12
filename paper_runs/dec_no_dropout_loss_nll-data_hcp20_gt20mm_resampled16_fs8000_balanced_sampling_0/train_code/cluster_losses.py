import torch
import torch.nn as nn
import torch.nn.functional as F


class GFLoss(nn.Module):
    def __init__(self):
        super(GFLoss, self).__init__()

    def forward(self, x, f, prob, n, p):
        if p == 'Inf':
            n = 1
        return (1./(2*n**2)) * torch.norm(f - torch.bmm(prob, x), p=float(p))**2
        #return 1 - torch.norm(x)

class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()

    def forward(self, c, f, prob, n=1, p=2):
        return torch.norm(f - torch.bmm(prob, c), p=float(p))
        #return 1 - torch.norm(x)


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1)
        b = b.sum()
        return b

class DistLoss(nn.Module):
    def __init__(self):
        super(DistLoss, self).__init__()

    def forward(self, x):
        # x must be tensors of size (bs, q, k), where bs is bacth_size,
        # q is features dim, and k is number of elements. This function computes
        # the sum of the euclidean distance of all the pairs (i, j), with i,j in
        # [0,k)

        l2norm = torch.norm(x.data)
        summation = 0.
        #F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        for b in range(x.size(0)):
            for i in range(1, x.size(2)-1):
                summation = summation + \
                    (F.pairwise_distance(x[b, :, :-i], x[b, :, i:])).mean()

        return summation/(i*(b+1))

class CSimLoss(nn.Module):
    def __init__(self):
        super(CSimLoss, self).__init__()

    def forward(self, x):
        # x must be tensors of size (bs, q, k), where bs is bacth_size,
        # q is features dim, and k is number of elements. This function computes
        # the sum of the cosine similarity of all the pairs (i, j), with i,j in
        # [0,k)

        summation = 0.
        for i in range(1, x.size(2)-1):
            summation = summation + \
                F.cosine_similarity(x[:, :, :-i], x[:, :, i:], dim=2).mean()

        return summation/i
