from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
"""

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

__all__ = ['CrossEntropyLabelSmooth', 'TripletLoss', 'CenterLoss']

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = Variable(targets, requires_grad=False)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n,feat_dim = inputs.size(0),inputs.size(1)
        
        #inputs_l2norm = inputs.norm(2, dim=1, keepdim=True).expand(n, feat_dim) 
        #inputs = inputs.div(inputs_l2norm+1e-12) 
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = (mask == 0)
        dist_ap, dist_an = [], []
        for i in range(n):
            rand_ap = dist[i][mask[i]][torch.from_numpy(np.random.choice(dist[i][mask[i]].size(0), 1)).cuda()]
            rand_an = dist[i][neg_mask[i]][torch.from_numpy(np.random.choice(dist[i][neg_mask[i]].size(0), 1)).cuda()]
            dist_ap.append(rand_ap)
            hard_negatives = np.where(np.logical_and(dist[i][neg_mask[i]].cpu().data.numpy() - rand_ap.cpu().data.numpy() > 0, dist[i][neg_mask[i]].cpu().data.numpy() - rand_ap.cpu().data.numpy() > 0))[0]
            if len(hard_negatives) > 0:
                dist_an.append(dist[i][neg_mask[i]][torch.from_numpy(np.random.choice(hard_negatives, 1)).cuda()])
            else:
                dist_an.append(rand_an)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        ''' 
        #debug    
        print('feat -> ', inputs[:30,:30])
        print('targets -> ', targets)
        print('mask -> ', mask[:30,:30])
        print('dist -> ', dist[:30,:30])
        print('dist_ap -> ', dist_ap)
        print('dist_an -> ', dist_an)
        #save grad
        inputs.register_hook(save_grad('feat'))
        dist.register_hook(save_grad('dist'))
        dist_ap.register_hook(save_grad('dist_ap'))
        dist_an.register_hook(save_grad('dist_an'))
        loss.register_hook(save_grad('loss'))
        '''
        return loss

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        classes = Variable(classes)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

if __name__ == '__main__':
    pass
