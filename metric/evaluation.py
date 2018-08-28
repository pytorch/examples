# coding : utf-8
from __future__ import absolute_import
import torch
import heapq
import numpy as np
import random

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def pairwise_distance(features, metric=None):
    n = len(features)
    x = torch.cat(features)
    x = x.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True)
    dist = dist.expand(n, n)
    dist = dist + dist.t()
    dist = dist - 2 * torch.mm(x, x.t()) + 1e6 * torch.eye(n)
    dist = torch.sqrt(dist)
    return dist


def pairwise_similarity(features):
    n = len(features)
    x = torch.cat(features)
    x = x.view(n, -1)
    similarity = torch.mm(x, x.t()) - 1e5 * torch.eye(n)
    return similarity


def nmi(features, labels, n_cluster):
    features = [to_numpy(x) for x in features]
    labels = to_numpy(labels)
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(features)
    nmi = normalized_mutual_info_score(labels, kmeans.labels_)
    return nmi


def recall(features, labels):
    dis_mat = pairwise_distance(features)
    dis_mat = to_numpy(dis_mat)
    labels = to_numpy(labels)
    m, n = dis_mat.shape

    # Sort and find correct matches
    top1 = top5 = 0
    indice = np.argsort(dis_mat, axis=1)
    for i in range(m):
        sim_indice = indice[i]
        if labels[i] == labels[sim_indice[0]]:
            top1 += 1
            top5 += 1
        elif labels[i] in labels[sim_indice[1:5]]:
            top5 += 1
            
    return top1/float(m), top5/float(m)
    

def Recall_at_ks_products(sim_mat, query_ids=None, gallery_ids=None):
    """
    :param sim_mat:
    :param query_ids
    :param gallery_ids

    for the Deep Metric problem, following the evaluation table of Proxy NCA loss
    only compute the [R@1, R@10, R@100]

    fast computation via heapq

    """
    sim_mat = to_numpy(sim_mat)
    m, n = sim_mat.shape
    num_max = int(1e4)
    
    # Fill up default values
    gallery_ids = np.asarray(gallery_ids)
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    
    # Ensure numpy array
    if m > num_max:
        samples = list(range(m))
        random.shuffle(samples)
        samples = samples[:num_max]
        sim_mat = sim_mat[samples, :]
        query_ids = [query_ids[k] for k in samples]
        m = num_max
    else:
        query_ids = np.asarray(query_ids)

    # Sort and find correct matches
    # indice = np.argsort(sim_mat, axis=1)
    num_valid = np.zeros(3)
    for i in range(m):
        x = sim_mat[i]
        indice = heapq.nlargest(100, range(len(x)), x.take)
        if query_ids[i] == gallery_ids[indice[0]]:
            num_valid += 1
        elif query_ids[i] in gallery_ids[indice[1:10]]:
            num_valid[1:] += 1
        elif query_ids[i] in gallery_ids[indice[10:]]:
            num_valid[2] += 1
    return num_valid/float(m)


def test_nmi():
    #label = [1, 2, 3]*2
    label = torch.from_numpy(np.array([1, 2, 3, 1, 2, 3]))
    X = torch.from_numpy(np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]))

    print(nmi(X, label, len(set(label))))


def test_recall():
    import torch
    feature = torch.Tensor([[1,1,1,1,1], [1,2,2,3,4], [2,2,2,2,2], [3,2,4,5,6], [1,2,3,4,5]])
    labels = torch.LongTensor([0,0,1,1,2])
    dis_mat, indice, top1, top5 = recall(feature, labels)
    print('dis_mat->\n', dis_mat)
    print('indice->\n', indice)
    print('labels->\n', labels)
    print('top1->', top1)
    print('top5->', top5)

if __name__ == '__main__':
    test_nmi()
    test_recall()
