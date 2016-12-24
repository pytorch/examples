import torch
import torch.nn as nn
import torch.autograd as ag
import torch.utils.trainer as trainer
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms

from roi_pooling import roi_pooling
from voc import VOCDetection, TransformVOCDetectionAnnotation
from faster_rcnn import FasterRCNN
from tqdm import tqdm

cls = ('__background__', # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
class_to_ind = dict(zip(cls, range(len(cls))))


train = VOCDetection('/home/francisco/work/datasets/VOCdevkit/', 'train',
            transform=transforms.ToTensor(),
            target_transform=TransformVOCDetectionAnnotation(class_to_ind, False))

def collate_fn(batch):
    imgs, gt = zip(*batch)
    return imgs[0].unsqueeze(0), gt[0]

train_loader = torch.utils.data.DataLoader(
            train, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

frcnn = FasterRCNN()

frcnn.train()
for i, (im, gt) in tqdm(enumerate(train_loader)):
  loss, scores, boxes = frcnn((im, gt))
  loss.backward()

#im, gt = train[0]
#im = im.unsqueeze(0)

#loss, scores, boxes = frcnn((im, gt))
#from IPython import embed; embed()
