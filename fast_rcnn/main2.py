import torch
import torch.nn as nn
import torch.autograd as ag
import torch.utils.trainer as trainer
import torch.utils.data
import numpy as np

from roi_pooling import roi_pooling
from voc import VOCDetection, TransformVOCDetectionAnnotation
from faster_rcnn import FasterRCNN

cls = ('__background__', # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
class_to_ind = dict(zip(cls, range(len(cls))))


train = VOCDetection('/home/francisco/work/datasets/VOCdevkit/', 'train',
            target_transform=TransformVOCDetectionAnnotation(class_to_ind, False))


#train_loader = torch.utils.data.DataLoader(
#            ds, batch_size=1, shuffle=True, num_workers=0)

frcnn = FasterRCNN()

frcnn.train()
#for i, (im, gt) in (enumerate(train_loader)):

im, gt = train[0]
if True:
            w, h = im.size
            im = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
            im = im.view(h, w, 3)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            im = im.transpose(0, 1).transpose(0, 2).contiguous()
            im = im.float().div_(255)
            im = im.unsqueeze(0)


loss, scores, boxes = frcnn((im, gt))
from IPython import embed; embed()
