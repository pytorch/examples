import torch
import torch.nn as nn
import torch.autograd as ag
import torch.utils.trainer as trainer
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
from rpn import RPN
import torch.optim as optim

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

class Features(nn.Container):
  def __init__(self):
    super(Features, self).__init__()
    self.m = nn.Conv2d(3, 3, 3, 16, 1)

  def forward(self, x):
    return self.m(x)

class Classifier(nn.Container):
  def __init__(self):
    super(Classifier, self).__init__()
    self.m1 = nn.Linear(3*7*7, 21)
    self.m2 = nn.Linear(3*7*7, 21*4)

  def forward(self, x):
    return self.m1(x), self.m2(x)

def pooler(x, rois):
  from roi_pooling import roi_pooling
  x = roi_pooling(x, rois, size=(7,7), spatial_scale=1.0/16.0)
  return x.view(x.size(0), -1)

class RPNClassifier(nn.Container):
  def __init__(self, n):
    super(RPNClassifier, self).__init__()
    self.m1 = nn.Conv2d(n, 18, 3, 1, 1)
    self.m2 = nn.Conv2d(n, 36, 3, 1, 1)

  def forward(self, x):
    return self.m1(x), self.m2(x)

rpn = RPN(RPNClassifier(3))

frcnn = FasterRCNN(Features(), pooler, Classifier(), rpn)

frcnn.train()

optimizer = optim.SGD(frcnn.parameters(), lr = 0.01, momentum=0.9)


from IPython import embed; embed()

#for i, (im, gt) in tqdm(enumerate(train_loader)):
#  optimizer.zero_grad()
#  loss, scores, boxes = frcnn((im, gt))
#  loss.backward()
#  optimizer.step()

#im, gt = train[0]
#im = im.unsqueeze(0)

#loss, scores, boxes = frcnn((im, gt))
#from IPython import embed; embed()
