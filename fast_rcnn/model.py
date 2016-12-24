import torch.nn as nn
from roi_pooling import roi_pooling as _roi_pooling

from rpn import RPN as _RPN
from faster_rcnn import FasterRCNN as _FasterRCNN

class _Features(nn.Container):
  def __init__(self):
    super(_Features, self).__init__()
    self.m = nn.Conv2d(3, 3, 3, 16, 1)

  def forward(self, x):
    return self.m(x)

class _Classifier(nn.Container):
  def __init__(self):
    super(_Classifier, self).__init__()
    self.m1 = nn.Linear(3*7*7, 21)
    self.m2 = nn.Linear(3*7*7, 21*4)

  def forward(self, x):
    return self.m1(x), self.m2(x)

def _pooler(x, rois):
  x = _roi_pooling(x, rois, size=(7,7), spatial_scale=1.0/16.0)
  return x.view(x.size(0), -1)

class _RPNClassifier(nn.Container):
  def __init__(self, n):
    super(_RPNClassifier, self).__init__()
    self.m1 = nn.Conv2d(n, 18, 3, 1, 1)
    self.m2 = nn.Conv2d(n, 36, 3, 1, 1)

  def forward(self, x):
    return self.m1(x), self.m2(x)

_features = _Features()
_classifier = _Classifier()
_rpn_classifier = _RPNClassifier(3)

_rpn = _RPN(
    classifier=_rpn_classifier
)

model = _FasterRCNN(
    features=_features,
    pooler=_pooler,
    classifier=_classifier,
    rpn=_rpn
)
