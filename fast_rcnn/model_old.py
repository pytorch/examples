import torch.nn as nn
from roi_pooling import roi_pooling

class Network(nn.Container):

  def __init__(self, features, classifier):
    super(Network, self).__init__()
    self.features = features
    self.classifier = classifier
  
  def forward(self, x):
    images, rois = x
    x = self.features(images)
    x = roi_pooling(x, rois, size=(3,3), spatial_scale=1.0/16.0)
    x = self.classifier(x)
    return x

def basic_net():
  features = nn.Sequential(nn.Conv2d(3,16,3,16,1,1))
  classifier = nn.Sequential(nn.Linear(3*3*16,10))
  return Network(features, classifier)

if __name__ == '__main__':
  import torch
  import torch.autograd
  m = basic_net()
  x = torch.autograd.Variable(torch.rand(1,3,224,224))
  b = torch.autograd.Variable(torch.LongTensor([[0,1,50,200,200],[0,50,50,200,200]]))
  o = m((x,b))
