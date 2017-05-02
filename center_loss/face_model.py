import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock, ResNet, model_urls
import math

class FaceModel(nn.Module):
    def __init__(self,num_classes, pretrained=True, **kwargs):
        super(FaceModel, self).__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            parameters =  model_zoo.load_url(model_urls['resnet18'])
            self.model.load_state_dict(parameters)
        self.model.avgpool = None
        self.model.fc1 = nn.Linear(512*3*4, 512)
        self.model.fc2 = nn.Linear(512, 512)
        self.model.classifier = nn.Linear(512, num_classes)
        self.register_buffer('centers', torch.zeros(num_classes, 512))
        self.num_classes = num_classes

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc1(x)
        #feature for center loss
        x = self.model.fc2(x)
        self.features = x
        x = self.model.classifier(x)
        return F.log_softmax(x)
