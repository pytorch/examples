import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock, ResNet, model_urls
import math

class FaceModel(nn.Module):
    def __init__(self,num_classes, pretrained=False, **kwargs):
        super(FaceModel, self).__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            parameters =  model_zoo.load_url(model_urls['resnet50'])
            self.model.load_state_dict(parameters)
        self.model.avgpool = None
        self.model.fc1 = nn.Linear(512*3*4, 512)
        self.model.fc2 = nn.Linear(512, 512)
        self.model.classifier = nn.Linear(512, num_classes)
        self.centers = torch.zeros(num_classes, 512).type(torch.FloatTensor)
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


class Net(nn.Module):
    def __init__(self,num_classes, pretrained=False, **kwargs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 2)
        self.fc3 = nn.Linear(2, num_classes)
        self.centers = torch.zeros(num_classes, 2).type(torch.FloatTensor)
        self.num_classes = num_classes

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        self.features = x
        x = self.fc3(x)
        return F.log_softmax(x)
