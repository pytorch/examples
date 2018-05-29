import torch
from torch import nn
from torchvision.models import resnet18

from device import device


class FaceModel(nn.Module):

    def __init__(self, num_classes=None):
        super().__init__()
        self.base = resnet18()
        self.extract_feature = nn.Linear(512*4*3, 512)
        self.num_classes = num_classes
        if self.num_classes:
            self.classifier = nn.Linear(512, num_classes)
            self.register_buffer('centers', (
                torch.rand(num_classes, 512).to(device) - 0.5) * 2)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        x = x.view(x.size(0), -1)
        feature = self.extract_feature(x)
        logits = self.classifier(feature) if self.num_classes else None

        feature_normed = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature))

        return logits, feature_normed
