import torch
from torch import nn

from device import device


class FaceModel(nn.Module):

    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        if num_classes:
            self.register_buffer('centers', (
                torch.rand(num_classes, feature_dim).to(device) - 0.5) * 2)
            self.classifier = nn.Linear(self.feature_dim, num_classes)
