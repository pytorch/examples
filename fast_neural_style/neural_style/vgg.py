import torch
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

    def forward(self, X):
        h = self.slice1(X)
        out1 = h
        h = self.slice2(h)
        out2 = h
        h = self.slice3(h)
        out3 = h
        h = self.slice4(h)
        out4 = h
        return [out1, out2, out3, out4]
