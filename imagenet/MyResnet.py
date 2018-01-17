import torch.nn as nn


class MyResnet(nn.Module):
    """ Finetunes just the last layer
    This freezes the weights of all layers except the last one.
    You should also look into finetuning previous layers, but slowly
    Ideally, do this first, then unfreeze all layers and tune further with small lr

    Arguments:
        original_model: Model to finetune
        arch: Name of model architecture
        num_classes: Number of classes to tune for
    """

    def __init__(self, original_model, arch, num_classes):
        super(MyResnet, self).__init__()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            self.features = original_model.features
            self.fc = nn.Sequential(*list(original_model.classifier.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(4096, num_classes)
            )
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(original_model.fc.in_features, num_classes)
            )
        elif arch.startswith('inception'):
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, num_classes)
            )
        else:
            raise ("Finetuning not supported on this architecture yet. Feel free to add")

        self.unfreeze(False)  # Freeze weights except last layer

    def unfreeze(self, unfreeze):
        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = unfreeze
        if hasattr(self, 'fc'):
            for p in self.fc.parameters():
                p.requires_grad = unfreeze

    def forward(self, x):
        f = self.features(x)
        if hasattr(self, 'fc'):
            f = f.view(f.size(0), -1)
            f = self.fc(f)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y