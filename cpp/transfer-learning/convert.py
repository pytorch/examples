"""
This python script converts the network into Script Module
"""
import torch
from torchvision import models

# Download and load the pre-trained model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Set upgrading the gradients to False
for param in model.parameters():
	param.requires_grad = False

# Save the model except the final FC Layer
resnet18 = torch.nn.Sequential(*list(model.children())[:-1])

example_input = torch.rand(1, 3, 224, 224)
script_module = torch.jit.trace(resnet18, example_input)
script_module.save('resnet18_without_last_layer.pt')
