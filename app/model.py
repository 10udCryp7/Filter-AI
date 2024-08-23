import torch
import torch.nn as nn
from torchvision.models import resnet50

class LandmarkDetector(nn.Module):
    def __init__(self):
        super(LandmarkDetector, self).__init__()

        # Load ResNet50 with pretrained weights
        self.resnet50 = resnet50(weights='DEFAULT')

        # Remove the original fully connected layer (fc)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])

        # Global Average Pooling and Fully Connected Layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc = nn.Linear(2048, 136)  # 2048 from ResNet50's last layer

    def forward(self, x):
        x = self.resnet50(x)  # Extract features using ResNet50
        x = self.global_avg_pool(x)  # Reduce spatial dimensions
        x = torch.flatten(x, 1)  # Flatten for fully connected layer
        x = self.fc(x)         # Predict 136 values (68 landmarks * 2 coordinates)
        x = x.view(-1, 68, 2)   # Reshape into landmark coordinates
        return x

