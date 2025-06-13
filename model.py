import torch.nn as nn
import torchvision.models as models
import torch

class ResNetBBoxRegressor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBBoxRegressor, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC

        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 5),  # class_prob, x_center, y_center, width, height
            nn.Sigmoid()        # Output between 0 and 1
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)
