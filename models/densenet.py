import torch.nn as nn
from torchvision.models import densenet121


class DeepfakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = densenet121(pretrained=True)
        self.backbone.classifier = nn.Linear(1024, 1)

    def forward(self, x):
        return self.backbone(x)
