import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from models.densenet import DeepfakeNet
from training.dataset import FrameDataset

dataset = FrameDataset("data/faces/sample", label=0)
img, _ = dataset[0]

model = DeepfakeNet()
model.eval()

with torch.no_grad():
    output = model(img.unsqueeze(0))

print("Model output:", output)
