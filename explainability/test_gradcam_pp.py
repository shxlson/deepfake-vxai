import torch
import cv2
import numpy as np

from models.densenet import DeepfakeNet
from training.dataset import FrameDataset
from explainability.gradcam_pp import GradCAMPlusPlus

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = DeepfakeNet().to(device)
model.eval()

# Load one frame
dataset = FrameDataset("data/faces/sample", label=0)
img, _ = dataset[0]
input_tensor = img.unsqueeze(0).to(device)

# Grad-CAM++
campp = GradCAMPlusPlus(
    model=model,
    target_layer=model.backbone.features
)

heatmap = campp.generate(input_tensor)

# ---- Visualization ----
heatmap = cv2.resize(heatmap, (224, 224))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

orig = img.permute(1, 2, 0).cpu().numpy()
orig = np.uint8(255 * orig)

overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

cv2.imwrite("gradcampp_result.jpg", overlay)
print("✅ Grad-CAM++ heatmap saved as gradcampp_result.jpg")
