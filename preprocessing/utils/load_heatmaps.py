import os
import cv2
import numpy as np
def load_heatmaps(folder):
 heatmaps = []
 files = sorted(os.listdir(folder))
 for file in files:
 path = os.path.join(folder, file)
 img = cv2.imread(path)
 if img is None:
 continue
 img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 img = img.astype(np.float32) / 255.0
 heatmaps.append(img)
 return heatmaps