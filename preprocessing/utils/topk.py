import numpy as np
def select_topk_by_cam(heatmaps, k=8):
 scores = [hm.mean() for hm in heatmaps]
 return np.argsort(scores)[-k:]