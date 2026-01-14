import numpy as np
def compute_activation_coverage(heatmaps, threshold=0.6):
 coverages = []
 for hm in heatmaps:
 active = (hm > threshold).sum()
 coverages.append(active / hm.size)
 return float(np.mean(coverages))
def count_activated_frames(heatmaps, threshold=0.6, min_ratio=0.1):
 count = 0
 for hm in heatmaps:
 ratio = (hm > threshold).sum() / hm.size
 if ratio >= min_ratio:
 count += 1
 return count
