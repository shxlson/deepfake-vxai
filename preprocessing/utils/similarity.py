from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def compute_prototype_similarity(features, fake_prototypes, real_prototypes):
 fake_scores = []
 real_scores = []
 # Ensure prototypes are 2D
 fake_prototypes = fake_prototypes.reshape(1, -1)
 real_prototypes = real_prototypes.reshape(1, -1)
 for feat in features:
 # Ensure feature is 2D
 feat = feat.reshape(1, -1)
 fake_sim = cosine_similarity(feat, fake_prototypes)[0][0]
 real_sim = cosine_similarity(feat, real_prototypes)[0][0]
 fake_scores.append(fake_sim)
 real_scores.append(real_sim)
 return float(np.mean(fake_scores)), float(np.mean(real_scores))
