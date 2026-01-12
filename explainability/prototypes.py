import torch
import torch.nn.functional as F
import numpy as np


class PrototypeBank:
    def __init__(self):
        self.real_prototypes = []
        self.fake_prototypes = []

    def add(self, embedding, label):
        """
        embedding: torch tensor [C]
        label: 0 = real, 1 = fake
        """
        embedding = embedding.detach().cpu()
        if label == 0:
            self.real_prototypes.append(embedding)
        else:
            self.fake_prototypes.append(embedding)

    def build(self):
        self.real_prototypes = torch.stack(self.real_prototypes)
        self.fake_prototypes = torch.stack(self.fake_prototypes)

    def similarity(self, embedding):
        """
        Returns max cosine similarity to real and fake prototypes
        """
        embedding = embedding.unsqueeze(0)

        sim_real = F.cosine_similarity(
            embedding, self.real_prototypes, dim=1
        ).max().item()

        sim_fake = F.cosine_similarity(
            embedding, self.fake_prototypes, dim=1
        ).max().item()

        return sim_real, sim_fake


def extract_region_embedding(features, cam):
    """
    features: [1, C, H, W]
    cam: numpy array [H, W] in [0,1]
    """
    cam = torch.tensor(cam).to(features.device)
    cam = cam.unsqueeze(0).unsqueeze(0)

    weighted = features * cam
    embedding = weighted.mean(dim=(2, 3)).squeeze()

    return embedding
