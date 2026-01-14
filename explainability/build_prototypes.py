import torch
from models.densenet import DeepfakeNet
from training.dataset import FrameDataset
from explainability.prototypes import PrototypeBank

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DeepfakeNet().to(device)
model.eval()

bank = PrototypeBank()

real_ds = FrameDataset("data/faces/train/real", label=0)
fake_ds = FrameDataset("data/faces/train/fake", label=1)


def extract_embedding(img):
    with torch.no_grad():
        feats = model.backbone.features(img.unsqueeze(0).to(device))
        emb = feats.mean(dim=(2, 3)).squeeze()
    return emb

# Build prototypes
for i in range(min(5, len(real_ds))):
    img, _ = real_ds[i]
    bank.add(extract_embedding(img), label=0)

for i in range(min(5, len(fake_ds))):
    img, _ = fake_ds[i]
    bank.add(extract_embedding(img), label=1)


bank.build()
print("✅ Prototype bank built")