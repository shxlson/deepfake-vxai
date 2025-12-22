import torch
from torch.utils.data import DataLoader
from models.densenet import DeepfakeNet
from training.dataset import FrameDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

real_ds = FrameDataset("data/faces/real", label=0)
fake_ds = FrameDataset("data/faces/fake", label=1)

dataset = real_ds + fake_ds
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = DeepfakeNet().to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    total_loss = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
