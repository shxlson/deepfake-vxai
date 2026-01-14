import os
import torch
from torch.utils.data import DataLoader
from models.densenet import DeepfakeNet
from training.dataset import FrameDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

train_real = FrameDataset("data/faces/train/real", label=0)
train_fake = FrameDataset("data/faces/train/fake", label=1)

dataset = train_real + train_fake
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = DeepfakeNet().to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0

    model.train()
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss = {total_loss:.4f}")

# --------------------------------------------------
# SAVE TRAINED WEIGHTS (CRITICAL ADDITION)
# --------------------------------------------------
os.makedirs("outputs/checkpoints", exist_ok=True)
torch.save(model.state_dict(), "outputs/checkpoints/densenet.pth")
print("Trained model weights saved to outputs/checkpoints/densenet.pth")
