import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FrameDataset(Dataset):
    def __init__(self, root_dir, label):
        self.root_dir = root_dir
        self.label = label
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".jpg")
        ]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)
        label = torch.tensor(self.label, dtype=torch.float32)

        return img, label

        '''testing the dataset'''
if __name__ == "__main__":
    dataset = FrameDataset("data/faces/sample", label=0)
    img, label = dataset[0]

    print("Image shape:", img.shape)
    print("Label:", label)
