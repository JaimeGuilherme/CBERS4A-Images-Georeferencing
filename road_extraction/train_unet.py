import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = (cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        return torch.tensor(image), torch.tensor(mask)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def CBR(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU())
        self.enc1 = CBR(1, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.sigmoid(self.final(d1))

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = RoadDataset("dataset_split/train/images", "dataset_split/train/masks")
    val_ds = RoadDataset("dataset_split/val/images", "dataset_split/val/masks")
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{epoch+1}/10] Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "unet_rodovias.pth")
    print("✅ Modelo salvo.")

    model.eval()
    imgs, masks = next(iter(val_loader))
    with torch.no_grad():
        preds = model(imgs.to(device)).cpu().numpy()
    plt.figure(figsize=(10, 3))
    for i in range(3):
        plt.subplot(3, 3, i*3+1); plt.imshow(imgs[i][0], cmap='gray'); plt.title("Imagem")
        plt.subplot(3, 3, i*3+2); plt.imshow(masks[i][0], cmap='gray'); plt.title("Máscara")
        plt.subplot(3, 3, i*3+3); plt.imshow(preds[i][0] > 0.5, cmap='gray'); plt.title("Predição")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    train_model()
