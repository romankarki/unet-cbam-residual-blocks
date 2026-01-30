import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SegDataset
from models.unet_plain import UNet
device = "cpu"

dataset = SegDataset("data/z_images", "data/z_masks")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()
    total_loss = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "unet_plain.pth")
