import torch
import matplotlib.pyplot as plt
from dataset import SegDataset
from models.resunet_cbam import CBAMResUNet
from models.unet_plain import UNet


dataset = SegDataset("data/images", "data/masks")

unet = UNet()
unet.load_state_dict(torch.load("unet_plain.pth"))
unet.eval()

cbam = CBAMResUNet()
cbam.load_state_dict(torch.load("cbam_resunet.pth"))
cbam.eval()

img, mask = dataset[0]

with torch.no_grad():
    pred_unet = unet(img.unsqueeze(0))[0][0]
    pred_cbam = cbam(img.unsqueeze(0))[0][0]

plt.figure(figsize=(16,4))
plt.subplot(1,4,1); plt.title("Input"); plt.imshow(img.permute(1,2,0))
plt.subplot(1,4,2); plt.title("GT"); plt.imshow(mask[0], cmap="gray")
plt.subplot(1,4,3); plt.title("U-Net"); plt.imshow(pred_unet, cmap="gray")
plt.subplot(1,4,4); plt.title("CBAM-ResUNet"); plt.imshow(pred_cbam, cmap="gray")
plt.show()