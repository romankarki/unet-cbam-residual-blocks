import torch
import matplotlib.pyplot as plt
from dataset import SegDataset
from models.resunet_cbam import CBAMResUNet
from models.unet_plain import UNet


NUM_IMAGES = 5
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

dataset = SegDataset(IMAGE_DIR, MASK_DIR)

unet = UNet()
unet.load_state_dict(torch.load("unet_plain.pth", map_location="cpu"))
unet.eval()

cbam = CBAMResUNet()
cbam.load_state_dict(torch.load("cbam_resunet.pth", map_location="cpu"))
cbam.eval()

fig, axes = plt.subplots(NUM_IMAGES, 4, figsize=(18, 4 * NUM_IMAGES))

with torch.no_grad():
    for i in range(NUM_IMAGES):
        img, mask = dataset[i]

        img_in = img.unsqueeze(0)

        pred_unet = unet(img_in)[0][0]
        pred_cbam = cbam(img_in)[0][0]

        axes[i, 0].imshow(img.permute(1, 2, 0))
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask[0], cmap="gray")
        axes[i, 1].set_title("GT")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_unet, cmap="gray")
        axes[i, 2].set_title("U-Net")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(pred_cbam, cmap="gray")
        axes[i, 3].set_title("CBAM-ResUNet")
        axes[i, 3].axis("off")

plt.tight_layout()
plt.show()
