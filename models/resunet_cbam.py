import torch
import torch.nn as nn
from models.cbam import CBAM
from models.residual import ResidualBlock

class CBAMResUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = ResidualBlock(3, 32)
        self.att1 = CBAM(32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualBlock(32, 64)
        self.att2 = CBAM(64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualBlock(64, 128)
        self.att3 = CBAM(128)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(128, 256)
        self.att4 = CBAM(256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ResidualBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ResidualBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ResidualBlock(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.att1(self.enc1(x))
        e2 = self.att2(self.enc2(self.pool1(e1)))
        e3 = self.att3(self.enc3(self.pool2(e2)))

        b = self.att4(self.bottleneck(self.pool3(e3)))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))
