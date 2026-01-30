import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        # Channel Attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )

        # Spatial Attention
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel attention
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool = torch.max(x, dim=2)[0]
        max_pool = torch.max(max_pool, dim=2)[0]

        channel_att = torch.sigmoid(
            self.mlp(avg_pool) + self.mlp(max_pool)
        ).view(b, c, 1, 1)

        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]

        spatial_att = torch.sigmoid(
            self.spatial(torch.cat([avg_out, max_out], dim=1))
        )

        return x * spatial_att
