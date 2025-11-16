# models.py
"""
Lightweight U-Net-style semantic segmentation model for Martian terrain (AI4Mars).

AI4Mars overview:
- Rovers: Curiosity (MSL), Opportunity, Spirit.
- Labels: soil, bedrock, sand, big rock, plus a null/no-label class mapped to 255.
  (0..3 used as terrain classes, 255 as null). :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) x2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling block: MaxPool -> DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upscaling block: (bilinear upsample or ConvTranspose2d) + DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # in_channels is cat(x_skip, x_up), so DoubleConv(in_channels, out_channels)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # If using transposed conv, we reduce channels before concatenation.
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x_up)

        # Pad if needed to handle odd spatial dims
        diff_y = x_skip.size(2) - x_up.size(2)
        diff_x = x_skip.size(3) - x_up.size(3)
        x_up = F.pad(
            x_up,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final conv to get logits."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for semantic segmentation.

    - in_channels: Navcam images are grayscale (1 channel) in AI4Mars. :contentReference[oaicite:2]{index=2}
    - num_classes: 4 (soil, bedrock, sand, big rock).
    - base_channels: controls model width; keep small for "lightweight" behaviour.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)

        self.outc = OutConv(base_channels, num_classes)

        # We'll use this layer as a natural target for CAM/Grad-CAM.
        # (Last decoder feature map, before classifier.)
        self.cam_target_layer = self.up4.conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

    def get_cam_layer(self) -> nn.Module:
        """
        Helper for explainability modules: returns the layer we want to hook
        for Grad-CAM. For this U-Net, we use the last decoder conv.
        """
        return self.cam_target_layer


def create_unet(
    in_channels: int = 1,
    num_classes: int = 4,
    base_channels: int = 32,
    bilinear: bool = True,
) -> UNet:
    """Factory function used in the notebook."""
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        bilinear=bilinear,
    )
