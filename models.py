# models.py
"""
Student & Teacher U-Net-style semantic segmentation models for Martian terrain (AI4Mars).

AI4Mars overview:
- Rovers: Curiosity (MSL), Opportunity, Spirit.
- Labels: soil, bedrock, sand, big rock, plus a null/no-label class mapped to 255.
  (0..3 used as terrain classes, 255 as null).
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# Student model: lightweight UNet
# ---------------------------------------------------------------------


class UNet(nn.Module):
    """
    Lightweight U-Net for semantic segmentation (student model).

    - in_channels: Navcam images are grayscale (1 channel) in AI4Mars.
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
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
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
    """
    Factory for the lightweight STUDENT model.
    """
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        bilinear=bilinear,
    )


# ---------------------------------------------------------------------
# Teacher model: deeper Attention U-Net
# ---------------------------------------------------------------------


class AttentionGate(nn.Module):
    """
    Attention gate for U-Net skip connections (roughly from Attention U-Net).

    g: gating signal from decoder (coarser feature)
    x: skip connection from encoder (finer feature)
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g: [B, F_g, H_g, W_g] (decoder)
        x: [B, F_l, H_x, W_x] (encoder skip)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # upsample g1 to match x1 spatially if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # [B,1,H,W] attention map
        return x * psi       # gated skip


class UpAttn(nn.Module):
    """
    Upscaling block with attention on skip connection.

    x_up: decoder feature (coarse)
    x_skip: encoder feature (fine) -> gated with attention
    """

    def __init__(
        self,
        skip_channels: int,
        up_channels: int,
        out_channels: int,
        bilinear: bool = True,
    ):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            # transposed conv keeps the number of channels = up_channels
            self.up = nn.ConvTranspose2d(
                up_channels, up_channels, kernel_size=2, stride=2
            )

        # Attention gate takes the true channel dims of g (decoder) and x (encoder)
        self.attn = AttentionGate(
            F_g=up_channels,
            F_l=skip_channels,
            F_int=max(out_channels // 2, 1),
        )

        # After gating, we concatenate skip and up -> (skip_channels + up_channels)
        self.conv = DoubleConv(skip_channels + up_channels, out_channels)

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x_up)

        # Pad if needed
        diff_y = x_skip.size(2) - x_up.size(2)
        diff_x = x_skip.size(3) - x_up.size(3)
        x_up = F.pad(
            x_up,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        # Attention-gated skip
        x_skip_attn = self.attn(x_up, x_skip)

        x = torch.cat([x_skip_attn, x_up], dim=1)
        return self.conv(x)



class AttentionUNet(nn.Module):
    """
    Bigger teacher model:
      - Deeper (5 down/5 up instead of 4)
      - Attention gates on skip connections (UpAttn)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # Encoder channel sizes
        c1 = base_channels            # after inc
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16
        c6 = base_channels * 32 // factor  # deepest

        # Encoder
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)
        self.down5 = Down(c5, c6)

        # Decoder with attention gates
        # Note: we pass (skip_channels, up_channels, out_channels)
        self.up1 = UpAttn(
            skip_channels=c5,
            up_channels=c6,
            out_channels=c5 // factor,
            bilinear=bilinear,
        )
        self.up2 = UpAttn(
            skip_channels=c4,
            up_channels=c5 // factor,
            out_channels=c4 // factor,
            bilinear=bilinear,
        )
        self.up3 = UpAttn(
            skip_channels=c3,
            up_channels=c4 // factor,
            out_channels=c3 // factor,
            bilinear=bilinear,
        )
        self.up4 = UpAttn(
            skip_channels=c2,
            up_channels=c3 // factor,
            out_channels=c2 // factor,
            bilinear=bilinear,
        )
        self.up5 = UpAttn(
            skip_channels=c1,
            up_channels=c2 // factor,
            out_channels=c1,
            bilinear=bilinear,
        )

        self.outc = OutConv(c1, num_classes)

        # CAM target layer: last decoder conv
        self.cam_target_layer = self.up5.conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        # Decoder
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)

        logits = self.outc(x)
        return logits

    def get_cam_layer(self) -> nn.Module:
        return self.cam_target_layer



def create_teacher_unet(
    in_channels: int = 1,
    num_classes: int = 4,
    base_channels: int = 64,
    bilinear: bool = True,
) -> AttentionUNet:
    """
    Factory for the TEACHER model (deeper, with attention).
    """
    return AttentionUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        bilinear=bilinear,
    )
