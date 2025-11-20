# models.py
"""
Student & Teacher U-Net–style semantic segmentation models for Martian terrain (AI4Mars).

Model families provided:
- **UNet**: lightweight student model used for distillation.
- **AttentionUNet**: deeper teacher architecture with attention-gated skip connections.

AI4Mars dataset:
- Rover images from Curiosity, Opportunity, and Spirit.
- Terrain classes: ``0 = soil``, ``1 = bedrock``, ``2 = sand``, ``3 = big rock``.
- ``255`` = no-label / ignored.
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
    r"""
    Two consecutive convolution–BatchNorm–ReLU blocks:

    .. math::

        x \mapsto \mathrm{ReLU}(\mathrm{BN}(\mathrm{Conv}(x)))

    applied twice.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of feature channels in the output.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the 2×(Conv–BN–ReLU) block."""
        return self.block(x)


class Down(nn.Module):
    r"""
    Downscaling (encoder) block consisting of:

    - ``MaxPool2d(2)``
    - ``DoubleConv(in_channels, out_channels)``

    Parameters
    ----------
    in_channels : int
        Channels of input feature map.
    out_channels : int
        Channels after convolution.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    r"""
    Upsampling (decoder) block for standard U-Net.

    Includes either:
    - bilinear upsampling, or
    - transposed convolution.

    Followed by ``DoubleConv``.

    Parameters
    ----------
    in_channels : int
        Total number of channels after concatenation of skip + upsampled features.
    out_channels : int
        Output feature channels.
    bilinear : bool
        Whether to use bilinear interpolation (preferred for lightweight model).
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # If transposed conv, reduce channels before concat
            self.up = nn.ConvTranspose2d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_up : torch.Tensor
            Decoder feature map (coarse).
        x_skip : torch.Tensor
            Encoder skip feature map (fine).

        Returns
        -------
        torch.Tensor
            Feature map after concatenation + convolution.
        """
        x_up = self.up(x_up)

        # Pad if shapes mismatch
        diff_y = x_skip.size(2) - x_up.size(2)
        diff_x = x_skip.size(3) - x_up.size(3)
        x_up = F.pad(
            x_up,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1×1 convolution producing class logits.

    Parameters
    ----------
    in_channels : int
        Channels of input feature map.
    num_classes : int
        Number of segmentation classes.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits of shape ``[B,num_classes,H,W]``."""
        return self.conv(x)


# ---------------------------------------------------------------------
# Student model: lightweight UNet
# ---------------------------------------------------------------------
class UNet(nn.Module):
    r"""
    Lightweight U-Net used as the **student** in distillation.

    Architecture: 4 down → 4 up with skip connections.

    Parameters
    ----------
    in_channels : int
        Input channels (``1`` for grayscale Navcam).
    num_classes : int
        Number of segmentation classes (AI4Mars uses ``4``).
    base_channels : int
        Width of first convolution layer. Determines model size.
    bilinear : bool
        Whether to use bilinear upsampling (recommended).
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

        # Encoder
        self.inc   = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8,  base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4,  base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2,  base_channels,             bilinear)

        self.outc = OutConv(base_channels, num_classes)

        # Layer used for Grad-CAM
        self.cam_target_layer = self.up4.conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the U-Net."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        return self.outc(x)

    def get_cam_layer(self) -> nn.Module:
        """
        Return the decoder layer used as the CAM target.

        This is consumed by explainability utilities such as Grad-CAM.
        """
        return self.cam_target_layer


def create_unet(
    in_channels: int = 1,
    num_classes: int = 4,
    base_channels: int = 32,
    bilinear: bool = True,
) -> UNet:
    """Factory for the lightweight student U-Net."""
    return UNet(in_channels, num_classes, base_channels, bilinear)


# ---------------------------------------------------------------------
# Teacher model: deeper Attention U-Net
# ---------------------------------------------------------------------
class AttentionGate(nn.Module):
    r"""
    **Attention gate (AG)** for U-Net skip connections, from *Attention U-Net*.

    Given encoder skip features :math:`x` and decoder gating features :math:`g`,
    the gate computes:

    .. math::

        \psi = \sigma(\mathrm{Conv}( \mathrm{ReLU}(W_g g + W_x x)))

        \tilde{x} = \psi \odot x

    so that skip features can be suppressed when irrelevant.

    Parameters
    ----------
    F_g : int
        Channels of the decoder gating signal.
    F_l : int
        Channels of the encoder skip feature map.
    F_int : int
        Reduced intermediate dimensionality for gating.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        g : torch.Tensor
            Decoder gating signal ``[B,F_g,H_g,W_g]``.
        x : torch.Tensor
            Encoder skip features ``[B,F_l,H_x,W_x]``.

        Returns
        -------
        torch.Tensor
            Attention-reweighted skip features ``[B,F_l,H_x,W_x]``.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # attention map [B,1,H,W]
        return x * psi


class UpAttn(nn.Module):
    r"""
    Upsampling block with **attention-gated skip connections**.

    Parameters
    ----------
    skip_channels : int
        Channels of encoder skip features.
    up_channels : int
        Channels of decoder feature map.
    out_channels : int
        Channels after concatenation + DoubleConv.
    bilinear : bool
        Whether to use bilinear interpolation instead of transposed conv.
    """

    def __init__(self,
                 skip_channels: int,
                 up_channels: int,
                 out_channels: int,
                 bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                up_channels, up_channels, kernel_size=2, stride=2
            )

        self.attn = AttentionGate(
            F_g=up_channels,
            F_l=skip_channels,
            F_int=max(out_channels // 2, 1),
        )

        self.conv = DoubleConv(skip_channels + up_channels, out_channels)

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """Apply attention gate → concat with upsampled features → DoubleConv."""
        x_up = self.up(x_up)

        diff_y = x_skip.size(2) - x_up.size(2)
        diff_x = x_skip.size(3) - x_up.size(3)
        x_up = F.pad(
            x_up,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2],
        )

        x_skip_attn = self.attn(x_up, x_skip)
        x = torch.cat([x_skip_attn, x_up], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    r"""
    **Teacher model**: deeper Attention U-Net.

    Differences from student U-Net:
    - 5 encoder + 5 decoder levels (deeper)
    - Attention gates on all skip connections
    - Wider feature maps (larger ``base_channels``)

    Parameters
    ----------
    in_channels : int
        Image channels.
    num_classes : int
        Number of segmentation classes.
    base_channels : int
        Width multiplier for encoder.
    bilinear : bool
        Bilinear vs transposed conv upsampling.
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

        # Encoder channels
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16
        c6 = base_channels * 32 // factor

        # Encoder
        self.inc   = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)
        self.down5 = Down(c5, c6)

        # Decoder with attention-gated skips
        self.up1 = UpAttn(c5,         c6,         c5 // factor, bilinear)
        self.up2 = UpAttn(c4, c5 // factor, c4 // factor, bilinear)
        self.up3 = UpAttn(c3, c4 // factor, c3 // factor, bilinear)
        self.up4 = UpAttn(c2, c3 // factor, c2 // factor, bilinear)
        self.up5 = UpAttn(c1, c2 // factor, c1,          bilinear)

        self.outc = OutConv(c1, num_classes)

        # Layer used for Grad-CAM
        self.cam_target_layer = self.up5.conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention U-Net."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up1(x6, x5)
        x = self.up2(x,  x4)
        x = self.up3(x,  x3)
        x = self.up4(x,  x2)
        x = self.up5(x,  x1)

        return self.outc(x)

    def get_cam_layer(self) -> nn.Module:
        """Return last decoder block for Grad-CAM."""
        return self.cam_target_layer


def create_teacher_unet(
    in_channels: int = 1,
    num_classes: int = 4,
    base_channels: int = 64,
    bilinear: bool = True,
) -> AttentionUNet:
    """Factory for the deeper teacher model."""
    return AttentionUNet(in_channels, num_classes, base_channels, bilinear)
