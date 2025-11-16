# explainability.py
"""
Explainability helpers for the Martian terrain segmentation model:

- Grad-CAM-style heatmaps for a particular terrain class.
- Integrated Gradients saliency maps for input pixels.
- "Neural PCA": PCA over feature channels to visualize dominant spatial components.

These are written to work with the UNet defined in models.py, but are generic
enough to adapt to other architectures.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def grad_cam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    target_layer: nn.Module,
) -> torch.Tensor:
    """
    Compute a Grad-CAM heatmap for a segmentation model.

    Args:
        model: segmentation model with logits output [B,C,H,W].
        input_tensor: single input image tensor [1,C,H,W] with gradients enabled.
        target_class: terrain class index (0..num_classes-1).
        target_layer: layer to hook (e.g. model.get_cam_layer()).

    Returns:
        heatmap: [1,1,H,W] normalized to [0,1].
    """
    model.eval()

    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

    try:
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        logits = model(input_tensor)  # [1,C,H,W]

        # Use average over spatial dimensions for the target class
        target_score = logits[:, target_class, :, :].mean()
        model.zero_grad()
        target_score.backward()

        feats = activations["value"]        # [1, F, h, w]
        grads = gradients["value"]          # [1, F, h, w]

        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1,F,1,1]
        cam = (weights * feats).sum(dim=1, keepdim=True)  # [1,1,h,w]
        cam = F.relu(cam)

        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )

        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach()
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor | None = None,
    steps: int = 50,
) -> torch.Tensor:
    """
    Integrated Gradients for segmentation.

    Args:
        model: segmentation model.
        input_tensor: [1,C,H,W] (requires_grad will be set internally).
        target_class: class index (0..num_classes-1).
        baseline: [1,C,H,W] baseline (defaults to zeros).
        steps: number of integration steps.

    Returns:
        attribution map [1,C,H,W] (same shape as input).
    """
    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    input_tensor = input_tensor.detach()
    baseline = baseline.detach()

    # interpolate between baseline and input
    scaled_inputs = [
        baseline + (float(i) / steps) * (input_tensor - baseline)
        for i in range(1, steps + 1)
    ]

    total_gradients = torch.zeros_like(input_tensor)

    for x in scaled_inputs:
        x = x.clone().detach().requires_grad_(True)
        logits = model(x)
        target_score = logits[:, target_class, :, :].mean()

        model.zero_grad()
        target_score.backward()

        assert x.grad is not None
        total_gradients += x.grad.detach()

    avg_gradients = total_gradients / float(steps)

    attributions = (input_tensor - baseline) * avg_gradients
    return attributions.detach()


def neural_pca(
    feature_map: torch.Tensor,
    n_components: int = 3,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    "Neural PCA" over feature channels.

    Given a feature map from some intermediate layer [C,H,W] (or [1,C,H,W]),
    we:
      - Flatten to [C, H*W],
      - Compute covariance across channels,
      - Extract top k eigenvectors,
      - Project back to spatial maps.

    Args:
        feature_map: [C,H,W] or [1,C,H,W].
        n_components: number of principal components.

    Returns:
        pcs: list of [H,W] tensors (principal component maps).
        eigenvalues: [n_components] tensor with corresponding eigenvalues.
    """
    if feature_map.dim() == 4:
        feature_map = feature_map[0]  # [C,H,W]
    assert feature_map.dim() == 3, "feature_map must be [C,H,W] or [1,C,H,W]"

    C, H, W = feature_map.shape
    F_flat = feature_map.view(C, -1)  # [C, H*W]

    # Zero-mean across spatial locations
    F_flat = F_flat - F_flat.mean(dim=1, keepdim=True)

    # Covariance [C,C] across channels
    cov = F_flat @ F_flat.T / (F_flat.shape[1] - 1)

    # Eigen-decomposition (ascending eigenvalues)
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # Take top-k components (descending order)
    k = min(n_components, C)
    top_indices = torch.arange(C - 1, C - k - 1, -1)
    pcs: List[torch.Tensor] = []
    top_eigvals = eigvals[top_indices]

    for idx in top_indices:
        v = eigvecs[:, idx]  # [C]
        # Combine channels according to eigenvector
        pc_flat = (v.view(C, 1) * F_flat).sum(dim=0)  # [H*W]
        pc_map = pc_flat.view(H, W)
        pcs.append(pc_map)

    return pcs, top_eigvals
