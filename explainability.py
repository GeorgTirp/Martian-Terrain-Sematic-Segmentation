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
from matplotlib import pyplot as plt

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

import random

def normalize_map(t: torch.Tensor):
    t = t.clone().detach()
    t_min, t_max = t.min(), t.max()
    if (t_max - t_min) > 1e-6:
        t = (t - t_min) / (t_max - t_min)
    else:
        t = torch.zeros_like(t)
    return t


def explain_per_class_examples(
    model,
    dataset,
    device,
    num_examples_per_class: int = 2,
    ig_steps: int = 32,
):
    """
    For each terrain class (soil, bedrock, sand, big_rock),
    show up to `num_examples_per_class` random examples with:

      - Grad-CAM
      - Integrated Gradients
      - Neural PCA

    Args:
        model: trained segmentation model.
        dataset: typically `test_loader.dataset`.
        device: torch.device.
        num_examples_per_class: how many examples per class to visualize.
        ig_steps: number of steps for Integrated Gradients.
    """
    model.eval()
    cam_layer = model.get_cam_layer()
    num_classes = len(AI4MARS_CLASS_NAMES)

    # --- 1. Scan dataset once to find indices per class ---
    class_to_indices = {c: [] for c in range(num_classes)}
    print("Scanning dataset once to find examples per class...")
    for idx in range(len(dataset)):
        _, mask = dataset[idx]  # mask is a torch.Tensor [H,W] on CPU
        mask_np = mask.numpy()

        for c in range(num_classes):
            if (mask_np == c).any():
                class_to_indices[c].append(idx)

        # Small early-exit optimization (optional)
        if all(len(class_to_indices[c]) >= num_examples_per_class for c in range(num_classes)):
            break

    # --- 2. For each class, sample some indices and visualize ---
    for c in range(num_classes):
        class_name = AI4MARS_CLASS_NAMES[c]
        indices = class_to_indices[c]

        if len(indices) == 0:
            print(f"\nClass '{class_name}' (id={c}): no examples found in dataset.")
            continue

        chosen = random.sample(indices, k=min(num_examples_per_class, len(indices)))
        print(
            f"\n=== Class '{class_name}' (id={c}) | "
            f"{len(indices)} total examples, showing {len(chosen)} ==="
        )

        for ex_id, idx in enumerate(chosen):
            print(f"- Example {ex_id+1}/{len(chosen)} (dataset idx: {idx})")

            input_img, target_mask = dataset[idx]
            input_img = input_img.unsqueeze(0).to(device)  # [1,C,H,W]
            target_mask = target_mask.to(device)           # [H,W]

            # ----- Grad-CAM -----
            cam_map = grad_cam(
                model=model,
                input_tensor=input_img,
                target_class=c,
                target_layer=cam_layer,
            )  # [1,1,H,W]

            # ----- Integrated Gradients -----
            ig_attr = integrated_gradients(
                model=model,
                input_tensor=input_img,
                target_class=c,
                baseline=torch.zeros_like(input_img),
                steps=ig_steps,
            )  # [1,C,H,W]

            # ----- Neural PCA on the same CAM layer features -----
            activations = {}

            def hook_activations(module, inp, out):
                activations["feat"] = out.detach().cpu()

            handle = cam_layer.register_forward_hook(hook_activations)
            with torch.no_grad():
                _ = model(input_img)
            handle.remove()

            feat_map = activations["feat"]  # [1,F,h,w]
            pcs, eigvals = neural_pca(feat_map, n_components=3)  # list of [H,W]

            # ----- Visualization -----
            img_np = input_img[0, 0].detach().cpu().numpy()

            cam_np = cam_map[0, 0].detach().cpu().numpy()
            ig_np = normalize_map(ig_attr[0, 0]).cpu().numpy()
            pc_maps = [normalize_map(pc).cpu().numpy() for pc in pcs]

            fig, axes = plt.subplots(2, 3, figsize=(12, 6))

            # Original image
            axes[0, 0].imshow(img_np, cmap="gray")
            axes[0, 0].set_title("Input")
            axes[0, 0].axis("off")

            # Grad-CAM overlay
            axes[0, 1].imshow(img_np, cmap="gray")
            axes[0, 1].imshow(cam_np, cmap="jet", alpha=0.5)
            axes[0, 1].set_title(f"Grad-CAM ({class_name})")
            axes[0, 1].axis("off")

            # Integrated Gradients overlay
            axes[0, 2].imshow(img_np, cmap="gray")
            axes[0, 2].imshow(ig_np, cmap="inferno", alpha=0.5)
            axes[0, 2].set_title(f"Integrated Gradients ({class_name})")
            axes[0, 2].axis("off")

            # Neural PCA components
            for i in range(3):
                ax = axes[1, i]
                pcm = pc_maps[i]
                ax.imshow(pcm, cmap="coolwarm")
                ax.set_title(f"Neural PCA PC{i+1}\n(eig={eigvals[i].item():.2f})")
                ax.axis("off")

            plt.tight_layout()
            plt.show()
