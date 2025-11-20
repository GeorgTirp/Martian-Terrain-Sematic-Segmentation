# explainability.py
"""
Explainability utilities for Martian terrain segmentation models.

This module provides unified interfaces for:
- **Grad-CAM** heatmaps over decoder features
- **Integrated Gradients** saliency maps
- **Neural PCA**: class-wise PCA in feature space, following the lecture slides

The tools are written to work with the U-Net architecture defined in
``models.py`` but are generic enough to be used with any segmentation
model that exposes a CAM target layer and a final 1×1 classifier.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Sequence
import random
import math
import torch
import torch.nn.functional as F
from torch import nn
from matplotlib import pyplot as plt
from PIL import Image


# ---------------------------------------------------------
#  Grad-CAM
# ---------------------------------------------------------
def grad_cam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    target_layer: nn.Module,
) -> torch.Tensor:
    r"""
    Compute a **Grad-CAM** heatmap for semantic segmentation.

    Grad-CAM produces a spatial importance map based on gradients flowing
    into a target convolutional layer:

    .. math::

        \mathrm{CAM}(x) =
            \mathrm{ReLU}
            \Big(
                \sum_f \alpha_f \cdot A_f(x)
            \Big),

    where:

    - :math:`A_f` are the feature maps
    - :math:`\alpha_f = \frac{1}{HW} \sum_{i,j}
      \frac{\partial y_k}{\partial A_f(i,j)}` are channel-wise weights
    - :math:`y_k` is the class score (here averaged over spatial pixels)

    Parameters
    ----------
    model : nn.Module
        Segmentation model outputting logits of shape ``[B,C,H,W]``.
    input_tensor : torch.Tensor
        Single input image of shape ``[1,C,H,W]``. Gradients must be enabled.
    target_class : int
        Class index ``0..num_classes-1`` for which to compute Grad-CAM.
    target_layer : nn.Module
        Layer to hook for feature maps and gradients (e.g. ``model.get_cam_layer()``).

    Returns
    -------
    torch.Tensor
        Heatmap of shape ``[1,1,H,W]`` normalized to ``[0,1]``.
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
        logits = model(input_tensor)

        target_score = logits[:, target_class].mean()
        model.zero_grad()
        target_score.backward()

        feats = activations["value"]   # [1,F,h,w]
        grads = gradients["value"]     # [1,F,h,w]

        weights = grads.mean(dim=(2, 3), keepdim=True)     # [1,F,1,1]
        cam = (weights * feats).sum(dim=1, keepdim=True)   # [1,1,h,w]
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )

        # Normalize
        cmin, cmax = cam.min(), cam.max()
        if (cmax - cmin) > 1e-6:
            cam = (cam - cmin) / (cmax - cmin)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach()
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


# ---------------------------------------------------------
#  Integrated Gradients
# ---------------------------------------------------------
def integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor | None = None,
    steps: int = 50,
) -> torch.Tensor:
    r"""
    Compute **Integrated Gradients** (IG) for segmentation.

    Integrated Gradients approximates the path integral:

    .. math::

        \mathrm{IG}(x)
        = (x - x_0)
          \times
          \int_{0}^{1}
            \frac{\partial f(x_0 + \alpha (x - x_0))}
            {\partial x}
          \, d\alpha,

    where :math:`x_0` is a baseline (typically zeros).

    Parameters
    ----------
    model : nn.Module
        Segmentation model.
    input_tensor : torch.Tensor
        Input image ``[1,C,H,W]``.
    target_class : int
        Class index whose score to differentiate.
    baseline : torch.Tensor, optional
        Baseline image ``[1,C,H,W]``. Default: zeros.
    steps : int
        Number of steps for Riemann sum approximation.

    Returns
    -------
    torch.Tensor
        Attribution map of shape ``[1,C,H,W]``.
    """
    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    input_tensor = input_tensor.detach()
    baseline = baseline.detach()

    scaled_inputs = [
        baseline + (i / steps) * (input_tensor - baseline)
        for i in range(1, steps + 1)
    ]

    total_gradients = torch.zeros_like(input_tensor)

    for x in scaled_inputs:
        x = x.clone().detach().requires_grad_(True)
        logits = model(x)
        target_score = logits[:, target_class].mean()

        model.zero_grad()
        target_score.backward()

        total_gradients += x.grad.detach()

    avg_gradients = total_gradients / steps
    attributions = (input_tensor - baseline) * avg_gradients
    return attributions.detach()


# ---------------------------------------------------------
#  Neural PCA Helper Functions
# ---------------------------------------------------------
def _get_classifier_weight_vector(model: nn.Module, class_id: int) -> torch.Tensor:
    r"""
    Extract the final 1×1-conv classifier weight vector :math:`w_k`.

    For class :math:`k`:

    .. math::
        w_k = \text{Conv1x1.weight}[k]

    Parameters
    ----------
    model : nn.Module
        Segmentation model with an ``outc`` or final conv layer.
    class_id : int
        Target class index.

    Returns
    -------
    torch.Tensor
        Weight vector ``[F]``.
    """
    outc = getattr(model, "outc", None)
    if outc is None:
        raise ValueError("Model has no attribute `outc`.")

    if hasattr(outc, "conv"):
        weight = outc.conv.weight
    elif isinstance(outc, nn.Conv2d):
        weight = outc.weight
    else:
        raise ValueError("Could not find classifier weight.")

    return weight[class_id, :, 0, 0].detach().clone()


def _extract_phi_batch(
    model: nn.Module,
    x: torch.Tensor,
    cam_layer: nn.Module,
) -> torch.Tensor:
    r"""
    Extract **φ(x)** — global average pooled features from the CAM layer.

    .. math::
        \phi(x) = \mathrm{GAP}(A(x)) \in \mathbb{R}^F

    Parameters
    ----------
    x : torch.Tensor
        Input batch ``[B,C,H,W]``.
    cam_layer : nn.Module
        Layer to hook activations from.

    Returns
    -------
    torch.Tensor
        Feature vectors ``[B,F]``.
    """
    activations: Dict[str, torch.Tensor] = {}

    def hook(_, __, out):
        activations["feat"] = out

    handle = cam_layer.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(x)
    handle.remove()

    feat = activations["feat"]          # [B,F,h,w]
    phi = feat.mean(dim=(2, 3))         # GAP -> [B,F]
    return phi


# ---------------------------------------------------------
#  Neural PCA
# ---------------------------------------------------------
def compute_class_neural_pca_features(
    model: nn.Module,
    dataset,
    device: torch.device,
    class_ids: Sequence[int],
    max_samples_per_class: int = 200,
    n_components: int = 5,
    min_per_class: int = 10,
) -> Dict[int, Dict[str, object]]:
    r"""
    Compute **Neural PCA** for each class, following the lecture slides.

    We compute:

    - Feature vectors :math:`\phi(x)`
    - Classifier weights :math:`w_k`
    - Class-specific embedding:

      .. math::
          \psi_k(x) = w_k \odot \phi(x)

    Then perform PCA over the set :math:`\{\psi_k(x_i)\}` for each class.

    Parameters
    ----------
    model : nn.Module
        Trained segmentation network.
    dataset :
        Dataset providing ``(image, mask)`` samples.
    device : torch.device
        Device for model inference.
    class_ids : sequence of int
        Class indices to process.
    max_samples_per_class : int
        Maximum number of samples to use per class.
    n_components : int
        Number of PCA components to retain.
    min_per_class : int
        Minimum samples required to run PCA.

    Returns
    -------
    dict
        Mapping ``class_id -> PCA results`` with keys:
        - ``mean_psi`` : ``[D]``
        - ``eigvecs`` : ``[L,D]``
        - ``eigvals`` : ``[L]``
        - ``alphas``  : ``[N,L]`` projection scores
        - ``indices`` : dataset indices used
        - ``psi``     : raw psi vectors ``[N,D]``
    """
    model.eval()
    cam_layer = model.get_cam_layer()

    print("[neural PCA] Scanning dataset for class presence...")
    class_to_indices: Dict[int, List[int]] = {c: [] for c in class_ids}

    for idx in range(len(dataset)):
        _, mask = dataset[idx]
        mask_np = mask.numpy()
        for c in class_ids:
            if (mask_np == c).any():
                class_to_indices[c].append(idx)

    results = {}

    for c in class_ids:
        indices = class_to_indices[c]
        if len(indices) < min_per_class:
            print(f"[neural PCA] Class {c}: {len(indices)} samples < min_per_class.")
            continue

        if len(indices) > max_samples_per_class:
            indices = indices[:max_samples_per_class]

        print(f"[neural PCA] Class {c}: using {len(indices)} samples.")

        w_k = _get_classifier_weight_vector(model, c).to(device)
        psi_list = []

        for idx in indices:
            img, _ = dataset[idx]
            x = img.unsqueeze(0).to(device)
            phi = _extract_phi_batch(model, x, cam_layer)[0]
            psi_list.append((w_k * phi).cpu())

        X = torch.stack(psi_list)      # [N,D]
        X_mean = X.mean(dim=0, keepdim=True)
        X_centered = X - X_mean

        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        L = min(n_components, Vh.shape[0])

        eigvecs = Vh[:L]                       # [L,D]
        eigvals = (S[:L] ** 2) / max(1, len(X) - 1)

        diff = X - X_mean
        proj = diff @ eigvecs.T               # [N,L]
        ones_dot_v = eigvecs.sum(dim=1)       # [L]
        alphas = proj * ones_dot_v

        results[c] = {
            "mean_psi": X_mean[0],
            "eigvecs": eigvecs,
            "eigvals": eigvals,
            "alphas": alphas,
            "indices": indices,
            "psi": X,
        }

    return results


# ---------------------------------------------------------
#  Neural PCA Visualization
# ---------------------------------------------------------
def show_top_neural_pca_images_for_class(
    neural_pca_results,
    dataset,
    class_id: int,
    component_idx: int = 0,
    top_k: int = 6,
):
    r"""
    Visualize the **top-k images** that maximally activate a neural PCA
    component for a given class.

    For PCA component :math:`v_\ell`, the ranking uses:

    .. math::
        \alpha_\ell^{(k)}(x_i)

    Parameters
    ----------
    neural_pca_results : dict
        Output of :func:`compute_class_neural_pca_features`.
    dataset :
        Dataset that returns ``(image, mask)``.
    class_id : int
        Class index to visualize.
    component_idx : int
        PCA component index (0-based).
    top_k : int
        Number of top activating images to display.
    """
    AI4MARS_CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]

    if class_id not in neural_pca_results:
        print(f"No neural PCA info for class {class_id}.")
        return

    info = neural_pca_results[class_id]
    alphas = info["alphas"]
    indices = info["indices"]
    class_name = AI4MARS_CLASS_NAMES[class_id]

    if component_idx >= alphas.shape[1]:
        print("Component out of range.")
        return

    comp_scores = alphas[:, component_idx]
    N = comp_scores.numel()
    k = min(top_k, N)

    top_vals, top_pos = torch.topk(comp_scores, k=k)

    print(
        f"\nClass '{class_name}' (id={class_id}), PCA component {component_idx+1}, "
        f"showing top {k}/{N} images."
    )

    ncols = min(k, 5)
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

    if nrows == 1:
        axes = [axes]

    for rank in range(k):
        pos = int(top_pos[rank])
        ds_idx = indices[pos]
        img, _ = dataset[ds_idx]
        img_np = img[0].cpu().numpy()

        r, c = divmod(rank, ncols)
        ax = axes[r][c]
        ax.imshow(img_np, cmap="gray")
        ax.set_title(f"rank {rank+1}\nα={float(top_vals[rank]):.2f}\nidx={ds_idx}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def normalize_map(t: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor linearly to the ``[0,1]`` range.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Normalized tensor.
    """
    t = t.detach()
    t_min, t_max = t.min(), t.max()
    return (t - t_min) / (t_max - t_min + 1e-8) if (t_max - t_min) > 1e-6 else torch.zeros_like(t)


# ---------------------------------------------------------
# Full Example Visualization
# ---------------------------------------------------------
def explain_per_class_examples(
    model,
    dataset,
    device,
    num_examples_per_class: int = 2,
    ig_steps: int = 32,
):
    """
    Generate combined explainability visualizations for each class:

    - Input image  
    - Grad-CAM  
    - Integrated Gradients  

    Parameters
    ----------
    model : nn.Module
        Trained segmentation model.
    dataset :
        Dataset providing ``(img, mask)``.
    device : torch.device
        Compute device.
    num_examples_per_class : int
        Number of examples per class.
    ig_steps : int
        IG integration steps.
    """
    AI4MARS_CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
    model.eval()
    cam_layer = model.get_cam_layer()

    class_to_indices = {c: [] for c in range(len(AI4MARS_CLASS_NAMES))}
    print("Scanning dataset once to find examples per class...")

    for idx in range(len(dataset)):
        _, mask = dataset[idx]
        mask_np = mask.numpy()
        for c in class_to_indices:
            if (mask_np == c).any():
                class_to_indices[c].append(idx)

        if all(len(v) >= num_examples_per_class for v in class_to_indices.values()):
            break

    for c, indices in class_to_indices.items():
        name = AI4MARS_CLASS_NAMES[c]

        if len(indices) == 0:
            print(f"No examples for class {name}.")
            continue

        chosen = random.sample(indices, min(num_examples_per_class, len(indices)))

        print(
            f"\n=== Class '{name}' (id={c}) | "
            f"{len(indices)} total examples, showing {len(chosen)} ==="
        )

        for i, idx in enumerate(chosen):
            print(f"- Example {i+1}: dataset idx {idx}")

            img, _ = dataset[idx]
            x = img.unsqueeze(0).to(device)

            cam = grad_cam(model, x, c, cam_layer)
            ig = integrated_gradients(model, x, c, baseline=torch.zeros_like(x), steps=ig_steps)

            img_np = x[0, 0].cpu().numpy()
            cam_np = cam[0, 0].cpu().numpy()
            ig_np = normalize_map(ig[0, 0]).cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 6))
            axes[0].imshow(img_np, cmap="gray");  axes[0].set_title("Input"); axes[0].axis("off")
            axes[1].imshow(img_np, cmap="gray"); axes[1].imshow(cam_np, cmap="jet", alpha=0.5)
            axes[1].set_title(f"Grad-CAM ({name})"); axes[1].axis("off")
            axes[2].imshow(img_np, cmap="gray"); axes[2].imshow(ig_np, cmap="inferno", alpha=0.5)
            axes[2].set_title(f"Integrated Gradients ({name})"); axes[2].axis("off")

            plt.tight_layout()
            plt.show()
