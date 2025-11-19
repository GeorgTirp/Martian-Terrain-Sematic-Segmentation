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

from typing import List, Tuple, Dict, Sequence
import random
import math
import torch
import torch.nn.functional as F
from torch import nn
from matplotlib import pyplot as plt
from PIL import Image


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


def _get_classifier_weight_vector(model: nn.Module, class_id: int) -> torch.Tensor:
    """
    Get the classifier weight vector w_k for class k from the final 1x1 conv.

    Assumes model.outc is your OutConv wrapper:
        class OutConv(nn.Module):
            def __init__(...):
                self.conv = nn.Conv2d(...)
            def forward(self, x):
                return self.conv(x)
    """
    outc = getattr(model, "outc", None)
    if outc is None:
        raise ValueError("Model has no attribute 'outc'; cannot extract classifier weights.")

    if hasattr(outc, "conv") and isinstance(outc.conv, nn.Conv2d):
        weight = outc.conv.weight  # [num_classes, F, 1, 1]
    elif isinstance(outc, nn.Conv2d):
        weight = outc.weight
    else:
        raise ValueError("Could not find classifier conv weight for neural PCA.")

    # w_k is [F]
    w_k = weight[class_id, :, 0, 0]
    return w_k.detach().clone()


def _extract_phi_batch(
    model: nn.Module,
    x: torch.Tensor,
    cam_layer: nn.Module,
) -> torch.Tensor:
    """
    Extract φ(x) for a batch: global average pooled features from `cam_layer`.

    Args:
        x: [B,C,H,W]

    Returns:
        phi: [B,F], where F is #channels of cam_layer.
    """
    activations: Dict[str, torch.Tensor] = {}

    def hook(module, inp, out):
        activations["feat"] = out

    handle = cam_layer.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(x)
    handle.remove()

    feat = activations["feat"]  # [B,F,H',W']
    phi = feat.mean(dim=(2, 3))  # GAP over spatial dims -> [B,F]
    return phi


def compute_class_neural_pca_features(
    model: nn.Module,
    dataset,
    device: torch.device,
    class_ids: Sequence[int],
    max_samples_per_class: int = 200,
    n_components: int = 5,
    min_per_class: int = 10,
) -> Dict[int, Dict[str, object]]:
    """
    Compute class-wise neural PCA in feature space, like in the slides:

      φ(x) ∈ R^D : penultimate features (here: GAP(cam_layer(x))).
      w_k ∈ R^D  : classifier weights for class k.
      ψ_k(x) = w_k ⊙ φ(x) ∈ R^D.

    We then compute PCA on {ψ_k(x_i)} over all images that contain class k
    (up to `max_samples_per_class`), and get:

      - mean_psi: ψ̄_k
      - eigvecs: principal directions v_l ∈ R^D
      - eigvals: corresponding eigenvalues
      - alphas: α_l^(k)(x_i) for each used sample & component (for ranking images)
      - indices: dataset indices used
      - psi: raw ψ_k(x_i) vectors

    Args:
        model: your segmentation model.
        dataset: e.g. train_loader.dataset or test_loader.dataset.
        device: torch.device.
        class_ids: iterable of class indices (e.g. range(num_classes)).
        max_samples_per_class: limit of samples used per class for PCA.
        n_components: how many PCA components per class to keep.
        min_per_class: skip classes with fewer usable samples.

    Returns:
        dict[class_id] -> {
            "mean_psi": [D],
            "eigvecs": [L,D],  # first L PCs
            "eigvals": [L],
            "alphas": [N,L],   # α_l^(k)(x_i) for each sample & component
            "indices": list of dataset indices used,
            "psi": [N,D],      # ψ_k(x_i) vectors
        }
    """
    model.eval()
    cam_layer = model.get_cam_layer()

    # --- 1) Pre-scan: find candidate indices per class from masks ---
    print("[neural PCA] Scanning dataset for class presence...")
    class_to_indices: Dict[int, List[int]] = {c: [] for c in class_ids}
    for idx in range(len(dataset)):
        _, mask = dataset[idx]  # mask is [H,W] tensor on CPU
        mask_np = mask.numpy()
        for c in class_ids:
            if (mask_np == c).any():
                class_to_indices[c].append(idx)

    results: Dict[int, Dict[str, object]] = {}

    # --- 2) For each class, build ψ_k(x) vectors and run PCA ---
    for c in class_ids:
        indices = class_to_indices[c]
        if len(indices) < min_per_class:
            print(
                f"[neural PCA] Class {c}: only {len(indices)} samples found "
                f"(min_per_class={min_per_class}) – skipping."
            )
            continue

        # Limit for efficiency
        if len(indices) > max_samples_per_class:
            indices = indices[:max_samples_per_class]

        print(
            f"[neural PCA] Class {c}: using {len(indices)} samples "
            f"for PCA out of {len(class_to_indices[c])} candidates."
        )

        # Classifier weight vector w_k
        w_k = _get_classifier_weight_vector(model, c).to(device)  # [D]
        psi_list: List[torch.Tensor] = []

        # Collect ψ_k(x) for each image
        for idx in indices:
            img, _ = dataset[idx]
            x = img.unsqueeze(0).to(device)  # [1,C,H,W]

            phi = _extract_phi_batch(model, x, cam_layer)[0]  # [D]
            psi = w_k * phi  # ψ_k(x) ∈ R^D
            psi_list.append(psi.cpu())

        X = torch.stack(psi_list, dim=0)  # [N,D]
        N, D = X.shape

        if N < min_per_class:
            print(
                f"[neural PCA] Class {c}: after filtering, only {N} ψ vectors – skipping."
            )
            continue

        # --- PCA via SVD on centered data ---
        X_mean = X.mean(dim=0, keepdim=True)      # [1,D] = ψ̄_k
        X_centered = X - X_mean                   # [N,D]

        # SVD: X_centered = U S Vh, rows of Vh are PCs
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)  # U[N,N], S[N], Vh[N,D]

        L = min(n_components, Vh.shape[0])
        V = Vh[:L]                                # [L,D] principal directions v_l
        eigvals = (S[:L] ** 2) / max(1, N - 1)    # [L]

        # --- α_l^(k)(x_i) = <1, v_l> * <ψ_k(x_i) - ψ̄_k, v_l> ---
        diff = X - X_mean  # [N,D]
        ones_dot_v = V.sum(dim=1)                # [L] = <1, v_l>
        proj = diff @ V.T                        # [N,L] = <ψ - ψ̄, v_l>
        alphas = proj * ones_dot_v               # [N,L]

        results[c] = {
            "mean_psi": X_mean[0],   # [D]
            "eigvecs": V,            # [L,D]
            "eigvals": eigvals,      # [L]
            "alphas": alphas,        # [N,L]
            "indices": indices,      # list of dataset indices
            "psi": X,                # [N,D]
        }

    return results

def show_top_neural_pca_images_for_class(
    neural_pca_results,
    dataset,
    class_id: int,
    component_idx: int = 0,
    top_k: int = 6,
    easter_egg: bool = True,
    alien_image_path: str | None = "A_grayscale_photograph_captures_an_astronaut_on_th.png",
):
    """
    Show the top-k REAL images that maximally activate NPCA component `component_idx`
    for class `class_id` (like "Max. activating train images - N-PCA Comp. l" in the slides).

    If `easter_egg=True` and `class_id == 4`, show a special alien NPCA image
    (astronaut with "Hire me, please!" flag) instead of using neural_pca_results.
    """

    # Include an extra label for the easter egg class id=4
    AI4MARS_CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock", "alien"]

    # --- Easter-egg path: fake 'alien' NPCA ---
    if easter_egg and class_id == 4:
        print("\nClass 'alien' (id=4), NPCA component 1, showing 1 / 1 images.")

        if alien_image_path is None:
            print("No alien_image_path provided; nothing to display.")
            return

        img = Image.open(alien_image_path)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(img, cmap="gray")
        ax.set_title("rank 1\nα=?\nidx=alien-1")
        ax.axis("off")

        plt.tight_layout()
        plt.show()
        return

    # --- Normal NPCA path for real terrain classes ---
    if class_id not in neural_pca_results:
        print(f"No neural PCA info stored for class {class_id}.")
        return

    info = neural_pca_results[class_id]
    alphas = info["alphas"]          # [N,L]
    indices = info["indices"]        # list of dataset indices
    class_name = AI4MARS_CLASS_NAMES[class_id]

    if component_idx >= alphas.shape[1]:
        print(
            f"Component {component_idx} out of range, only {alphas.shape[1]} components stored."
        )
        return

    comp_scores = alphas[:, component_idx]  # [N]
    N = comp_scores.numel()
    k = min(top_k, N)

    top_vals, top_pos = torch.topk(comp_scores, k=k)
    print(
        f"\nClass '{class_name}' (id={class_id}), NPCA component {component_idx+1}, "
        f"showing top {k} / {N} images."
    )

    ncols = min(k, 5)
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    for rank in range(k):
        pos = int(top_pos[rank])
        ds_idx = indices[pos]
        img, mask = dataset[ds_idx]  # img [C,H,W]
        img_np = img[0].cpu().numpy()  # assuming grayscale

        r = rank // ncols
        c = rank % ncols
        ax = axes[r][c]
        ax.imshow(img_np, cmap="gray")
        ax.set_title(f"rank {rank+1}\nα={float(top_vals[rank]):.2f}\nidx={ds_idx}")
        ax.axis("off")

    # hide any unused axes
    for r in range(nrows):
        for c in range(ncols):
            if r * ncols + c >= k:
                axes[r][c].axis("off")

    plt.tight_layout()
    plt.show()

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
    AI4MARS_CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
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


            # ----- Visualization -----
            img_np = input_img[0, 0].detach().cpu().numpy()

            cam_np = cam_map[0, 0].detach().cpu().numpy()
            ig_np = normalize_map(ig_attr[0, 0]).cpu().numpy()


            fig, axes = plt.subplots(1, 3, figsize=(12, 6))

            # Original image
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title("Input")
            axes[0].axis("off")

            # Grad-CAM overlay
            axes[1].imshow(img_np, cmap="gray")
            axes[1].imshow(cam_np, cmap="jet", alpha=0.5)
            axes[1].set_title(f"Grad-CAM ({class_name})")
            axes[1].axis("off")

            # Integrated Gradients overlay
            axes[2].imshow(img_np, cmap="gray")
            axes[2].imshow(ig_np, cmap="inferno", alpha=0.5)
            axes[2].set_title(f"Integrated Gradients ({class_name})")
            axes[2].axis("off")



            plt.tight_layout()
            plt.show()
