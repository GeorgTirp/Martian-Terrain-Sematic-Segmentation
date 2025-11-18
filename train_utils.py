# train_utils.py
"""
Training and evaluation helpers for semantic segmentation.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import AI4MARS_IGNORE_INDEX


def _compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = AI4MARS_IGNORE_INDEX,
) -> Dict[str, float]:
    """
    Computes pixel accuracy and per-class IoUs for a single batch.
    """

    with torch.no_grad():
        preds = logits.argmax(dim=1)  # [B,H,W]

        valid_mask = targets != ignore_index
        if valid_mask.sum() == 0:
            return {"pixel_acc": 0.0, "miou": 0.0}

        correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
        total = valid_mask.sum().item()
        pixel_acc = correct / max(total, 1)

        ious = []
        for c in range(num_classes):
            pred_c = (preds == c) & valid_mask
            target_c = (targets == c) & valid_mask

            intersection = (pred_c & target_c).sum().item()
            union = (pred_c | target_c).sum().item()

            if union == 0:
                continue
            ious.append(intersection / union)

        miou = float(sum(ious) / max(len(ious), 1))
        return {"pixel_acc": float(pixel_acc), "miou": miou}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    loss_fn: nn.Module | None = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Returns:
        dict with average loss, pixel accuracy, and mean IoU.
    """
    model.train()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(ignore_index=AI4MARS_IGNORE_INDEX)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    running_loss = 0.0
    total_samples = 0
    acc_sum = 0.0
    miou_sum = 0.0

    for imgs, masks in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        batch_size = imgs.size(0)
        total_samples += batch_size

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            loss = loss_fn(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size

        metrics = _compute_batch_metrics(logits, masks, num_classes)
        acc_sum += metrics["pixel_acc"] * batch_size
        miou_sum += metrics["miou"] * batch_size

    avg_loss = running_loss / max(total_samples, 1)
    avg_acc = acc_sum / max(total_samples, 1)
    avg_miou = miou_sum / max(total_samples, 1)

    return {"loss": avg_loss, "pixel_acc": avg_acc, "miou": avg_miou}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    """
    Evaluate on a validation/test set.
    """
    model.eval()

    loss_fn = nn.CrossEntropyLoss(ignore_index=AI4MARS_IGNORE_INDEX)

    running_loss = 0.0
    total_samples = 0
    acc_sum = 0.0
    miou_sum = 0.0

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            batch_size = imgs.size(0)
            total_samples += batch_size

            logits = model(imgs)
            loss = loss_fn(logits, masks)

            running_loss += loss.item() * batch_size

            metrics = _compute_batch_metrics(logits, masks, num_classes)
            acc_sum += metrics["pixel_acc"] * batch_size
            miou_sum += metrics["miou"] * batch_size

    avg_loss = running_loss / max(total_samples, 1)
    avg_acc = acc_sum / max(total_samples, 1)
    avg_miou = miou_sum / max(total_samples, 1)

    return {"loss": avg_loss, "pixel_acc": avg_acc, "miou": avg_miou}
