# train_utils.py
"""
Training and evaluation helpers for semantic segmentation.
"""

from __future__ import annotations

from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.auto import tqdm  # NEW: for nice progress bars

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
    use_tqdm: bool = False,
    epoch: int | None = None,
    num_epochs: int | None = None,
    scheduler: Optional[_LRScheduler] = None,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model, dataloader, optimizer, device, num_classes: as before.
        loss_fn: optional loss; defaults to CrossEntropy with ignore_index.
        use_amp: mixed precision flag.
        use_tqdm: if True, wraps the dataloader in a tqdm progress bar.
        epoch: current epoch index (1-based, for display only).
        num_epochs: total number of epochs (for display only).

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

    # wrap dataloader with tqdm if requested
    if use_tqdm:
        if epoch is not None and num_epochs is not None:
            desc = f"Train Epoch {epoch}/{num_epochs}"
        else:
            desc = "Training"
        data_iter = tqdm(dataloader, desc=desc, unit="batch", leave=False)
    else:
        data_iter = dataloader

    for imgs, masks in data_iter:
        imgs = imgs.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).long()

        batch_size = imgs.size(0)
        total_samples += batch_size

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            loss = loss_fn(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size

        metrics = _compute_batch_metrics(logits, masks, num_classes)
        acc_sum += metrics["pixel_acc"] * batch_size
        miou_sum += metrics["miou"] * batch_size

        # optional live postfix on the progress bar
        if use_tqdm:
            avg_loss_so_far = running_loss / max(total_samples, 1)
            avg_miou_so_far = miou_sum / max(total_samples, 1)
            data_iter.set_postfix(
                loss=f"{avg_loss_so_far:.3f}",
                mIoU=f"{avg_miou_so_far:.3f}",
            )

    avg_loss = running_loss / max(total_samples, 1)
    avg_acc = acc_sum / max(total_samples, 1)
    avg_miou = miou_sum / max(total_samples, 1)

    return {"loss": avg_loss, "pixel_acc": avg_acc, "miou": avg_miou}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_tqdm: bool = False,
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

    if use_tqdm:
        data_iter = tqdm(dataloader, desc="Eval", unit="batch", leave=False)
    else:
        data_iter = dataloader

    with torch.no_grad():
        for imgs, masks in data_iter:
            imgs = imgs.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).long()

            batch_size = imgs.size(0)
            total_samples += batch_size

            logits = model(imgs)
            loss = loss_fn(logits, masks)

            running_loss += loss.item() * batch_size

            metrics = _compute_batch_metrics(logits, masks, num_classes)
            acc_sum += metrics["pixel_acc"] * batch_size
            miou_sum += metrics["miou"] * batch_size

            if use_tqdm:
                avg_loss_so_far = running_loss / max(total_samples, 1)
                avg_miou_so_far = miou_sum / max(total_samples, 1)
                data_iter.set_postfix(
                    loss=f"{avg_loss_so_far:.3f}",
                    mIoU=f"{avg_miou_so_far:.3f}",
                )

    avg_loss = running_loss / max(total_samples, 1)
    avg_acc = acc_sum / max(total_samples, 1)
    avg_miou = miou_sum / max(total_samples, 1)

    return {"loss": avg_loss, "pixel_acc": avg_acc, "miou": avg_miou}


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        path: file path to save to (e.g. "checkpoints/best_model.pt").
        model: model to save (state_dict only).
        optimizer: optional optimizer (state_dict).
        scheduler: optional LR scheduler (state_dict).
        epoch: optional current epoch number.
        metrics: optional dict of metrics (e.g. best val loss/miou).
        extra: optional extra metadata dict (anything you like).
    """
    state: Dict[str, Any] = {
        "model_state": model.state_dict(),
    }

    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if metrics is not None:
        state["metrics"] = metrics
    if extra is not None:
        state["extra"] = extra

    torch.save(state, path)
    print(f"[checkpoint] Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        path: path to the checkpoint file.
        model: model to load weights into.
        optimizer: optional optimizer to restore.
        scheduler: optional scheduler to restore.
        map_location: where to map loaded tensors.

    Returns:
        A dict with any extra info stored in the checkpoint, e.g.:
          {
            "epoch": int | None,
            "metrics": dict | None,
            "extra": dict | None,
          }
    """
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state"])
    print(f"[checkpoint] Loaded model weights from {path}")

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("[checkpoint] Restored optimizer state.")

    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        print("[checkpoint] Restored scheduler state.")

    info: Dict[str, Any] = {
        "epoch": checkpoint.get("epoch", None),
        "metrics": checkpoint.get("metrics", None),
        "extra": checkpoint.get("extra", None),
    }
    return info
