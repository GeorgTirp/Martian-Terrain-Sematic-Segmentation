# optimizers.py
"""
Optimizer helpers.

Priority:
1. Muon (if installed and use_muon=True).
2. NAdam (NadamW-style) if available in torch.optim.
3. AdamW as a safe fallback.

Also provides a cosine LR schedule with warmup.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

try:
    from muon import Muon  # KellerJordan Muon
    _HAS_MUON = True
except Exception:
    Muon = None  # type: ignore
    _HAS_MUON = False


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    use_muon: bool = True,
) -> torch.optim.Optimizer:
    """
    Create an optimizer for the given model.

    Logic:
        - If Muon is available and use_muon=True: use Muon on all trainable params.
        - Else if torch.optim.NAdam exists: use NAdam with weight decay (NadamW-style).
        - Else: fallback to AdamW.
    """
    # Only trainable parameters
    params_list = [p for p in model.parameters() if p.requires_grad]

    # 1) Muon
    if use_muon and _HAS_MUON:
        print("[optimizers] Using Muon optimizer.")
        # Muon signature: Muon(params, lr=0.02, weight_decay=0, momentum=0.95)
        return Muon(params_list, lr=lr, weight_decay=weight_decay)  # type: ignore[arg-type]

    # 2) NAdam (NadamW-ish)
    if hasattr(torch.optim, "NAdam"):
        print("[optimizers] Using NAdam (NadamW-style) optimizer.")
        # In recent PyTorch versions, NAdam supports weight_decay.
        return torch.optim.NAdam(params_list, lr=lr, weight_decay=weight_decay)  # type: ignore[attr-defined]

    # 3) AdamW fallback
    if use_muon and not _HAS_MUON:
        print(
            "[optimizers] Muon requested but not found. "
            "Install via `pip install git+https://github.com/KellerJordan/Muon`."
        )
    print("[optimizers] Using AdamW optimizer.")
    return torch.optim.AdamW(params_list, lr=lr, weight_decay=weight_decay)


def create_cosine_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine LR schedule with linear warmup, similar to Hugging Face's implementation.

    Args:
        optimizer: the optimizer to wrap.
        num_warmup_steps: number of warmup steps (usually ~10% of total steps).
        num_training_steps: total number of training steps (epochs * steps_per_epoch).
        num_cycles: number of cosine cycles (0.5 = decay to 0 once; 1.0 = full up+down).

    Returns:
        A torch.optim.lr_scheduler.LambdaLR instance.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup from 0 -> 1
            return float(current_step) / max(1, num_warmup_steps)

        # After warmup: cosine decay from 1 -> 0
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
