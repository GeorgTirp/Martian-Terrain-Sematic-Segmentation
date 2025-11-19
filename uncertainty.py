# uncertainty.py
from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Per-pixel predictive entropy from logits.

    Args:
        logits: [B, C, H, W]

    Returns:
        entropy: [B, H, W]  (higher = more uncertain)
    """
    probs = F.softmax(logits, dim=1)                  # [B,C,H,W]
    log_probs = torch.log(probs.clamp_min(1e-8))
    entropy = -(probs * log_probs).sum(dim=1)         # [B,H,W]
    return entropy


@torch.no_grad()
def max_prob_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """
    1 - max softmax probability per pixel.

    Args:
        logits: [B, C, H, W]

    Returns:
        unc: [B, H, W] in [0, 1] (approx), higher = more uncertain.
    """
    probs = F.softmax(logits, dim=1)
    max_p, _ = probs.max(dim=1)                       # [B,H,W]
    unc = 1.0 - max_p
    return unc
