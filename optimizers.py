# optimizers.py
"""
Optimizer helpers.

If `muon` is installed (https://github.com/KellerJordan/Muon) we use Muon,
otherwise we fall back to AdamW.

Muon is an optimizer that orthogonalizes updates (via Newton-Schulz iteration)
and has been shown to speed up training for various deep networks. :contentReference[oaicite:7]{index=7}
"""

from __future__ import annotations

import torch

try:
    # As per official repo usage example. :contentReference[oaicite:8]{index=8}
    from muon import Muon  # type: ignore

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

    - If Muon is available and use_muon=True: use Muon on all parameters.
      (For serious experiments you might follow the recommended pattern:
       Muon for 2D weights + AdamW for biases, norms, etc., but we keep this
       simple here.)
    - Else: use AdamW on all parameters.
    """
    params = model.parameters()

    if use_muon and _HAS_MUON:
        print("[optimizers] Using Muon optimizer.")
        return Muon(params, lr=lr)  # type: ignore[arg-type]
    else:
        if use_muon and not _HAS_MUON:
            print(
                "[optimizers] Muon requested but not found. "
                "Install via `pip install git+https://github.com/KellerJordan/Muon`. "
                "Falling back to AdamW."
            )
        else:
            print("[optimizers] Using AdamW optimizer.")
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
