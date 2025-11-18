# optimizers.py
"""
Optimizer helpers.

If `muon` is installed (https://github.com/KellerJordan/Muon) we use Muon,
otherwise we fall back to AdamW.

Muon is an optimizer that orthogonalizes updates (via Newton-Schulz iteration)
and has been shown to speed up training for various deep networks. :contentReference[oaicite:7]{index=7}
"""

# optimizers.py
from __future__ import annotations

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

    - If Muon is available and use_muon=True: use Muon on all parameters.
    - Else: use AdamW on all parameters.
    """
    # IMPORTANT: Muon expects a *list* of torch.nn.Parameter, not a generator.
    params_list = [p for p in model.parameters()]

    if use_muon and _HAS_MUON:
        print("[optimizers] Using Muon optimizer.")
        # KellerJordan's Muon signature: Muon(params, lr=0.02, weight_decay=0, momentum=0.95)
        return Muon(params_list, lr=lr, weight_decay=weight_decay)  # type: ignore[arg-type]
    else:
        if use_muon and not _HAS_MUON:
            print(
                "[optimizers] Muon requested but not found. "
                "Install via `pip install git+https://github.com/KellerJordan/Muon`. "
                "Falling back to AdamW."
            )
        else:
            print("[optimizers] Using AdamW optimizer.")
        return torch.optim.AdamW(params_list, lr=lr, weight_decay=weight_decay)
