# uncertainty.py
"""
Uncertainty quantification helpers for semantic segmentation.

This module provides simple, post-hoc **epistemic-style** uncertainty
proxies based purely on the softmax output of a model:

- :func:`predictive_entropy`:
    Per-pixel predictive entropy :math:`H(p(y \mid x))`.
- :func:`max_prob_uncertainty`:
    ``1 - max_softmax_prob`` per pixel.

Both functions take pre-softmax logits of shape ``[B, C, H, W]`` and
return per-pixel uncertainty maps of shape ``[B, H, W]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
    r"""
    Compute **per-pixel predictive entropy** from logits.

    For each pixel, if :math:`p_c` is the softmax probability of class
    :math:`c \in \{1, \dots, C\}`, the entropy is:

    .. math::

        H(p) = -\sum_{c=1}^C p_c \log p_c

    Higher values indicate more uncertainty (more "spread out" distribution).

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs of shape ``[B, C, H, W]``.

    Returns
    -------
    torch.Tensor
        Entropy map of shape ``[B, H, W]``, where larger values
        correspond to more uncertain predictions.
    """
    probs = F.softmax(logits, dim=1)                      # [B,C,H,W]
    log_probs = torch.log(probs.clamp_min(1e-8))         # numerical safety
    entropy = -(probs * log_probs).sum(dim=1)            # [B,H,W]
    return entropy


@torch.no_grad()
def max_prob_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    r"""
    Compute a simple **max-softmax-based** uncertainty proxy.

    For each pixel, let:

    .. math::

        p_{\max} = \max_c p_c,

    where :math:`p_c` is the softmax probability of class :math:`c`.

    We define the uncertainty score as:

    .. math::

        u = 1 - p_{\max}.

    If the model is very confident (e.g. :math:`p_{\max} \approx 1`),
    then :math:`u \approx 0`. If the model is unsure (probabilities are
    more uniform), :math:`p_{\max}` is lower and :math:`u` increases.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs of shape ``[B, C, H, W]``.

    Returns
    -------
    torch.Tensor
        Uncertainty scores of shape ``[B, H, W]``, typically in the
        range :math:`[0, 1]`, where larger values indicate higher
        uncertainty.
    """
    probs = F.softmax(logits, dim=1)                     # [B,C,H,W]
    max_p, _ = probs.max(dim=1)                          # [B,H,W]
    unc = 1.0 - max_p
    return unc
