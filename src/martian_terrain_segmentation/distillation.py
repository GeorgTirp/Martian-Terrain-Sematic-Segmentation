# distillation.py
"""
Knowledge distillation utilities for semantic segmentation.

This module currently provides:

- :class:`SegmentationKDLoss`: a pixel-wise distillation loss combining
  standard cross-entropy with a KL divergence term between teacher and
  student logits, following the classic KD formulation:

  .. math::

      L = \\alpha \\cdot \\mathrm{CE}(y, s)
          + (1 - \\alpha) T^2
            \\cdot \\mathrm{KL}
            ( \\mathrm{softmax}(t/T) \\;\\Vert\\; \\mathrm{softmax}(s/T) ),

  where ``s`` and ``t`` denote student and teacher logits respectively,
  and the loss is averaged over all non-ignored pixels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationKDLoss(nn.Module):
    r"""
    Pixel-wise knowledge distillation loss for semantic segmentation.

    This loss combines a standard supervised cross-entropy term with a
    KL-divergence term that encourages the student to match the
    teacher's softened class probabilities (as in Hinton et al., 2015).

    The loss is defined as:

    .. math::

        L = \alpha \cdot \mathcal{L}_{\text{CE}}(y, s)
          + (1 - \alpha) T^2
            \cdot \mathcal{L}_{\text{KD}}(t, s),

    where

    - :math:`\mathcal{L}_{\text{CE}}` is pixel-wise cross-entropy w.r.t.
      the ground truth labels :math:`y`.
    - :math:`\mathcal{L}_{\text{KD}}` is the pixel-wise KL divergence
      between teacher and student predictive distributions with
      temperature :math:`T`.

    The KL term is computed per pixel, masked by ``ignore_index``, and
    averaged over all valid pixels.

    Parameters
    ----------
    ignore_index : int
        Label value in the target mask that should be ignored when
        computing both the CE and KD terms (e.g. unlabeled/void pixels).
    alpha : float, default=0.5
        Trade-off parameter between the supervised CE loss and the
        distillation (KL) loss. ``alpha=1.0`` means pure CE,
        ``alpha=0.0`` means pure distillation.
    T : float, default=2.0
        Temperature used to soften the teacher and student logits in
        the KD term. Larger values produce softer probability
        distributions and typically richer distillation signals.

    Examples
    --------
    .. code-block:: python

        kd_loss_fn = SegmentationKDLoss(ignore_index=255, alpha=0.5, T=2.0)

        # student_logits, teacher_logits: [B, C, H, W]
        # targets: [B, H, W] with values in {0..C-1} or ignore_index
        loss = kd_loss_fn(student_logits, teacher_logits, targets)
    """

    def __init__(self, ignore_index: int, alpha: float = 0.5, T: float = 2.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.T = T
        # reduction="none" so we can mask and normalize manually over valid pixels
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the combined CE + KD loss for segmentation.

        Parameters
        ----------
        student_logits : torch.Tensor
            Logits from the student network of shape ``[B, C, H, W]``.
        teacher_logits : torch.Tensor
            Logits from the teacher network of shape ``[B, C, H, W]``.
            These are treated as fixed targets (no gradient should flow
            into the teacher).
        targets : torch.Tensor
            Ground-truth segmentation mask of shape ``[B, H, W]`` with
            integer class indices in ``[0, C-1]`` or ``ignore_index``.

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the total distillation loss.

        Notes
        -----
        The returned loss is:

        .. math::

            L = \\alpha \\cdot \\mathrm{CE}(y, s)
              + (1 - \\alpha) T^2
                \\cdot \\mathrm{KL}
                ( \\mathrm{softmax}(t/T) \\;\\Vert\\; \\mathrm{softmax}(s/T) ),

        where:

        - The cross-entropy term is averaged over all pixels with
          ``targets != ignore_index``.
        - The KL term is also averaged over valid pixels and scaled by
          :math:`T^2` following the original KD formulation.
        """
        B, C, H, W = student_logits.shape

        # --- CE term (on ground truth) ---
        ce_loss = self.ce(student_logits, targets)  # [B,H,W]
        valid_mask = (targets != self.ignore_index).float()
        ce_loss = (ce_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

        # --- KD term (on teacher vs student logits) ---
        T = self.T
        s_logits = student_logits / T
        t_logits = teacher_logits / T

        s_log_probs = F.log_softmax(s_logits, dim=1)  # [B,C,H,W]
        t_probs = F.softmax(t_logits, dim=1)          # [B,C,H,W]

        # mask out ignore_index pixels
        # valid_mask: [B,H,W]
        kl_per_pixel = F.kl_div(
            s_log_probs,
            t_probs,
            reduction="none",
        ).sum(dim=1)  # [B,H,W] after summing over C

        kd_loss = (kl_per_pixel * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        kd_loss = kd_loss * (T * T)

        return self.alpha * ce_loss + (1.0 - self.alpha) * kd_loss
