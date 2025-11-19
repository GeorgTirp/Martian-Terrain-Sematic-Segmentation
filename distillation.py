# distillation.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationKDLoss(nn.Module):
    """
    Pixel-wise knowledge distillation loss for segmentation:

        L = alpha * CE(student, y) + (1 - alpha) * T^2 * KL(
                softmax(teacher/T) || softmax(student/T)
            )

    where KL is computed per-pixel and averaged over all pixels.
    """

    def __init__(self, ignore_index: int, alpha: float = 0.5, T: float = 2.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.T = T
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def forward(self, student_logits, teacher_logits, targets):
        """
        Args:
            student_logits: [B,C,H,W]
            teacher_logits: [B,C,H,W] (no grad)
            targets: [B,H,W] long
        """
        B, C, H, W = student_logits.shape

        # --- CE term (on ground truth) ---
        ce_loss = self.ce(student_logits, targets)  # [B,H,W]
        valid_mask = (targets != self.ignore_index).float()
        ce_loss = (ce_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

        # --- KD term (on teacher vs student logits) ---
        # Flatten spatial dims
        T = self.T
        s_logits = student_logits / T
        t_logits = teacher_logits / T

        s_log_probs = F.log_softmax(s_logits, dim=1)  # [B,C,H,W]
        t_probs = F.softmax(t_logits, dim=1)          # [B,C,H,W]

        # mask out ignore_index pixels
        valid_mask_4d = valid_mask.unsqueeze(1)  # [B,1,H,W]

        kl_per_pixel = F.kl_div(
            s_log_probs,
            t_probs,
            reduction="none",
        ).sum(dim=1)  # [B,H,W]

        kd_loss = (kl_per_pixel * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        kd_loss = kd_loss * (T * T)

        return self.alpha * ce_loss + (1.0 - self.alpha) * kd_loss
