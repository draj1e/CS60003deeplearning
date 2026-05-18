from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        valid = targets != self.ignore_index
        safe_targets = targets.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_targets, self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1).float()
        probs = probs * valid
        one_hot = one_hot * valid
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        denominator = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


def build_loss(name: str, num_classes: int) -> nn.Module:
    ce = nn.CrossEntropyLoss(ignore_index=255)
    dice = DiceLoss(num_classes=num_classes)
    if name == "ce":
        return ce
    if name == "dice":
        return dice
    if name in {"ce_dice", "combo"}:
        return lambda logits, targets: ce(logits, targets) + dice(logits, targets)
    raise ValueError(f"unknown loss: {name}")
