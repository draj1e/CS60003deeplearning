from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    mask = (targets >= 0) & (targets < num_classes)
    indices = num_classes * targets[mask].to(torch.int64) + preds[mask].to(torch.int64)
    return torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def mean_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    preds = logits.argmax(dim=1)
    valid = targets != ignore_index
    ious: list[torch.Tensor] = []
    for class_id in range(num_classes):
        pred_mask = (preds == class_id) & valid
        target_mask = (targets == class_id) & valid
        union = pred_mask | target_mask
        if union.any():
            intersection = pred_mask & target_mask
            ious.append(intersection.sum().float() / union.sum().float().clamp_min(1))
    if not ious:
        return 0.0
    return torch.stack(ious).mean().item()
