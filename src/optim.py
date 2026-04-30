from __future__ import annotations

import numpy as np


class SGD:
    def __init__(self, params: dict[str, np.ndarray], lr: float) -> None:
        self.params = params
        self.lr = lr

    def step(self, grads: dict[str, np.ndarray]) -> None:
        for key, grad in grads.items():
            self.params[key] -= self.lr * grad.astype(self.params[key].dtype, copy=False)


def decayed_lr(initial_lr: float, decay: float, epoch: int) -> float:
    return initial_lr * (decay ** max(epoch - 1, 0))
