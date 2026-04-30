from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ForwardCache:
    x: np.ndarray
    z1: np.ndarray
    a1: np.ndarray
    z2: np.ndarray
    a2: np.ndarray
    logits: np.ndarray
    probs: np.ndarray


class MLPClassifier:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (256, 128),
        output_dim: int = 10,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        if activation not in {"relu", "sigmoid", "tanh"}:
            raise ValueError("activation must be one of: relu, sigmoid, tanh")
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        rng = np.random.default_rng(seed)
        dims = [input_dim, hidden_dims[0], hidden_dims[1], output_dim]
        self.params: dict[str, np.ndarray] = {}
        for layer in range(1, 4):
            fan_in, fan_out = dims[layer - 1], dims[layer]
            scale = np.sqrt(2.0 / fan_in) if activation == "relu" and layer < 3 else np.sqrt(1.0 / fan_in)
            self.params[f"W{layer}"] = (rng.normal(0.0, scale, size=(fan_in, fan_out))).astype(np.float32)
            self.params[f"b{layer}"] = np.zeros((1, fan_out), dtype=np.float32)

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(z, 0.0)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        return np.tanh(z)

    def _activation_grad(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (z > 0).astype(np.float32)
        if self.activation == "sigmoid":
            return a * (1.0 - a)
        return 1.0 - a**2

    def forward(self, x: np.ndarray) -> ForwardCache:
        z1 = x @ self.params["W1"] + self.params["b1"]
        a1 = self._activate(z1)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = self._activate(z2)
        logits = a2 @ self.params["W3"] + self.params["b3"]
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return ForwardCache(x, z1, a1, z2, a2, logits, probs)

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray, weight_decay: float = 0.0) -> tuple[float, dict[str, np.ndarray]]:
        cache = self.forward(x)
        n = x.shape[0]
        data_loss = -np.log(cache.probs[np.arange(n), y] + 1e-12).mean()
        l2_loss = 0.5 * weight_decay * sum(np.sum(self.params[f"W{i}"] ** 2) for i in range(1, 4))
        loss = float(data_loss + l2_loss)

        dlogits = cache.probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n

        grads: dict[str, np.ndarray] = {}
        grads["W3"] = cache.a2.T @ dlogits + weight_decay * self.params["W3"]
        grads["b3"] = dlogits.sum(axis=0, keepdims=True)

        da2 = dlogits @ self.params["W3"].T
        dz2 = da2 * self._activation_grad(cache.z2, cache.a2)
        grads["W2"] = cache.a1.T @ dz2 + weight_decay * self.params["W2"]
        grads["b2"] = dz2.sum(axis=0, keepdims=True)

        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * self._activation_grad(cache.z1, cache.a1)
        grads["W1"] = cache.x.T @ dz1 + weight_decay * self.params["W1"]
        grads["b1"] = dz1.sum(axis=0, keepdims=True)
        return loss, grads

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        preds: list[np.ndarray] = []
        for start in range(0, x.shape[0], batch_size):
            probs = self.forward(x[start : start + batch_size]).probs
            preds.append(np.argmax(probs, axis=1))
        return np.concatenate(preds)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        payload = dict(self.params)
        payload.update(
            input_dim=np.array(self.input_dim),
            hidden_dims=np.asarray(self.hidden_dims),
            output_dim=np.array(self.output_dim),
            activation=np.array(self.activation),
            metadata=np.array(metadata or {}, dtype=object),
        )
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "MLPClassifier":
        data = np.load(path, allow_pickle=True)
        model = cls(
            input_dim=int(data["input_dim"]),
            hidden_dims=tuple(int(v) for v in data["hidden_dims"]),
            output_dim=int(data["output_dim"]),
            activation=str(data["activation"]),
        )
        for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            model.params[key] = data[key]
        return model
