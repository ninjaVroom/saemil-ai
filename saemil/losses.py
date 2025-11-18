"""Mean losses. Each call returns (loss: float, grad w.r.t. its first argument)."""
import numpy as np

from .activations import sigmoid


class SoftmaxCrossEntropy:
    """Fused softmax + CE on logits Z (B, C) with integer targets y (B,)."""

    def __call__(self, logits, y_int):
        y = np.asarray(y_int)
        if logits.ndim != 2 or y.shape != (logits.shape[0],) or y.dtype.kind not in "iu":
            raise ValueError("expected logits (B, C) and integer targets (B,)")
        if y.size and (y.min() < 0 or y.max() >= logits.shape[1]):
            raise ValueError(f"targets must be in [0, {logits.shape[1]})")
        B = logits.shape[0]
        rows = np.arange(B)
        s = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(s)
        a = e.sum(axis=1, keepdims=True)
        loss = np.mean(np.log(a[:, 0]) - s[rows, y])
        dz = e / a
        dz[rows, y] -= 1
        dz /= B
        return float(loss), dz


class BinaryCrossEntropy:
    """BCE on logits z with 0/1 targets of the same shape."""

    def __call__(self, logits, y01):
        y = np.asarray(y01, dtype=logits.dtype)
        if y.shape != logits.shape:
            raise ValueError(f"target shape {y.shape} != logits shape {logits.shape}")
        loss = np.mean(np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits))))
        return float(loss), (sigmoid(logits) - y) / logits.size


class MSE:
    def __call__(self, pred, target):
        t = np.asarray(target, dtype=pred.dtype)
        if t.shape != pred.shape:
            raise ValueError(f"target shape {t.shape} != pred shape {pred.shape}")
        diff = pred - t
        return float(np.mean(diff * diff)), 2 * diff / pred.size
