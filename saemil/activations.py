"""Activations and stable sigmoid/softmax helpers."""
import numpy as np

from .layers import Layer


def sigmoid(x):
    """Stable sigmoid on sign subsets, so neither branch evaluates exp of a large positive."""
    x = np.asarray(x)
    out = np.empty_like(x, dtype=np.result_type(x, np.float32))
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def softmax(z):
    """Row-wise softmax over the last axis with max subtraction."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class ReLU(Layer):
    def forward(self, x, training):
        if training:
            self.cache = x > 0  # derivative 0 at exactly 0
        return np.maximum(x, 0)

    def backward(self, dy):
        return dy * self.cache


class Sigmoid(Layer):
    def forward(self, x, training):
        a = sigmoid(x)
        if training:
            self.cache = a
        return a

    def backward(self, dy):
        a = self.cache
        return dy * a * (1 - a)


class Tanh(Layer):
    def forward(self, x, training):
        a = np.tanh(x)
        if training:
            self.cache = a
        return a

    def backward(self, dy):
        a = self.cache
        return dy * (1 - a * a)
