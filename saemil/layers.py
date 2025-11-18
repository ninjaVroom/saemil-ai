"""Layers with explicit forward/backward. Caches are written only when training=True and alias the input."""
from typing import Any

import numpy as np


class Layer:
    cache: Any = None  # whatever backward() needs; set by forward(training=True)

    def __init__(self):
        self.config = {}

    def forward(self, x, training: bool):
        raise NotImplementedError

    def backward(self, dy):
        raise NotImplementedError

    def params(self) -> dict:
        return {}

    def grads(self) -> dict:
        return {}


class Linear(Layer):
    """Y = X @ W + b. W ~ N(0, sqrt(2/D)) for "he", N(0, sqrt(1/D)) for "lecun"; b = 0."""

    def __init__(self, d_in, d_out, init, rng, dtype=np.float32):
        if init not in ("he", "lecun"):
            raise ValueError(f"init must be 'he' or 'lecun', got {init!r}")
        std = np.sqrt((2.0 if init == "he" else 1.0) / d_in)
        self.config = {"d_in": int(d_in), "d_out": int(d_out), "init": init}
        self.W = (rng.standard_normal((d_in, d_out)) * std).astype(dtype)
        self.b = np.zeros(d_out, dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x, training):
        x = np.asarray(x, dtype=self.W.dtype)  # no silent upcast in float32 mode
        if training:
            self.cache = x
        return x @ self.W + self.b

    def backward(self, dy):
        self.dW[...] = self.cache.T @ dy
        self.db[...] = dy.sum(axis=0)
        return dy @ self.W.T

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}


class Embedding(Layer):
    """Y[b,t,:] = E[I[b,t],:]. E ~ N(0, 0.1). No gradient w.r.t. the integer IDs."""

    def __init__(self, vocab, dim, rng, dtype=np.float32):
        self.config = {"vocab": int(vocab), "dim": int(dim)}
        self.E = (rng.standard_normal((vocab, dim)) * 0.1).astype(dtype)
        self.dE = np.zeros_like(self.E)

    def forward(self, x, training):
        ids = np.asarray(x)
        if ids.dtype.kind not in "iu":
            raise TypeError(f"Embedding expects integer IDs, got {ids.dtype}")
        if ids.size and (ids.min() < 0 or ids.max() >= self.E.shape[0]):
            raise ValueError(f"Embedding IDs must be in [0, {self.E.shape[0]})")
        if training:
            self.cache = ids
        return self.E[ids]

    def backward(self, dy):
        self.dE.fill(0)
        np.add.at(self.dE, self.cache.ravel(), dy.reshape(-1, self.E.shape[1]))  # unbuffered: duplicates add up
        return None

    def params(self):
        return {"E": self.E}

    def grads(self):
        return {"E": self.dE}


class Flatten(Layer):
    """(B, T, K) -> (B, T*K)."""

    def forward(self, x, training):
        if training:
            self.cache = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dy):
        return dy.reshape(self.cache)
