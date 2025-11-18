"""In-place optimizers over named param/grad dicts."""
import numpy as np


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        for name, p in params.items():
            p -= self.lr * grads[name]


class Adam:
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0  # one shared step counter for all tensors
        self.m, self.v = {}, {}

    def step(self, params, grads):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for name, p in params.items():
            g = grads[name]
            if name not in self.m:
                self.m[name] = np.zeros_like(p)
                self.v[name] = np.zeros_like(p)
            m, v = self.m[name], self.v[name]
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
