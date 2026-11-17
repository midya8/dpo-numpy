"""Adam, plainly written."""

from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        b1c = 1 - self.b1 ** self.t
        b2c = 1 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m += (1 - self.b1) * (g - m)
            v += (1 - self.b2) * (g * g - v)
            p.data -= self.lr * (m / b1c) / (np.sqrt(v / b2c) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
