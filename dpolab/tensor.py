"""A small reverse-mode autograd core, written from scratch.

Only what a character-level transformer and a DPO loss need. Values are
float64 throughout so finite-difference checks and exact-fraction reasoning
stay meaningful at this scale.
"""

from __future__ import annotations

import numpy as np


class Tensor:
    __slots__ = ("data", "grad", "parents", "back", "requires_grad")

    def __init__(self, data, parents=(), back=None, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.parents = parents
        self.back = back
        self.requires_grad = requires_grad

    @property
    def shape(self):
        return self.data.shape

    def _accumulate(self, g):
        g = np.asarray(g, dtype=np.float64)
        if g.shape != self.data.shape:
            g = _reduce_to(g, self.data.shape)
        if self.grad is None:
            self.grad = g
        else:
            self.grad = self.grad + g

    def zero_grad(self):
        self.grad = None

    def backward(self, seed=None):
        if seed is None:
            seed = np.ones_like(self.data)
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, visited = stack.pop()
            if visited:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node.parents:
                if id(p) not in seen:
                    stack.append((p, False))
        self._accumulate(seed)
        for node in reversed(topo):
            if node.back is not None and node.grad is not None:
                node.back(node.grad)

    # elementwise -----------------------------------------------------------
    def __add__(self, other):
        other = _wrap(other, self)
        def back(g):
            self._accumulate(g)
            other._accumulate(g)
        return Tensor(self.data + other.data, (self, other), back,
                      self.requires_grad or other.requires_grad)

    __radd__ = __add__

    def __mul__(self, other):
        other = _wrap(other, self)
        def back(g):
            self._accumulate(g * other.data)
            other._accumulate(g * self.data)
        return Tensor(self.data * other.data, (self, other), back,
                      self.requires_grad or other.requires_grad)

    __rmul__ = __mul__

    def __neg__(self):
        def back(g):
            self._accumulate(-g)
        return Tensor(-self.data, (self,), back, self.requires_grad)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        other = _wrap(other, self)
        def back(g):
            self._accumulate(g / other.data)
            other._accumulate(-g * self.data / (other.data * other.data))
        return Tensor(self.data / other.data, (self, other), back,
                      self.requires_grad or other.requires_grad)

    # shapes ----------------------------------------------------------------
    def reshape(self, *shape):
        def back(g):
            self._accumulate(g.reshape(self.data.shape))
        return Tensor(self.data.reshape(*shape), (self,), back, self.requires_grad)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(reversed(range(self.data.ndim)))
        def back(g):
            self._accumulate(np.transpose(g, np.argsort(axes)))
        return Tensor(np.transpose(self.data, axes), (self,), back, self.requires_grad)

    # reductions ------------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        def back(g):
            if axis is None:
                self._accumulate(np.broadcast_to(g, self.data.shape).copy())
                return
            if not keepdims:
                g = np.expand_dims(g, axis)
            self._accumulate(np.broadcast_to(g, self.data.shape).copy())
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), back,
                      self.requires_grad)

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # unary -----------------------------------------------------------------
    def log(self):
        def back(g):
            self._accumulate(g / self.data)
        return Tensor(np.log(self.data), (self,), back, self.requires_grad)

    def exp(self):
        out_data = np.exp(self.data)
        def back(g):
            self._accumulate(g * out_data)
        return Tensor(out_data, (self,), back, self.requires_grad)

    def sigmoid(self):
        out_data = 1.0 / (1.0 + np.exp(-self.data))
        def back(g):
            self._accumulate(g * out_data * (1.0 - out_data))
        return Tensor(out_data, (self,), back, self.requires_grad)

    def sqrt(self):
        out_data = np.sqrt(self.data)
        def back(g):
            self._accumulate(0.5 * g / out_data)
        return Tensor(out_data, (self,), back, self.requires_grad)

    def __matmul__(self, other):
        return matmul(self, other)

    # indexing --------------------------------------------------------------
    def take(self, index, axis=-1):
        """Gather along an axis. `index` either matches the rank of the data
        (with k along `axis`) or is one rank lower."""
        index = np.asarray(index)
        if index.ndim == self.data.ndim - 1:
            index = np.expand_dims(index, axis=axis)
        def back(g):
            ax = axis % self.data.ndim
            shape = self.data.shape
            coords = []
            for d, sd in enumerate(shape):
                if d == ax:
                    coords.append(index)
                else:
                    ar = np.arange(sd).reshape(
                        [sd if dd == d else 1 for dd in range(self.data.ndim)])
                    coords.append(np.broadcast_to(ar, g.shape))
            flat = np.ravel_multi_index(coords, shape)
            full = np.zeros(self.data.size, dtype=np.float64)
            np.add.at(full, flat, g)
            self._accumulate(full.reshape(shape))
        out = np.take_along_axis(self.data, index, axis=axis)
        return Tensor(out, (self,), back, self.requires_grad)

    def take_last(self, index):
        """Gather along axis 1, then keep only the last selected position."""
        return self.take(index, axis=1)[:, -1]

    def __getitem__(self, item):
        def back(g):
            full = np.zeros_like(self.data)
            np.add.at(full, item, g)
            self._accumulate(full)
        return Tensor(self.data[item], (self,), back, self.requires_grad)

    def concat(self, other, axis=0):
        other = _wrap(other, self)
        def back(g):
            heads = [a.shape[axis] for a in (self, other)]
            self._accumulate(np.split(g, np.cumsum(heads)[:-1], axis=axis)[0])
            other._accumulate(np.split(g, np.cumsum(heads)[:-1], axis=axis)[1])
        return Tensor(np.concatenate([self.data, other.data], axis=axis),
                      (self, other), back, self.requires_grad or other.requires_grad)


def _reduce_to(g, shape):
    """Sum a gradient down to the shape of the tensor it feeds, undoing
    every broadcast the forward pass applied."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def _wrap(value, like):
    if isinstance(value, Tensor):
        return value
    return Tensor(np.broadcast_to(np.asarray(value, dtype=np.float64),
                                 like.data.shape), (), None, False)


def matmul(a, b):
    """Batched matmul with clean gradients for 2D, 3D and 4D operands."""
    a, b = _wrap(a, a), _wrap(b, b) if isinstance(b, Tensor) else (a, b)
    a_t = a if isinstance(a, Tensor) else Tensor(a)
    b_t = b if isinstance(b, Tensor) else Tensor(b)
    def back(g):
        if a_t.data.ndim >= 2:
            a_t._accumulate(np.matmul(g, np.swapaxes(b_t.data, -1, -2)))
        if b_t.data.ndim >= 2:
            b_t._accumulate(np.matmul(np.swapaxes(a_t.data, -1, -2), g))
    return Tensor(np.matmul(a_t.data, b_t.data), (a_t, b_t), back,
                  a_t.requires_grad or b_t.requires_grad)


def logsumexp(x, axis=-1, keepdims=False):
    m = x.data.max(axis=axis, keepdims=True)
    out_data = np.log(np.exp(x.data - m).sum(axis=axis, keepdims=True)) + m
    if not keepdims:
        out_data = np.squeeze(out_data, axis=axis)
    def back(g):
        soft = np.exp(x.data - m)
        soft = soft / soft.sum(axis=axis, keepdims=True)
        if not keepdims:
            soft = soft * np.expand_dims(g, axis)
        else:
            soft = soft * g
        x._accumulate(soft)
    return Tensor(out_data, (x,), back, x.requires_grad)


def gelu(x):
    """Tanh-approximation GELU, kept as two autograd ops so gradients compose."""
    inner = x * 0.7978845608028654 * (1.0 + x * x * 0.044715)
    return x * 0.5 * (1.0 + (inner).sigmoid())


def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    centered = x - mu
    var = (centered * centered).mean(axis=-1, keepdims=True)
    normed = centered / (var + eps).sqrt()
    return normed * gamma + beta
