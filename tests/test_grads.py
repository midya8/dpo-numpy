"""Finite-difference checks for the autograd core and the model.

Each test compares analytic gradients from tensor.py against central
differences on a handful of random coordinates. float64 throughout.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpolab.tensor import Tensor, gelu, layer_norm, logsumexp, matmul
from dpolab.dpo import DPOLoss
from dpolab.model import TinyGPT


def numeric(f, x, coords, eps=1e-6):
    """Central finite differences of f() with respect to entries of x."""
    g = {}
    for i in coords:
        old = x[i]
        x[i] = old + eps
        fp = f()
        x[i] = old - eps
        fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def sample_coords(shape, n=6, seed=0):
    rng = np.random.default_rng(seed)
    return [tuple(rng.integers(0, d) for d in shape) for _ in range(n)]


def check(fn, x, analytic, tol, n=6, seed=0):
    coords = sample_coords(x.shape, n, seed)
    num = numeric(fn, x, coords)
    worst = max(abs(num[i] - analytic[i]) for i in coords)
    assert worst < tol, f"max diff {worst:.3e} exceeds {tol:.1e}"


def test_matmul():
    rng = np.random.default_rng(0)
    a = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    b = Tensor(rng.normal(size=(4, 5)), requires_grad=True)
    y = matmul(a, b)
    y.sum().backward()
    check(lambda: float(matmul(Tensor(a.data, requires_grad=False),
                               Tensor(b.data, requires_grad=False)).sum().data),
          a.data, a.grad, 1e-7)


def test_gelu_matmul():
    rng = np.random.default_rng(1)
    x = Tensor(rng.normal(size=(4, 6)), requires_grad=True)
    w = Tensor(rng.normal(size=(6, 3)), requires_grad=True)
    gelu(matmul(x, w)).sum().backward()

    def f():
        return float(gelu(matmul(Tensor(x.data), Tensor(w.data))).sum().data)

    check(f, x.data, x.grad, 1e-7)
    check(f, w.data, w.grad, 1e-7)


def test_layer_norm():
    rng = np.random.default_rng(2)
    x = Tensor(rng.normal(size=(3, 5)), requires_grad=True)
    gamma = Tensor(np.ones(5), requires_grad=True)
    beta = Tensor(np.zeros(5), requires_grad=True)
    layer_norm(x, gamma, beta).sum().backward()

    def f():
        return float(layer_norm(Tensor(x.data), Tensor(gamma.data),
                                Tensor(beta.data)).sum().data)

    check(f, x.data, x.grad, 1e-7)
    check(f, gamma.data, gamma.grad, 1e-7)
    check(f, beta.data, beta.grad, 1e-7)


def test_logsumexp():
    rng = np.random.default_rng(3)
    x = Tensor(rng.normal(size=(2, 7)), requires_grad=True)
    logsumexp(x, axis=-1).sum().backward()

    def f():
        return float(logsumexp(Tensor(x.data), axis=-1).sum().data)

    check(f, x.data, x.grad, 1e-7)


def test_take_repeated_indices():
    rng = np.random.default_rng(4)
    x = Tensor(rng.normal(size=(2, 3, 5)), requires_grad=True)
    idx = np.array([[0, 1, 1], [2, 2, 0]])  # repeats on purpose
    x.take(idx).sum().backward()

    def f():
        return float(Tensor(x.data).take(idx).sum().data)

    check(f, x.data, x.grad, 1e-7)


def test_model_nll():
    model = TinyGPT(3, d_model=16, n_layers=1, n_heads=2, block_size=8, seed=5)
    rng = np.random.default_rng(6)
    toks = rng.integers(0, 3, size=(4, 8))
    tg = rng.integers(0, 3, size=(4, 8))
    loss = model.nll(toks, tg)
    loss.backward()

    params = model.parameters()
    emb = min(params, key=lambda p: p.data.size)

    def f():
        return float(model.nll(toks, tg).data)

    check(f, emb.data, emb.grad, 2e-6)


def test_dpo_loss_grad():
    policy = TinyGPT(3, d_model=16, n_layers=1, n_heads=2, block_size=8, seed=7)
    ref = TinyGPT(3, d_model=16, n_layers=1, n_heads=2, block_size=8, seed=8)
    rng = np.random.default_rng(9)
    chosen = rng.integers(0, 3, size=(3, 8))
    rejected = rng.integers(0, 3, size=(3, 8))
    beta = 0.1

    ref_lp_c = ref.log_probs(chosen).data.sum(axis=1)
    ref_lp_r = ref.log_probs(rejected).data.sum(axis=1)
    crit = DPOLoss(policy, None, beta=beta)
    loss, margins = crit(chosen, rejected, ref_lp_c, ref_lp_r)

    # loss value against the formula, computed straight from log probs
    pc = policy.log_probs(chosen).data
    pl = policy.log_probs(rejected).data
    # mean per-token log prob difference, matching dpo.margins' reduction
    gap = ((pc.sum(axis=1) - ref_lp_c)
           - (pl.sum(axis=1) - ref_lp_r))
    manual = -np.log(1.0 / (1.0 + np.exp(-beta * gap))).mean()
    assert abs(float(loss.data) - manual) < 1e-8

    loss.backward()
    params = policy.parameters()
    emb = min(params, key=lambda p: p.data.size)

    def f():
        return float(DPOLoss(policy, None, beta=beta)(
            chosen, rejected, ref_lp_c, ref_lp_r)[0].data)

    check(f, emb.data, emb.grad, 2e-6)
