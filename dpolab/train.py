"""SFT the reference, then DPO, then measure what moved.

    python -m dpolab.train sft    --steps 800
    python -m dpolab.train dpo    --steps 600 --beta 1.0
    python -m dpolab.train eval   --tag dpo
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .data import make_dataset, make_uniform_corpus, sample_string, violates
from .dpo import DPOLoss
from .model import TinyGPT
from .optim import Adam
from .tensor import Tensor

CKPT = "checkpoints"
RUNS = "runs"


def batches(n, size, rng):
    idx = rng.permutation(n)
    for i in range(0, n - size + 1, size):
        yield idx[i:i + size]


def sft(model, corpus, steps, batch_size, lr, seed):
    """Train on uniform strings: next-token prediction, nothing else."""
    rng = np.random.default_rng(seed)
    opt = Adam(model.parameters(), lr=lr)
    n = len(corpus)
    t0 = time.time()
    for step in range(steps):
        idx = rng.integers(0, n, size=batch_size)
        seq = np.stack([corpus[i] for i in idx])
        toks, tg = seq[:, :-1], seq[:, 1:]
        loss = model.nll(toks, tg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"sft step {step:4d} nll {float(loss.data):.4f}")
    print(f"sft done in {time.time() - t0:.0f}s")


def ref_log_probs(model, data, idx):
    """Frozen log-prob sums of the policy-as-reference for chosen/rejected."""
    ch, rj = data.batch(idx)
    with_ch = model.log_probs(ch)  # (B, T)
    ch_sum = Tensor(with_ch.data.sum(axis=1))
    rj_sum = Tensor(model.log_probs(rj).data.sum(axis=1))
    return ch_sum, rj_sum


def dpo(model, data, ref_model, steps, batch_size, lr, beta, seed, log_every=50):
    rng = np.random.default_rng(seed)
    opt = Adam(model.parameters(), lr=lr)
    n = len(data.rows)
    loss_fn = DPOLoss(model, None, beta=beta)
    t0 = time.time()
    for step in range(steps):
        idx = rng.integers(0, n, size=batch_size)
        ch, rj = data.batch(idx)
        # reference pass, frozen: no gradients flow into ref_model, and the
        # policy shares its weights at init so we snapshot its outputs
        ref_lp_c, ref_lp_r = ref_outputs(ref_model, data, idx)
        loss, margins = loss_fn(ch, rj, ref_lp_c, ref_lp_r)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            acc, mean_m, _ = loss_fn.stats(ch, rj, ref_lp_c, ref_lp_r)
            print(f"dpo step {step:4d} loss {float(loss.data):.4f} "
                  f"acc {acc:.3f} margin {mean_m:.3f}")
    print(f"dpo done in {time.time() - t0:.0f}s")


def ref_outputs(ref_model, data, idx):
    ch, rj = data.batch(idx)
    lp_c = ref_model.log_probs(ch)
    lp_r = ref_model.log_probs(rj)
    return Tensor(lp_c.data.sum(axis=1)), Tensor(lp_r.data.sum(axis=1))


def sample_free(model, data, prefix, n_new, rng, temperature=1.0):
    ids = data.encode(prefix).tolist()
    for _ in range(n_new):
        ctx = np.array(ids[-model.block_size:], dtype=np.int64)[None, :]
        lg = model.logits(ctx).data[0, -1] / temperature
        p = np.exp(lg - lg.max()); p /= p.sum()
        ids.append(int(rng.choice(len(p), p=p)))
    return data.decode(ids)


def eval_model(model, data, n=200, seed=0, n_new=8):
    rng = np.random.default_rng(seed)
    viol = 0
    total = 0
    for i in range(n):
        p, _, _ = data.rows[i]
        out = sample_free(model, data, p, n_new, rng)
        if violates(out):
            viol += 1
        total += 1
    return viol / total


def save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{f"p{i}": p.data for i, p in enumerate(model.parameters())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sft", "dpo", "eval"])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    train_data, eval_data = make_dataset(seed=args.seed)
    V = len(train_data.stoi)
    seq_len = len(train_data.rows[0][0]) + len(train_data.rows[0][1])
    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(RUNS, exist_ok=True)

    if args.cmd == "sft":
        corpus = make_uniform_corpus(4096, seq_len, seed=args.seed)
        model = TinyGPT(V, args.d_model, args.layers, args.heads, seq_len,
                        seed=args.seed)
        sft(model, corpus, args.steps, args.batch, args.lr, args.seed)
        save(model, f"{CKPT}/reference.npz")
        v = eval_model(model, eval_data)
        print(f"reference violation rate: {v:.3f}")
        json.dump({"sft_violation_rate": v}, open(f"{RUNS}/sft.json", "w"))

    elif args.cmd == "dpo":
        ref = TinyGPT(V, args.d_model, args.layers, args.heads, seq_len,
                      seed=args.seed)
        z = np.load(f"{CKPT}/reference.npz")
        for p, (_, arr) in zip(ref.parameters(), z.items()):
            p.data[...] = arr
        model = TinyGPT(V, args.d_model, args.layers, args.heads, seq_len,
                        seed=args.seed)
        for p, (_, arr) in zip(model.parameters(), z.items()):
            p.data[...] = arr
        dpo(model, train_data, ref, args.steps, args.batch, args.lr, args.beta,
            args.seed)
        save(model, f"{CKPT}/dpo.npz")
        v = eval_model(model, eval_data)
        print(f"dpo violation rate: {v:.3f}")

    elif args.cmd == "eval":
        for name in ["reference", "dpo"]:
            path = f"{CKPT}/{name}.npz"
            if not os.path.exists(path):
                continue
            m = TinyGPT(V, args.d_model, args.layers, args.heads, seq_len,
                        seed=args.seed)
            z = np.load(path)
            for p, (_, arr) in zip(m.parameters(), z.items()):
                p.data[...] = arr
            v = eval_model(m, eval_data)
            print(f"{name}: violation rate {v:.3f}")


if __name__ == "__main__":
    main()
