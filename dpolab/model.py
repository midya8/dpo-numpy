"""A tiny character-level transformer, hand-written on the autograd core.

Nothing here is clever: pre-LN blocks, one causal attention head per layer,
learned positional embeddings, untied output head. Small on purpose. The
whole point of the repo is what the DPO update does to this model, so the
architecture stays out of the way.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor, gelu, layer_norm, logsumexp, matmul


class TinyGPT:
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=1, block_size=32,
                 seed=0):
        if d_model % n_heads:
            raise ValueError("d_model must divide evenly across heads")
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size = block_size

        scale = 0.02
        def param(*shape):
            t = Tensor(rng.normal(0.0, scale, size=shape))
            t.requires_grad = True
            return t

        self.tok_emb = param(vocab_size, d_model)
        self.pos_emb = param(block_size, d_model)
        self.blocks = []
        for _ in range(n_layers):
            self.blocks.append({
                "ln1_g": param(d_model), "ln1_b": param(d_model),
                "wq": param(d_model, d_model), "wk": param(d_model, d_model),
                "wv": param(d_model, d_model), "wo": param(d_model, d_model),
                "ln2_g": param(d_model), "ln2_b": param(d_model),
                "w1": param(d_model, 4 * d_model), "w2": param(4 * d_model, d_model),
            })
        self.lnf_g = param(d_model)
        self.lnf_b = param(d_model)
        self.head = param(d_model, vocab_size)
        self.head_b = param(vocab_size)

    def parameters(self):
        ps = [self.tok_emb, self.pos_emb, self.lnf_g, self.lnf_b, self.head, self.head_b]
        for blk in self.blocks:
            ps.extend(blk.values())
        return ps

    def _attention(self, h, blk):
        """Causal multi-head attention. h: (B, T, D) as a Tensor."""
        ln = layer_norm(h, blk["ln1_g"], blk["ln1_b"])
        B, T, D = ln.data.shape
        q = matmul(ln, blk["wq"])
        k = matmul(ln, blk["wk"])
        v = matmul(ln, blk["wv"])
        hd = self.head_dim
        nh = self.n_heads
        # (B, T, D) -> (B, nh, T, hd) -> scores (B, nh, T, T)
        qs = q.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        ks = k.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        vs = v.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        scores = matmul(qs, ks.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(hd))
        causal = np.triu(np.ones((T, T)), 1) * -1e9
        probs = (scores + causal).exp() / logsumexp(scores + causal, axis=-1, keepdims=True)
        ctx = matmul(probs, vs)                       # (B, nh, T, hd)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, D)
        return h + matmul(ctx, blk["wo"])

    def _mlp(self, h, blk):
        ln = layer_norm(h, blk["ln2_g"], blk["ln2_b"])
        return h + gelu(matmul(ln, blk["w1"])) @ blk["w2"]

    def logits(self, tokens):
        """tokens: (B, T) int array. Returns logits (B, T, V)."""
        B, T = tokens.shape
        if T > self.block_size:
            raise ValueError(f"sequence of {T} exceeds block size {self.block_size}")
        h = self.tok_emb[tokens] + self.pos_emb[:T]
        for blk in self.blocks:
            h = self._attention(h, blk)
            h = self._mlp(h, blk)
        h = layer_norm(h, self.lnf_g, self.lnf_b)
        return h @ self.head + self.head_b

    def log_probs(self, tokens):
        """Log softmax of logits at every position: (B, T)."""
        lg = self.logits(tokens)
        lse = logsumexp(lg, axis=-1, keepdims=True)
        chosen = lg.take(tokens[:, :, None], axis=-1)
        return (chosen - lse)[:, :, 0]

    def nll(self, tokens, targets):
        """Mean next-token NLL over a (B, T) target batch."""
        lg = self.logits(tokens)
        lse = logsumexp(lg, axis=-1, keepdims=True)
        chosen = lg.take(targets[:, :, None], axis=-1)
        return -((chosen - lse).mean())
