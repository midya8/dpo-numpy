"""The DPO objective, written out in full.

For a pair (y_w, y_l) given prompt x, with policy pi and reference pi_ref,
both conditioned on the same tokens:

    margin = beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
                    - (log pi(y_l|x) - log pi_ref(y_l|x)) ]
    loss   = -log sigmoid(margin)

The reference log-probs are fixed at the start of training, so per pair
only the policy needs a forward pass. The quantity being pushed is the
log-ratio difference between chosen and rejected; nothing else enters the
loss.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor, logsumexp


class DPOLoss:
    def __init__(self, model, ref_log_probs_fn, beta=1.0):
        self.model = model
        self.ref_fn = ref_log_probs_fn
        self.beta = beta

    def margins(self, chosen, rejected, ref_chosen, ref_rejected):
        """Sequence-level log-prob sums minus the reference's, times beta."""
        pol_ch = self.model.log_probs(chosen).sum(axis=1)
        pol_rj = self.model.log_probs(rejected).sum(axis=1)
        return self.beta * ((pol_ch - ref_chosen) - (pol_rj - ref_rejected))

    def __call__(self, chosen, rejected, ref_chosen, ref_rejected):
        m = self.margins(chosen, rejected, ref_chosen, ref_rejected)
        # -log sigmoid(m)
        loss = -(m.sigmoid()).log().mean()
        return loss, m

    def stats(self, chosen, rejected, ref_chosen, ref_rejected):
        """No-gradient view for logging: (accuracy, mean margin, mean KL)."""
        pol_ch = self.model.log_probs(chosen).sum(axis=1)
        pol_rj = self.model.log_probs(rejected).sum(axis=1)
        d_ch = (pol_ch - ref_chosen).data
        d_rj = (pol_rj - ref_rejected).data
        margins = self.beta * (d_ch - d_rj)
        acc = float((margins > 0).mean())
        kl = float(np.mean((d_ch + d_rj)) / -2.0)  # E[KL] = -E[log ratio] per token
        return acc, float(margins.mean()), kl


def seq_log_probs_no_grad(model, tokens):
    """Reference-model log-prob sums, computed once and frozen."""
    lp = model.log_probs(tokens)
    lse = lp.data  # already log softmax per position
    return Tensor(lse.sum(axis=1))


def kl_per_token(model, tokens, ref_log_probs):
    """Mean KL(policy || ref) over a batch, in nats per token."""
    lp = model.log_probs(tokens).data
    diff = lp - ref_log_probs
    return float(diff.mean())
