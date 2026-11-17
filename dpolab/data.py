"""The toy task: learn a rule no supervised example ever states.

Sequences are strings over {a, b, c}. The rule is a single banned bigram:
'ac' may not appear anywhere in a sequence. The reference model is trained
on iid-uniform strings, where 'ac' shows up with probability 1/9 per bigram,
so the rule is something the model has never seen expressed.

Preference pairs are then built from the rule itself: given a uniform
prefix, the chosen completion contains no 'ac', the rejected one contains
at least one. If DPO works, the policy drifts away from the reference
toward completions that obey a constraint it was never told about.
"""

from __future__ import annotations

import numpy as np

ALPHABET = "abc"
BANNED = "ac"


def violates(seq: str) -> bool:
    return BANNED in seq


def sample_string(rng, length, forbid=False, require=False, prefix=""):
    """Draw a uniform string of `length` over the alphabet.

    With `forbid`, reject draws where the full sequence (prefix included)
    would contain 'ac'. With `require`, keep only draws that contain it.
    """
    while True:
        body = "".join(rng.choice(list(ALPHABET), size=length))
        full = prefix + body
        v = violates(full)
        if forbid and v:
            continue
        if require and not v:
            continue
        return body


def make_split(rng, n, prefix_len, comp_len):
    # prompts obey the rule too: a violating prompt would make the chosen
    # side impossible and the rejected side trivially already bad
    prefixes = [sample_string(rng, prefix_len, forbid=True) for _ in range(n)]
    rows = []
    for p in prefixes:
        chosen = sample_string(rng, comp_len, forbid=True, prefix=p)
        rejected = sample_string(rng, comp_len, require=True, prefix=p)
        rows.append((p, chosen, rejected))
    return rows


class PreferenceData:
    def __init__(self, rows, stoi):
        self.rows = rows
        self.stoi = stoi
        self.itos = {i: c for c, i in stoi.items()}

    def encode(self, text):
        return np.array([self.stoi[c] for c in text], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def batch(self, idx):
        """(chosen, rejected) token batches: (B, prefix+comp) each."""
        ch, rj = [], []
        for i in idx:
            p, c, r = self.rows[i]
            ch.append(self.encode(p + c))
            rj.append(self.encode(p + r))
        return np.stack(ch), np.stack(rj)


def make_uniform_corpus(n, length, seed=0):
    """Plain uniform strings. The reference model trains on these, so the
    banned bigram sits at its natural 1/9 frequency and the rule never
    enters supervised data."""
    rng = np.random.default_rng(seed)
    stoi = {c: i for i, c in enumerate(ALPHABET)}
    texts = [sample_string(rng, length) for _ in range(n)]
    return [np.array([stoi[c] for c in t], dtype=np.int64) for t in texts]


def make_dataset(n_train=4096, n_eval=512, prefix_len=8, comp_len=8, seed=0):
    rng = np.random.default_rng(seed)
    stoi = {c: i for i, c in enumerate(ALPHABET)}
    train = make_split(rng, n_train, prefix_len, comp_len)
    eval_rows = make_split(rng, n_eval, prefix_len, comp_len)
    return PreferenceData(train, stoi), PreferenceData(eval_rows, stoi)
