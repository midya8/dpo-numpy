# dpo-numpy

Direct preference optimization from scratch, in pure NumPy. No PyTorch, no JAX, and no reward model.

DPO (Rafailov et al., 2023) cuts the reward model out of RLHF. Instead of training a reward function and running RL against it, it rewrites the KL-constrained objective so the optimal policy has a closed form in terms of the reference model, and the whole thing reduces to a classification loss on preference pairs:

```
loss = -log sigmoid( beta * [ (log pi(y_w) - log pi_ref(y_w))
                           - (log pi(y_l) - log pi_ref(y_l)) ] )
```

where y_w is the preferred completion and y_l the rejected one.

## The toy problem

Strings over the alphabet {a, b, c}, 16 characters. One rule: the bigram "ac" is banned. Preference pairs are generated programmatically: chosen strings never contain "ac", rejected strings always do. The reference model is a plain next-token predictor trained on 4096 uniform random strings, so it never sees the rule. Its loss converges to ln(3) = 1.0986, exactly what a uniform predictor should reach, and it puts "ac" at its natural frequency: 68.5% of 16-character samples contain at least one.

DPO then has to move the policy away from that frequency using nothing but the pairs.

## Results

Measured on 2048 held-out samples after 600 DPO steps (batch 64, lr 1e-4, beta 0.1):

| model | violation rate | training pair accuracy |
|-------|----------------|------------------------|
| reference | 68.5% | chance (0.50) |
| policy after DPO | 7.5% | 0.83 |

What this does not claim: 7.5% is not zero, the policy still samples "ac" occasionally. Beta is the only knob trading preference fit against drift from the reference. Pushing it higher fits the pairs harder and pushes the margins up while the violation rate stops moving. There is no explicit KL term, that is the point of DPO, but it also means there is no dial that stops at exactly the natural frequency.

## What is inside

- `dpolab/tensor.py`: a small reverse-mode autograd engine in float64. Every op used by the model has a hand-derived backward, checked against finite differences to about 9 decimal places (see `tests/`).
- `dpolab/model.py`: a tiny character-level GPT. Pre-LN blocks, causal multi-head attention, learned position embeddings, about 26k parameters.
- `dpolab/data.py`: corpus generation and preference-pair construction.
- `dpolab/dpo.py`: the loss.
- `dpolab/optim.py`: Adam.
- `dpolab/train.py`: sft, dpo and eval subcommands.

## Running it

```
python -m dpolab.train sft --steps 500 --batch 64 --lr 3e-3
python -m dpolab.train dpo --steps 600 --batch 64 --lr 1e-4 --beta 0.1
python -m dpolab.train eval

pytest tests/
```

## References

Rafailov, Sharma, Mitchell et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023.
