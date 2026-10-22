# σN-Ens: implicit ensembles

Code for *Controllable Diversity in Normalization-Based Implicit Ensembles via
Softmax-Temperature Modulation* (TMLR).

σN-Ens turns one network into an ensemble. The members share every weight except their norms
and their heads. A temperature `τ` sets how different the members are.

## Usage

```python
import torch
import torch.nn.functional as F
from torch import nn
from tssnorm import forward_tasks, get_norms, make_heads, optim_factory, softmax_loss, wrap

model.fc = make_heads(model.fc.in_features, num_classes, tasks=4)   # one head per member
wrap(model, init="normal")                                          # one norm per member

optimizer = optim_factory(model, torch.optim.SGD, lr=0.05, momentum=0.9)
norms = get_norms(model, "layer*.bn2")                              # the norms to regularise

for x, y in loader:
    logits = forward_tasks(model, x)                                # {member: logits}
    loss = sum(F.cross_entropy(l, y) for l in logits.values()) / len(logits)
    loss = loss + 1e-2 * softmax_loss(norms, tau=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    probs = forward_tasks(model, x, stack=True).softmax(-1).mean(0)    # the ensemble's prediction
```

To start from a pretrained model, pass `convert` to `wrap`. It keeps the pretrained scales.
`scale+bias` keeps every positive scale and the shift. `clip+bias` keeps the scales below 1. The
paper uses `clip+bias` for its CNNs and `scale+bias` for its ViTs. A negative scale cannot be
kept, so check a checkpoint's scales before converting it.

```python
# The heads get a plain BatchNorm, which wrap converts with the rest of the model.
model.fc = make_heads(model.fc.in_features, num_classes, tasks=4, norm=nn.BatchNorm1d)
wrap(model, convert="scale+bias")
```

### Diversity

- `tau` between 0.1 and 1 works well. A lower `tau` gives more diverse members.
- The diversity of the scales tracks the diversity of the outputs. `get_mean_similarity(model)`
  reads it from the scales, without data.
- With `M` members, each feature is used by some subset of `k` of them, a soft M-choose-k
  allocation. A lower `tau` gives smaller subsets.
- `tau` moves the ensemble along the tradeoff between accuracy and calibration.

The paper explains each point.

## Contents

| path | content |
|------|---------|
| [`examples/demo.ipynb`](examples/demo.ipynb) | train from scratch, wrap a pretrained CNN or ViT, measure uncertainty, choose `τ`, save and load |
| [`examples/benchmarks.ipynb`](examples/benchmarks.ipynb) | rerun the paper's benchmarks and compare with its numbers |
| [`integrations/`](integrations/) | training code for the benchmarks |

## Citation

```bibtex
@article{suteu2026snens,
  title   = {Controllable Diversity in Normalization-Based Implicit Ensembles via
             Softmax-Temperature Modulation},
  author  = {Suteu, Mihai and Serban, Ovidiu},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=Jk5B1JFrPK}
}
```
