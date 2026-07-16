# TSsNorm — Task-Specific Sigmoid Normalization

Reference implementation for **"Simplifying Multi-Task Architectures Through Task-Specific
Normalization"** (TMLR).

Multi-task learning, using normalization layers alone. Every `BatchNorm` / `LayerNorm` in a
shared backbone is made **sigmoid-modulated** (its affine scale becomes `sigmoid(gamma)`, a
bounded per-feature scaler) and **task-specific** (one copy per task). No task-specific feature
extractors, no routing — the scalers do the work, and they double as an interpretable
task-filter importance matrix.

```python
import torch
from tssnorm import (wrap, active_task, get_param_group, get_other_param_group,
                     SIG_BNS, SIG_LNS, SIG_NORMS)

tasks, lr = ["seg", "depth", "normals"], 1e-4
wrap(model.encoder, tasks)                                # BN/LN -> per-task sigmoid norms

optim = torch.optim.SGD([                                 # discriminative learning rates
    {**get_param_group(model, SIG_BNS), "lr": lr * 100},  # BN scalers   (alpha = 100)
    {**get_param_group(model, SIG_LNS), "lr": lr * 10},   # LN scalers   (alpha = 10)
    {**get_other_param_group(model, excluding=SIG_NORMS), "lr": lr},
])

out = {}
for task in tasks:                                        # one forward per task
    with active_task(model.encoder, task):
        out[task] = model(x)
```

The three groups partition the model's parameters: the scalers train fast, everything else at
the base rate. A group is simply empty if the backbone has no norms of that kind.

`wrap` modulates `BatchNorm1d/2d` and `LayerNorm`. It refuses a normalization layer it does not
recognise rather than leaving it shared behind your back — a subclass can fold extra behaviour
into its forward that replacing it would drop (timm's `BatchNormAct2d` fuses an activation).
Where the swap *is* safe, opt in once:

```python
import timm.layers
from tssnorm import register_norm, SigLayerNorm

register_norm(timm.layers.LayerNorm, SigLayerNorm)        # timm ViTs, DeiT, …
```

See [`examples/basic_usage.ipynb`](examples/basic_usage.ipynb) for the full tour.

## Install

```bash
pip install -e .                # the core `tssnorm` package (torch only)
pip install -e ".[analysis]"    # plus the Section 5 analysis toolkit (numpy, scipy, matplotlib)
```

To run the integration examples as well, recreate the test environment:

```bash
pip install -r requirements.txt
```


## What's here

| path | what |
|------|------|
| `tssnorm/` | the method: `norms.py` (layers) + `utils.py` (model surgery, param groups, importance) |
| `tssnorm/analysis.py` | the Section 5 toolkit: capacity, task relationships, filter specialization |
| `examples/basic_usage.ipynb` | wrap a model, param groups, init, dispatch, importance, pretrained conversion |
| `analysis/` | demo of the analysis toolkit on a trained NYUv2 TSsBN model |
| `integrations/libmtl/` | TSsBN in [LibMTL](https://github.com/median-research-group/LibMTL) — Table 2 |
| `integrations/mlore/` | TSsBN on ViT-S in [MLoRE](https://github.com/YuqiYang213/MLoRE) — Table 3 |

## Citation

```bibtex
@article{suteu2026tssnorm,
  title   = {Simplifying Multi-Task Architectures Through Task-Specific Normalization},
  author  = {Suteu, Mihai and Serban, Ovidiu},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=QNO893OXrS}
}
```

## License

MIT.
