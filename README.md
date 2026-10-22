# TSsNorm

The official repo for TSσBN and σN-Ens, both relying on the σNorm primitive for modulation.

The method changes the normalization layers of a network. The scale of each BatchNorm and
LayerNorm becomes `sigmoid(gamma)`, and the learned shift is dropped. Each task or member gets its own
copy of these layers, while the rest of the backbone is shared. The sigmoid keeps every scale between
0 and 1, and can be interpreted as the importance of that channel for a given task.

Based on these importances, capacity and relationships can be read out and influenced during training.


The two papers apply the method to different kinds of task.

| folder | use | a task is | paper |
|---|---|---|---|
| [`mtl/`](mtl/) | multi-task learning (TSσNorm) | one prediction target, such as depth | *Simplifying Multi-Task Architectures Through Task-Specific Normalization* |
| [`ensemble/`](ensemble/) | implicit ensembles (σN-Ens) | an ensemble member | *Controllable Diversity in Normalization-Based Implicit Ensembles via Softmax-Temperature Modulation* |


## Usage

```python
import torch
from tssnorm import active_task, forward_tasks, optim_factory, wrap

tasks = ["seg", "depth", "normals"]
wrap(model.encoder, tasks)                                # sigmoid norms, one copy per task
optim = optim_factory(model, torch.optim.SGD, lr=1e-4)    # scales at 100 times the base rate

features = forward_tasks(model.encoder, x)                # {task: output}, one pass per task
with active_task(model.encoder, "seg"):                   # or one task at a time
    seg = model.encoder(x)
```

The scales need a higher learning rate than the other weights. `optim_factory` puts them in
their own parameter group. For other groupings, `get_optimizer` takes groups with an `lr_mul` each.

Each application has its own README. [`mtl/`](mtl/) covers multi-task learning: an analysis of
what the scales learned, and two integrations. [`ensemble/`](ensemble/) covers ensembles: training
from scratch or from a pretrained model, uncertainty, and the benchmarks.

`wrap` handles `BatchNorm1d`, `BatchNorm2d` and `LayerNorm`. It raises an error for any other
norm layer. Some subclasses change the forward pass. For example, timm's `BatchNormAct2d` adds
an activation. If a subclass is safe to replace, register it once:

```python
import timm.layers
from tssnorm import register_norm, SigLayerNorm

register_norm(timm.layers.LayerNorm, SigLayerNorm)
```

[`examples/basic_usage.ipynb`](examples/basic_usage.ipynb) tours the whole API on a small
model.

## Install

```bash
pip install -e .                # the tssnorm package, needs only torch
pip install -e ".[analysis]"    # adds numpy, scipy and matplotlib for tssnorm.mtl
```

`requirements.txt` pins the environment for the package and the example notebooks. Each
integration has its own `requirements.txt`. Install each in its own environment, because the
frameworks pin different versions.

## Layout

| path | content |
|------|---------|
| `tssnorm/norms.py` | the layers |
| `tssnorm/utils.py` | model surgery, heads, optimisers, importance and similarity |
| `tssnorm/losses.py` | the diversity regulariser |
| `tssnorm/mtl.py` | analysis for multi-task learning |
| `examples/basic_usage.ipynb` | tour of the API |
| `mtl/` | multi-task learning: analysis notebook, LibMTL and MLoRE integrations |
| `ensemble/` | implicit ensembles: demo and benchmark notebooks, TorchUncertainty integrations |

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

@article{suteu2026sigmanens,
  title   = {Controllable Diversity in Normalization-Based Implicit Ensembles via
             Softmax-Temperature Modulation},
  author  = {Suteu, Mihai and Serban, Ovidiu},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=Jk5B1JFrPK}
}
```

## License

MIT.
