# σN-Ens on ViTs in TorchUncertainty

Converts a pretrained ViT-B/16 into σN-Ens and fine-tunes it on CIFAR-100 or ImageNet, with
[TorchUncertainty](https://github.com/torch-uncertainty/torch-uncertainty). These are the
paper's ViT experiments. The test adds the OpenOOD near and far sets, which need
TorchUncertainty's OodEval branch.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e ../../..         # tssnorm, from the root of this repo
```

## Train and test

```bash
python train.py fit --config configs/cifar100/vit_b16.yaml --data.root ./data
python train.py fit --config configs/imagenet/vit_b16.yaml --data.root <imagenet>
```

After training, `train.py` tests the model on the in-distribution test set and the OpenOOD sets.
CIFAR-100 and the OpenOOD sets download on the first run. ImageNet does not. Point
`--data.root` at a folder with `train/`, `val/` and the devkit. ImageNet's OpenOOD sets take
about 40 GB with their archives.

Batch 256 fits an 80 GB GPU. On a smaller one, lower `--data.batch_size` and raise
`--trainer.accumulate_grad_batches` to match.

## From a notebook

```python
from train import main

main(["fit", "--config", "configs/cifar100/vit_b16.yaml", "--data.root", "./data"])
```

[`examples/benchmarks.ipynb`](../../examples/benchmarks.ipynb) runs both configs and compares
the results with the paper.

## Load a trained model

```python
import torch
from tssnorm import forward_tasks
from tssnorm_tu import SigmaNEns

routine = SigmaNEns.load_from_checkpoint("logs/.../checkpoints/<checkpoint>.ckpt").eval()
with torch.no_grad():
    probs = forward_tasks(routine.model.model, x, stack=True).softmax(-1).mean(0)
```

## Use in your own project

[`tssnorm_tu.py`](tssnorm_tu.py) is self-contained. `SigmaNEns` is a TorchUncertainty
classification routine, and `ViTCIFAR100` and `ViTImageNet` are its datamodules:

```python
from functools import partial

import torch
from torch_uncertainty import TUTrainer
from tssnorm_tu import SigmaNEns, ViTCIFAR100

routine = SigmaNEns(num_classes=100, tau=0.5, reg_on=["blocks.*.norm2"], eval_ood=True,
                    optimizer=partial(torch.optim.SGD, lr=0.01, momentum=0.9))
TUTrainer(max_steps=10_000, precision="bf16-mixed").fit(
    routine, ViTCIFAR100("./data", batch_size=256, eval_ood=True))
```
