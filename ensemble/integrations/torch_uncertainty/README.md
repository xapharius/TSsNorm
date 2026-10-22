# σN-Ens in TorchUncertainty

Trains and tests σN-Ens with
[TorchUncertainty](https://github.com/torch-uncertainty/torch-uncertainty) on CIFAR-10 and
CIFAR-100. These are the paper's CNN experiments.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e ../../..         # tssnorm, from the root of this repo
```

`requirements.txt` pins the versions we tested with.

## Train and test

```bash
python train.py fit --config configs/cifar10/resnet18.yaml --data.root ./data
```

After training, `train.py` tests the model. The test reports accuracy, NLL, calibration and
out-of-distribution detection against SVHN. The data is downloaded on the first run.

Fine-tune a trained single model:

```bash
python train.py fit --config configs/cifar10/resnet18_finetune.yaml --data.root ./data \
    --model.checkpoint single.ckpt
```

Change any setting from the command line:

```bash
python train.py fit --config configs/cifar10/resnet18.yaml --data.root ./data \
    --model.tau 1.0 --model.num_estimators 8      # 8 members
```

## From a notebook

```python
from train import main

main(["fit", "--config", "configs/cifar10/resnet18.yaml", "--data.root", "./data"])
```

[`examples/benchmarks.ipynb`](../../examples/benchmarks.ipynb) runs every config and compares
the results with the paper.

## Configs

`configs/<dataset>/<backbone>.yaml` trains from scratch. `configs/<dataset>/<backbone>_finetune.yaml`
converts a trained single model and fine-tunes it for 25 epochs. The datasets are `cifar10` and
`cifar100`. The backbones are `resnet18`, `resnet50` and `wideresnet28x10`.

These configs use 4 members and `τ = 0.1`. The from-scratch recipes are TorchUncertainty's,
without dropout. The fine-tune configs use Adam with a cosine schedule.

To fine-tune, first train a single model for the same dataset and backbone, with
TorchUncertainty's standard config. For ResNet-18 on CIFAR-10 that is
`experiments/classification/cifar10/configs/resnet18/standard.yaml` in the TorchUncertainty
repository.

## Load a trained model

```python
import torch
from tssnorm import forward_tasks
from tssnorm_tu import SigmaNEns

routine = SigmaNEns.load_from_checkpoint("logs/.../checkpoints/<checkpoint>.ckpt").eval()
with torch.no_grad():
    probs = forward_tasks(routine.model.model, x, stack=True).softmax(-1).mean(0)
```

A fine-tuned model also needs the single model's checkpoint, at the path it was trained with.

## Use in your own project

[`tssnorm_tu.py`](tssnorm_tu.py) is self-contained. `SigmaNEns` is a TorchUncertainty
classification routine:

```python
from functools import partial

import torch
from torch_uncertainty import TUTrainer
from torch_uncertainty.datamodules import CIFAR10DataModule
from tssnorm_tu import SigmaNEns

routine = SigmaNEns(num_classes=10, arch="resnet18", tau=0.1, reg_on=["layer*.bn2"],
                    optimizer=partial(torch.optim.SGD, lr=0.05, momentum=0.9, weight_decay=5e-4))
TUTrainer(max_epochs=75).fit(routine, CIFAR10DataModule("./data", batch_size=128))
```
