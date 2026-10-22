"""σN-Ens in TorchUncertainty.

`SigmaNEns` is a TorchUncertainty classification routine. It builds the ensemble on a
TorchUncertainty CNN, adds the softmax regulariser to the loss, and gives the scalers their own
learning rate. Run it from the command line with `train.py`, or use it in Python:

>>> from functools import partial
>>> routine = SigmaNEns(num_classes=10, arch="resnet18", reg_on=["layer*.bn2"],
...                     optimizer=partial(torch.optim.SGD, lr=0.05, momentum=0.9))
>>> TUTrainer(max_epochs=75).fit(routine, CIFAR10DataModule(root="./data", batch_size=128))
"""

from typing import List, Optional

import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import nn
from torch_uncertainty.models.classification import resnet, wideresnet28x10
from torch_uncertainty.routines import ClassificationRoutine
from torch_uncertainty.transforms import RepeatTarget

from tssnorm import (forward_tasks, get_mean_similarity, get_norms, make_heads, optim_factory,
                     softmax_loss, wrap)


def backbone(arch: str, num_classes: int) -> nn.Module:
    """A TorchUncertainty CIFAR backbone: `resnet18`, `resnet50` or `wideresnet28x10`."""
    if arch in ("resnet18", "resnet50"):
        return resnet(arch=int(arch[len("resnet"):]), num_classes=num_classes, in_channels=3,
                      style="cifar", conv_bias=False, dropout_rate=0.0)
    if arch == "wideresnet28x10":
        return wideresnet28x10(num_classes=num_classes, in_channels=3, style="cifar",
                               conv_bias=True, dropout_rate=0.0)
    raise ValueError(f"unknown arch {arch!r}; use 'resnet18', 'resnet50' or 'wideresnet28x10'")


def load(model: nn.Module, checkpoint: str) -> None:
    """Load a trained TorchUncertainty model, from a Lightning checkpoint or a state dict.

    TorchUncertainty's WideResNet baseline has no conv biases. Missing ones are set to zero,
    so the model computes what the checkpoint did.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {k.removeprefix("model."): v for k, v in state.get("state_dict", state).items()}
    conv_biases = {f"{n}.bias": m.bias for n, m in model.named_modules()
                   if isinstance(m, nn.Conv2d) and m.bias is not None}
    state = {**{k: torch.zeros_like(b) for k, b in conv_biases.items()}, **state}
    model.load_state_dict(state)


class Ensemble(nn.Module):
    """Runs every member and concatenates their logits into `[members * batch, classes]`,
    the layout TorchUncertainty expects."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return forward_tasks(self.model, x, stack=True).flatten(0, 1)


def sigma_n_ens(arch: str, num_classes: int, num_estimators: int = 4,
                checkpoint: Optional[str] = None) -> Ensemble:
    """A σN-Ens model, trained from scratch or converted from a single model's checkpoint."""
    model = backbone(arch, num_classes)
    if checkpoint is None:
        model.linear = make_heads(model.linear.in_features, num_classes, tasks=num_estimators)
        wrap(model, init="normal")
    else:
        load(model, checkpoint)
        # The heads get a plain norm, which wrap converts together with the backbone's.
        model.linear = make_heads(model.linear.in_features, num_classes, tasks=num_estimators,
                                  norm=nn.BatchNorm1d)
        wrap(model, convert="clip+bias")
    return Ensemble(model)


class SigmaNEns(ClassificationRoutine):
    """σN-Ens as a TorchUncertainty classification routine.

    Args:
        num_classes: number of classes.
        arch: `resnet18`, `resnet50` or `wideresnet28x10`.
        num_estimators: number of members.
        tau: temperature of the regulariser. A lower `tau` makes the members more diverse.
        reg_weight: weight of the regulariser in the loss.
        reg_on: name patterns of the regularised norms, such as `["layer*.bn2"]`. `None`
            regularises every norm.
        lr_mul: learning rate of the scalers, relative to the rest.
        checkpoint: a trained single model to convert. `None` trains from scratch.
        optimizer: builds the optimizer from the parameter groups.
        lr_scheduler: builds the learning rate scheduler from the optimizer.
        eval_ood: evaluate out-of-distribution detection.
        eval_shift: evaluate under distribution shift.
    """

    def __init__(
        self,
        num_classes: int,
        arch: str = "resnet18",
        num_estimators: int = 4,
        tau: float = 0.1,
        reg_weight: float = 1e-2,
        reg_on: Optional[List[str]] = None,
        lr_mul: float = 100.0,
        checkpoint: Optional[str] = None,
        optimizer: OptimizerCallable = torch.optim.SGD,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
        eval_ood: bool = False,
        eval_shift: bool = False,
    ):
        super().__init__(
            model=sigma_n_ens(arch, num_classes, num_estimators, checkpoint),
            num_classes=num_classes,
            loss=nn.CrossEntropyLoss(),
            is_ensemble=True,
            format_batch_fn=RepeatTarget(num_estimators),
            eval_ood=eval_ood,
            eval_shift=eval_shift,
        )
        self.save_hyperparameters(ignore=["optimizer", "lr_scheduler"])
        self.tau, self.reg_weight, self.lr_mul = tau, reg_weight, lr_mul
        self.reg_norms = get_norms(self.model.model, *(reg_on or []))
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler

    def training_step(self, batch):
        loss = super().training_step(batch)
        reg = softmax_loss(self.reg_norms, self.tau)
        self.log("train_reg", reg)
        return loss + self.reg_weight * reg

    def on_train_epoch_end(self):
        """Log how similar the members are, from their scales."""
        self.log("train/similarity", get_mean_similarity(self.model.model))

    def configure_optimizers(self):
        optimizer = optim_factory(self.model, self.optimizer, lr_mul=self.lr_mul)
        if self.lr_scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": self.lr_scheduler(optimizer)}

