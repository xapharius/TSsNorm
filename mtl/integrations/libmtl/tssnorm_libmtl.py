"""TSsBN and TSBN for LibMTL, without editing LibMTL.

Targets LibMTL's GitHub `main`, which LibMTL's own `examples/` are written for. `register()`
adds `TSsBN` and `TSBN` to `LibMTL.architecture`, where LibMTL looks architectures up by name.
`TSsBNTrainer` subclasses `Trainer` and overrides `_prepare_optimizer` to give the task-specific
scalers their higher learning rate. `Trainer.__init__` already calls `_prepare_optimizer`, so
nothing else changes. `TSsBNTrainer` also uses its `device` argument, where the upstream
`Trainer` always uses `cuda:0`.

Usage
-----
>>> from tssnorm_libmtl import TSsBNTrainer, register
>>> register()                                       # make architecture="TSsBN" resolvable
>>> trainer = TSsBNTrainer(
...     task_dict=..., weighting="EW", architecture="TSsBN",
...     encoder_class=..., decoders=...,             # a backbone that uses BatchNorm
...     arch_args={"convert": "clip+bias"},           # keep the pretrained scales
...     bn_lr_mul=100, device="cuda:0",
...     ...,
... )
>>> trainer.train(train_loader, test_loader, epochs)
"""

import torch

import LibMTL.architecture
from LibMTL import Trainer
from LibMTL.architecture.abstract_arch import AbsArchitecture

from tssnorm import (
    SIG_BNS,
    TSNorm,
    active_task,
    get_importance,
    get_other_param_group,
    get_param_group,
    taskify,
    wrap,
)

__all__ = ["TSsBN", "TSBN", "TSsBNTrainer", "register"]


class _TSNormArch(AbsArchitecture):
    """Shared behaviour: a per-task forward over a shared, norm-modulated encoder."""

    def forward(self, inputs, task_name=None):
        out = {}
        for task in self.task_name:
            if task_name is not None and task != task_name:
                continue
            with active_task(self.encoder, task):
                rep = self.encoder(inputs)
            rep = self._prepare_rep(rep, task, same_rep=False)
            out[task] = self.decoders[task](rep)
        return out

    def get_share_params(self):
        """The shared parameters: everything in the encoder that is not task-specific."""
        return get_other_param_group(self.encoder, excluding=TSNorm)["params"]

    def zero_grad_share_params(self):
        for param in self.get_share_params():
            param.grad = None


class TSsBN(_TSNormArch):
    """Task-Specific sigmoid-modulated normalization: shared encoder, per-task scalers.

    `convert` and `init` go to `tssnorm.wrap`, through LibMTL's `arch_args`. The default suits
    a pretrained backbone: it keeps the pretrained scales and the frozen shift. For a from-scratch encoder use `convert=None, init="const"`.
    """

    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device,
                 convert="clip+bias", init=None, **kwargs):
        super().__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        self.encoder = self.encoder_class()
        wrap(self.encoder, task_name, convert=convert, init=init)

    def importance(self, per_layer=False):
        """The task-filter importance matrix `sigmoid(gamma)`."""
        return get_importance(self.encoder, per_layer=per_layer)


class TSBN(_TSNormArch):
    """Task-Specific BatchNorm baseline: per-task BN, without the sigmoid modulation.

    Has no scalers to accelerate, so run it through `TSsBNTrainer` too: the multiplier groups
    come out empty and it trains every parameter at the base learning rate.
    """

    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super().__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        self.encoder = self.encoder_class()
        taskify(self.encoder, task_name)


_OPTIMS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adagrad": torch.optim.Adagrad,
    "rmsprop": torch.optim.RMSprop,
}
_SCHEDULERS = {
    "exp": torch.optim.lr_scheduler.ExponentialLR,
    "step": torch.optim.lr_scheduler.StepLR,
    "cos": torch.optim.lr_scheduler.CosineAnnealingLR,
    "reduce": torch.optim.lr_scheduler.ReduceLROnPlateau,
}


class TSsBNTrainer(Trainer):
    """`LibMTL.Trainer` with discriminative learning rates for the task-specific scalers.

    The sigmoid-BN scalers train at `lr * bn_lr_mul` with weight decay off, everything else at
    `lr`. Only `_prepare_optimizer` and the device are overridden. The rest of
    `Trainer` is untouched.

    `device` defaults to `cuda:0`, like upstream. Unlike upstream `Trainer`, which always uses
    `cuda:0`, this one uses the argument, so `"cpu"` works too.
    """

    def __init__(self, *args, bn_lr_mul=100, device="cuda:0", **kwargs):
        # Set before super().__init__, which prepares the model and optimizer during construction.
        self.bn_lr_mul = bn_lr_mul
        self._device = device
        super().__init__(*args, **kwargs)

    def _prepare_model(self, *args, **kwargs):
        self.device = torch.device(self._device)   # upstream sets cuda:0 before this call
        super()._prepare_model(*args, **kwargs)

    def _prepare_optimizer(self, optim_param, scheduler_param):
        base = {k: v for k, v in optim_param.items() if k != "optim"}
        lr = optim_param["lr"]

        groups = [
            {**get_param_group(self.model, SIG_BNS), **base, "lr": lr * self.bn_lr_mul, "weight_decay": 0.0},
            {**get_other_param_group(self.model, excluding=SIG_BNS), **base},
        ]
        groups = [g for g in groups if g["params"]]  # drop empty (e.g. TSBN has no sigmoid-BN scalers)
        self.optimizer = _OPTIMS[optim_param["optim"]](groups)

        if scheduler_param is not None:
            scheduler_arg = {k: v for k, v in scheduler_param.items() if k != "scheduler"}
            self.scheduler = _SCHEDULERS[scheduler_param["scheduler"]](self.optimizer, **scheduler_arg)
        else:
            self.scheduler = None


def register():
    """Make `TSsBN` and `TSBN` resolvable by name in LibMTL's architecture registry."""
    LibMTL.architecture.TSsBN = TSsBN
    LibMTL.architecture.TSBN = TSBN
