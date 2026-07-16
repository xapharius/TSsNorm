"""
Sigmoid-modulated and task-specific normalization layers.

`SigBatchNorm*` / `SigLayerNorm` replace the affine transform of a normalization layer with
a single bounded scaler per feature. The sigmoid modulates feature usage — it down-weights,
it does not binarise:

    sigma_norm(x; gamma) = sigmoid(gamma) * x_hat

`TSNorm` holds one normalization layer per task and dispatches on the active task. Composing
the two gives TSsBN/TSsLN: fully shared feature extractors, task-specific modulation.

Example
-------
>>> import torch, torch.nn as nn
>>> from tssnorm.norms import SigBatchNorm2d, TSNorm
>>>
>>> # a fresh layer, its scalers initialised from scratch ...
>>> norm = SigBatchNorm2d(16, init="normal")
>>> # ... or one shaped after an existing layer (copies no weights):
>>> norm = SigBatchNorm2d.like(nn.BatchNorm2d(16))
>>>
>>> # one copy per task, dispatched on whichever task is active:
>>> ts = TSNorm.like(norm, tasks=["seg", "depth"])
>>> ts.active_task = "seg"
>>> y = ts(torch.randn(4, 16, 8, 8))

The BatchNorm forward is adapted from
https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html
"""

import copy
from typing import Iterable, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = [
    "SigBatchNorm1d",
    "SigBatchNorm2d",
    "SigLayerNorm",
    "TSNorm",
    "convert_weights",
    "init_weight",
]


def init_weight(weight: Tensor, how: str = "const") -> None:
    """Initialise a modulation weight `gamma` in place.

    `const` starts every scaler at the centre of its range, sigmoid(0) = 0.5, so no feature is
    preferred and every task can move freely in either direction; `normal` and `uniform` spread
    them over (0, 1) instead, starting the tasks already differentiated.
    """
    if how == "const":
        nn.init.constant_(weight, 0.0)
    elif how == "normal":
        nn.init.normal_(weight, mean=0.0, std=1.0)
    elif how == "uniform":
        nn.init.uniform_(weight, -3.0, 3.0)
    else:
        raise ValueError(f"unknown init {how!r}; use 'const', 'normal', or 'uniform'")


def convert_weights(old: nn.Module, new: nn.Module, how: Optional[str] = None) -> None:
    """Carry a replaced norm's weights onto its modulated counterpart, in place.

    `None` discards them (train from scratch). `clip` maps the old scale onto the new scaler as
    `gamma <- logit(clip(scale, .05, .95))`, so `sigmoid(gamma)` recovers the pretrained scale
    wherever it already lay inside `(.05, .95)` — most of it does (App. F.3); scales outside
    saturate at the clamp. `clip+bias` also copies the frozen bias — an affine layer cannot be
    made linear unless beta is zero, so keeping beta avoids a conversion shock.

    Either strategy also carries the replaced layer's running statistics. They are as much of a
    pretrained norm's learned state as its scale, and starting them from scratch would undo the
    conversion the moment the model is evaluated.
    """
    if how is None:
        return
    if how not in ("clip", "clip+bias"):
        raise ValueError(f"unknown convert {how!r}; use None, 'clip', or 'clip+bias'")
    if how == "clip+bias" and old.bias is None:
        raise ValueError(
            f"convert='clip+bias' has no bias to copy: this {type(old).__name__} has none "
            "(nn.LayerNorm(bias=False), say). Use convert='clip'."
        )
    weight = old.weight.data.clone().clamp(0.05, 0.95).logit()
    new.weight.data.copy_(weight.to(new.weight))
    if how == "clip+bias":
        new.bias = nn.Parameter(old.bias.data.clone().to(new.weight), requires_grad=False)
    for name, buffer in old.named_buffers(recurse=False):    # BN's running stats; LN has none
        target = getattr(new, name, None)
        if isinstance(target, Tensor) and target.shape == buffer.shape:
            target.data.copy_(buffer.data.to(target))


class _SigBatchNorm(_BatchNorm):
    """Shared behaviour for the sigmoid-modulated BatchNorms."""

    def __init__(self, *args, init: str = "const", **kwargs):
        super().__init__(*args, **kwargs)
        if self.weight is None:
            raise ValueError(f"{type(self).__name__} needs affine=True: the scale is the method")
        self.bias = None
        init_weight(self.weight, init)

    def forward(self, input: Tensor) -> Tensor:
        """`_BatchNorm.forward`, with `sigmoid(gamma)` as the scale."""
        self._check_input_dim(input)

        momentum = self.momentum or 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1  # type: ignore[has-type]
            if self.momentum is None:  # cumulative moving average
                momentum = 1.0 / float(self.num_batches_tracked)

        # Batch statistics while training, and when there are no running ones to fall back on.
        use_batch_stats = self.training or self.running_mean is None
        # Running statistics are only passed when they are read (eval) or updated (train).
        stateful = not self.training or self.track_running_stats

        return F.batch_norm(
            input,
            self.running_mean if stateful else None,
            self.running_var if stateful else None,
            self.weight.sigmoid(),
            self.bias,
            use_batch_stats,
            momentum,
            self.eps,
        )

    @classmethod
    def like(cls, bn: _BatchNorm, init: str = "const"):
        """A fresh layer shaped like `bn`: same device, dtype, and train/eval mode, so replacing
        `bn` with it changes nothing but the scale. Copies no weights."""
        if not bn.affine:
            raise ValueError(
                f"cannot modulate {type(bn).__name__}(affine=False): it has no scale to replace"
            )
        return cls(
            num_features=bn.num_features,
            eps=bn.eps,
            momentum=bn.momentum,
            affine=True,
            track_running_stats=bn.track_running_stats,
            init=init,
        ).to(device=bn.weight.device, dtype=bn.weight.dtype).train(bn.training)


class SigBatchNorm1d(_SigBatchNorm, nn.BatchNorm1d):
    pass


class SigBatchNorm2d(_SigBatchNorm, nn.BatchNorm2d):
    pass


class SigLayerNorm(nn.LayerNorm):
    """Sigmoid-modulated LayerNorm."""

    def __init__(self, *args, init: str = "const", **kwargs):
        super().__init__(*args, **kwargs)
        if self.weight is None:
            raise ValueError(f"{type(self).__name__} needs elementwise_affine=True: "
                             "the scale is the method")
        self.bias = None          # the scaler is the only affine parameter
        init_weight(self.weight, init)

    def forward(self, input: Tensor) -> Tensor:
        # bias is None unless a `clip+bias` conversion copied a frozen beta onto it.
        return F.layer_norm(input, self.normalized_shape, self.weight.sigmoid(), self.bias, self.eps)

    @classmethod
    def like(cls, ln: nn.LayerNorm, init: str = "const"):
        """A fresh layer shaped like `ln`: same device, dtype, and train/eval mode. Copies no
        weights."""
        if not ln.elementwise_affine:
            raise ValueError(
                f"cannot modulate {type(ln).__name__}(elementwise_affine=False): "
                "it has no scale to replace"
            )
        return cls(
            normalized_shape=ln.normalized_shape,
            eps=ln.eps,
            elementwise_affine=True,
            init=init,
        ).to(device=ln.weight.device, dtype=ln.weight.dtype).train(ln.training)


class TSNorm(nn.ModuleDict):
    """One normalization layer per task, dispatching on the active task.

    The task can be given per call, or set once for a whole forward pass with
    `set_active_task` / the `active_task` context manager.
    """

    def __init__(self, norms: dict):
        if not norms:
            raise ValueError("TSNorm needs at least one task")
        super().__init__({str(task): norm for task, norm in norms.items()})
        self.active_task: Optional[str] = None

    @property
    def tasks(self) -> List[str]:
        return list(self.keys())

    @classmethod
    def like(cls, norm: nn.Module, tasks: Iterable):
        """One deep copy of `norm` per task."""
        return cls({task: copy.deepcopy(norm) for task in tasks})

    def forward(self, x: Tensor, task=None) -> Tensor:
        task = self.active_task if task is None else task
        if task is None:
            raise RuntimeError(
                "no active task; set one with set_active_task() or the active_task() "
                f"context manager, or pass task= to this layer. Tasks: {self.tasks}"
            )
        if (task := str(task)) not in self:
            raise KeyError(f"no norm for task {task!r}; have {self.tasks}")
        return self[task](x)

    def weights(self) -> Tensor:
        """The per-task scalers, stacked into a `[tasks, features]` matrix."""
        return torch.stack([norm.weight for norm in self.values()])

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{len(self)} x {next(iter(self.values()))}]"
