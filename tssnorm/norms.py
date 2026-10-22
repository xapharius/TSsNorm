"""
Sigmoid-modulated and task-specific normalization layers.

`SigBatchNorm*` / `SigLayerNorm` replace the affine transform of a normalization layer with
a single bounded scaler per feature, and no learned shift. The sigmoid scales each feature
down smoothly:

    sigma_norm(x; gamma) = k * sigmoid(gamma) * x_hat

`k` is a fixed per-layer constant. It is 1, except after `convert="scale"`, which sets it to
`max|scale| / 0.95`.

`TSModule` holds one module per task and dispatches on the active task. `TSNorm` is a
`TSModule` of normalization layers. A model whose norms are `TSNorm`s of sigmoid norms shares
every other weight between its tasks.

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
    "TSModule",
    "TSNorm",
    "convert_weights",
    "init_weight",
]


def init_weight(weight: Tensor, how: str = "const") -> None:
    """Initialise a modulation weight `gamma` in place.

    `const` starts every scaler at the centre of its range, sigmoid(0) = 0.5, so no feature is
    preferred. `normal` and `uniform` spread the scalers over (0, 1), so the tasks start
    different.
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

    `None` discards them (train from scratch). Otherwise the pretrained scale is mapped onto
    the new scaler:

    - `clip` sets `gamma = logit(scale)`, so every scale in `(0, 1)` is reproduced exactly. A
      scale of 1 or more becomes 0.95, and a scale of 0 or less becomes 0.05.
    - `scale` sets `k = max|scale| / .95` and `gamma = logit(scale / k)`, so `k * sigmoid(gamma)`
      reproduces every positive pretrained scale. Use it when scales exceed 1, as in
      transformer LayerNorms and many ImageNet CNNs.

    Neither can represent a negative scale, which flips a feature's sign. Some training recipes
    learn many; converting such a model changes its function.

    A `+bias` suffix (`clip+bias`, `scale+bias`) also copies the bias and freezes it. Without
    it, an affine layer's shift is lost on conversion.

    Both strategies also carry the replaced layer's running statistics. They are as much of a
    pretrained norm's learned state as its scale, and starting them from scratch would undo the
    conversion the moment the model is evaluated.
    """
    if how is None:
        return
    strategy, _, suffix = how.partition("+")
    if strategy not in ("clip", "scale") or suffix not in ("", "bias"):
        raise ValueError(
            f"unknown convert {how!r}; use None, 'clip', 'clip+bias', 'scale', or 'scale+bias'"
        )
    if suffix and old.bias is None:
        raise ValueError(
            f"convert={how!r} has no bias to copy: this {type(old).__name__} has none "
            f"(nn.LayerNorm(bias=False), say). Use convert={strategy!r}."
        )
    scale = old.weight.detach().clone()
    if strategy == "clip":
        weight = torch.where(scale >= 1, 0.95, torch.where(scale <= 0, 0.05, scale)).logit()
    else:
        k = scale.abs().max() / 0.95
        weight = (scale.clamp(min=1e-3) / k).logit()
        new.k.fill_(k)
    new.weight.data.copy_(weight.to(new.weight))
    if suffix:
        new.bias = nn.Parameter(old.bias.detach().clone().to(new.weight), requires_grad=False)
    for name, buffer in old.named_buffers(recurse=False):    # BN's running stats; LN has none
        target = getattr(new, name, None)
        if isinstance(target, Tensor) and target.shape == buffer.shape:
            target.data.copy_(buffer.data.to(target))


class _Modulated(nn.Module):
    """What the modulated norms share: the scale `k * sigmoid(gamma)`, and no learned bias."""

    weight: nn.Parameter
    k: Tensor

    def _modulate(self, init: str) -> None:
        self.bias = None
        self.register_buffer("k", torch.ones((), dtype=self.weight.dtype, device=self.weight.device))
        init_weight(self.weight, init)

    def scale(self) -> Tensor:
        return self.k * self.weight.sigmoid()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.setdefault(prefix + "k", torch.ones_like(self.k))   # saved before `k` existed
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class _SigBatchNorm(_Modulated, _BatchNorm):
    """Shared behaviour for the sigmoid-modulated BatchNorms."""

    def __init__(self, *args, init: str = "const", **kwargs):
        super().__init__(*args, **kwargs)
        if self.weight is None:
            raise ValueError(f"{type(self).__name__} needs affine=True: the scale is the method")
        self._modulate(init)

    def forward(self, input: Tensor) -> Tensor:
        """`_BatchNorm.forward`, with `k * sigmoid(gamma)` as the scale."""
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
            self.scale(),
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


class SigLayerNorm(_Modulated, nn.LayerNorm):
    """Sigmoid-modulated LayerNorm."""

    def __init__(self, *args, init: str = "const", **kwargs):
        super().__init__(*args, **kwargs)
        if self.weight is None:
            raise ValueError(f"{type(self).__name__} needs elementwise_affine=True: "
                             "the scale is the method")
        self._modulate(init)

    def forward(self, input: Tensor) -> Tensor:
        # bias is None unless a `+bias` conversion copied a frozen beta onto it.
        return F.layer_norm(input, self.normalized_shape, self.scale(), self.bias, self.eps)

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


class TSModule(nn.ModuleDict):
    """One module per task, dispatching on the active task.

    The task can be given per call, or set once for a whole forward pass with
    `set_active_task` / the `active_task` context manager.
    """

    def __init__(self, modules: dict):
        if not modules:
            raise ValueError(f"{type(self).__name__} needs at least one task")
        super().__init__({str(task): module for task, module in modules.items()})
        self.active_task: Optional[str] = None

    @property
    def tasks(self) -> List[str]:
        return list(self.keys())

    @classmethod
    def like(cls, module: nn.Module, tasks: Iterable):
        """One deep copy of `module` per task."""
        return cls({task: copy.deepcopy(module) for task in tasks})

    def forward(self, x: Tensor, task=None) -> Tensor:
        task = self.active_task if task is None else task
        if task is None:
            raise RuntimeError(
                "no active task; set one with set_active_task() or the active_task() "
                f"context manager, or pass task= to this layer. Tasks: {self.tasks}"
            )
        if (task := str(task)) not in self:
            raise KeyError(f"no module for task {task!r}; have {self.tasks}")
        return self[task](x)

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{len(self)} x {next(iter(self.values()))}]"


class TSNorm(TSModule):
    """One normalization layer per task."""

    def weights(self) -> Tensor:
        """The per-task scalers, stacked into a `[tasks, features]` matrix."""
        return torch.stack([norm.weight for norm in self.values()])
