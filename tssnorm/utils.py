"""
Model surgery, parameter grouping, and importance readout.

The method is three steps, applied to any model that already uses normalization layers:

    sigmoidify(model)                # BN/LN  ->  modulated sigma-BN/LN
    taskify(model, tasks)            # each norm  ->  one copy per task
    init_norms(model, how="const")   # initialise the scalers

`wrap` runs all three. Where the norms of a model need different treatment (a pretrained
backbone whose scales exceed the sigmoid co-domain, say) the steps compose per submodule.

Example
-------
>>> import torch
>>> from tssnorm import wrap, active_task, get_importance
>>> from tssnorm import get_param_group, get_other_param_group, SIG_BNS, SIG_NORMS
>>>
>>> wrap(model.encoder, tasks=["seg", "depth", "normals"])          # from scratch
>>>
>>> # discriminative learning rates: the scalers move faster than what they modulate
>>> optim = torch.optim.SGD([
...     {**get_param_group(model, SIG_BNS), "lr": lr * 100},
...     {**get_other_param_group(model, excluding=SIG_NORMS), "lr": lr},
... ])
>>>
>>> out = {}                                                        # one forward per task
>>> for task in model.tasks:
...     with active_task(model.encoder, task):
...         out[task] = model(x)
>>>
>>> I = get_importance(model.encoder)                               # [tasks, filters]
"""

from contextlib import contextmanager
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _NormBase

from .norms import (
    SigBatchNorm1d,
    SigBatchNorm2d,
    SigLayerNorm,
    TSNorm,
    convert_weights,
    init_weight,
)

__all__ = [
    "NORMS",
    "register_norm",
    "SIG_BNS",
    "SIG_LNS",
    "SIG_NORMS",
    "active_task",
    "get_importance",
    "get_other_param_group",
    "get_param_group",
    "init_norms",
    "set_active_task",
    "sigmoidify",
    "taskify",
    "wrap",
]

#: The normalization layers we know how to modulate, and what they become. Extend with
#: `register_norm` for a subclass whose forward is equivalent to its base.
_SIGMOID = {
    nn.BatchNorm1d: SigBatchNorm1d,
    nn.BatchNorm2d: SigBatchNorm2d,
    nn.LayerNorm: SigLayerNorm,
}

#: Every normalization layer torch defines, handled or not. Used to decide what counts as a
#: norm at all, so an unhandled one is reported rather than walked past.
_ALL_NORMS: Tuple[type, ...] = tuple(
    t for t in (_NormBase,                      # BatchNorm*, SyncBatchNorm, InstanceNorm*
                nn.GroupNorm,
                nn.LayerNorm,
                getattr(nn, "RMSNorm", None),   # torch >= 2.4
                nn.LocalResponseNorm)
    if t is not None
)

NORMS: Tuple[type, ...] = tuple(_SIGMOID)
SIG_BNS: Tuple[type, ...] = (SigBatchNorm1d, SigBatchNorm2d)
SIG_LNS: Tuple[type, ...] = (SigLayerNorm,)
SIG_NORMS: Tuple[type, ...] = SIG_BNS + SIG_LNS

Norms = Union[type, Tuple[type, ...]]


def register_norm(norm_cls: type, sig_cls: type) -> None:
    """Teach `sigmoidify` to modulate `norm_cls` by replacing it with `sig_cls`.

    Dispatch is by exact type, because a subclass may fold extra behaviour into its forward
    that a swap would drop — timm's `BatchNormAct2d` fuses an activation, its `LayerNorm2d`
    permutes axes. Register a subclass whose forward *is* equivalent to its base, and it will
    be modulated like the base:

    >>> import timm.layers
    >>> register_norm(timm.layers.LayerNorm, SigLayerNorm)   # a fast-path alias for nn.LayerNorm
    """
    _SIGMOID[norm_cls] = sig_cls


class _Default:
    """Marker for `wrap`'s `init`, whose default depends on whether `convert` was given."""


_DEFAULT = _Default()


# --- surgery --------------------------------------------------------------------------


def _children(module: nn.Module):
    """`named_children()` without its deduplication.

    `named_children` yields a reused module instance only under its first name, which for
    surgery would leave the other names holding the original layer. Replacing every name keeps
    the model whole; memoizing on identity keeps whatever tying it asked for.
    """
    return [(name, child) for name, child in module._modules.items() if child is not None]


def _refuse_unmodulatable(module: nn.Module, selected: Norms, prefix: str = "") -> None:
    """Report any selected norm we have no modulated counterpart for, before replacing any.

    A norm subclass may fold extra behaviour into its forward that replacing it would drop —
    timm's `BatchNormAct2d` fuses an activation, its `LayerNorm2d` permutes axes — so it is not
    assumed safe. Checking up front means a refused model is left exactly as it came in, and
    `register_norm` and a retry start from a clean tree.
    """
    for name, child in _children(module):
        path = f"{prefix}{name}"
        if isinstance(child, SIG_NORMS):          # already modulated
            continue
        if isinstance(child, selected):
            if type(child) not in _SIGMOID:
                raise TypeError(
                    f"cannot modulate {type(child).__module__}.{type(child).__name__} at "
                    f"{path!r}: tssnorm modulates {', '.join(t.__name__ for t in NORMS)}. A "
                    "subclass may fold extra behaviour into its forward that replacing it "
                    "would drop, so it is not assumed safe — call register_norm() if it is, "
                    "or narrow `norms` to leave this layer alone."
                )
            continue
        _refuse_unmodulatable(child, selected, f"{path}.")


def sigmoidify(module: nn.Module, convert: Optional[str] = None, init: str = "const",
               norms: Optional[Norms] = None) -> nn.Module:
    """Replace every selected normalization layer with its sigmoid-modulated counterpart,
    in place.

    `convert` names a strategy for inheriting the replaced layer's weights (see
    `convert_weights`); the default discards them and keeps the fresh `init`.

    `norms` selects which layer types to modulate; the default, `None`, means *every*
    normalization layer — so one this cannot modulate is reported rather than quietly walked
    past, which would leave the model partly un-methoded. Narrow `norms` to treat parts of a
    model differently, and the rest are then left alone deliberately. See `register_norm` to
    add support for a norm subclass.
    """
    selected = _ALL_NORMS if norms is None else norms
    _refuse_unmodulatable(module, selected)       # before any surgery: never half-convert a model
    modulated: dict = {}                          # id(old) -> new, for norms reused under two names

    def walk(parent: nn.Module) -> None:
        for name, child in _children(parent):
            if isinstance(child, SIG_NORMS):      # already modulated
                continue
            if isinstance(child, selected):
                if id(child) not in modulated:
                    new = _SIGMOID[type(child)].like(child, init=init)
                    convert_weights(child, new, how=convert)
                    modulated[id(child)] = new
                setattr(parent, name, modulated[id(child)])
                continue
            walk(child)

    walk(module)
    return module


def taskify(module: nn.Module, tasks: Iterable, norms: Optional[Norms] = None) -> nn.Module:
    """Give each task its own copy of every selected normalization layer, in place.

    `norms` defaults to every normalization layer; narrow it to duplicate only some.
    """
    norms = _ALL_NORMS if norms is None else norms
    tasks = list(tasks)
    duplicated: dict = {}                         # id(old) -> TSNorm, for norms reused under two names

    def walk(parent: nn.Module) -> None:
        for name, child in _children(parent):
            if isinstance(child, TSNorm):
                continue
            if isinstance(child, norms):
                if id(child) not in duplicated:
                    duplicated[id(child)] = TSNorm.like(child, tasks)
                setattr(parent, name, duplicated[id(child)])
                continue
            walk(child)

    walk(module)
    return module


def init_norms(module: nn.Module, how: str = "const") -> nn.Module:
    """(Re)initialise every scaler. Run after `taskify`, so a stochastic init draws per task."""
    for norm in module.modules():
        if isinstance(norm, SIG_NORMS):
            init_weight(norm.weight, how)
    return module


def wrap(
    module: nn.Module,
    tasks: Iterable,
    convert: Optional[str] = None,
    init: Union[str, None, "_Default"] = _DEFAULT,
    norms: Optional[Norms] = None,
) -> nn.Module:
    """Modulate the norms, give each task its own, initialise. The method, in one call.

    `init` runs last, so a stochastic init gives every task an independent draw. By default it
    is `"const"` for a from-scratch wrap, and `None` when `convert` is given — initialising
    would otherwise overwrite the scalers `convert` just inherited. Pass `init` explicitly to
    force either behaviour. `norms` defaults to every normalization layer (see `sigmoidify`).
    """
    tasks = [str(t) for t in ([tasks] if isinstance(tasks, str) else tasks)]
    already = {t for m in module.modules() if isinstance(m, TSNorm) for t in m.tasks}
    if already and already != set(tasks):
        raise ValueError(
            f"this model is already task-specific for {sorted(already)}; wrapping it again for "
            f"{sorted(set(tasks))} would be a no-op. Build a fresh model for a new task set."
        )
    if isinstance(init, _Default):
        init = None if convert is not None else "const"
    sigmoidify(module, convert=convert, init=init or "const", norms=norms)
    taskify(module, tasks, norms=norms)
    if init is not None:
        init_norms(module, how=init)
    if not any(isinstance(m, TSNorm) for m in module.modules()):
        raise ValueError(
            "wrap() made nothing task-specific: no BatchNorm1d/2d or LayerNorm was found "
            "under this module (note it looks at children, so pass the model, not a bare "
            "norm), or `norms` excluded them all"
        )
    return module


# --- task dispatch --------------------------------------------------------------------


def set_active_task(module: nn.Module, task) -> nn.Module:
    """Point every `TSNorm` in the model at one task."""
    for norm in module.modules():
        if isinstance(norm, TSNorm):
            norm.active_task = str(task)
    return module


@contextmanager
def active_task(module: nn.Module, task):
    """Scope a forward pass to one task, restoring whatever was active before.

    >>> for task in model.tasks:
    ...     with active_task(model, task):
    ...         out[task] = model(x)
    """
    previous = [(n, n.active_task) for n in module.modules() if isinstance(n, TSNorm)]
    set_active_task(module, task)
    try:
        yield module
    finally:
        for norm, previous_task in previous:
            norm.active_task = previous_task


# --- parameter groups -----------------------------------------------------------------


def _parameters(module: nn.Module, of: Norms) -> List[nn.Parameter]:
    """Every parameter belonging to a submodule of one of the given types, deduplicated."""
    seen, params = set(), []
    for child in module.modules():
        if not isinstance(child, of):
            continue
        for param in child.parameters():
            if id(param) not in seen:
                seen.add(id(param))
                params.append(param)
    return params


def get_param_group(module: nn.Module, of: Norms) -> dict:
    """The parameters of every submodule of the given types, as an optimizer param group.

    The scalers want a much higher learning rate than the weights they modulate, which is
    what makes them allocate capacity before those weights move far (Algorithm 1):

    >>> optim = SGD([
    ...     {**get_param_group(model, SIG_BNS), "lr": lr * 100},
    ...     {**get_param_group(model, SIG_LNS), "lr": lr * 10},
    ...     {**get_other_param_group(model, excluding=SIG_NORMS), "lr": lr},
    ... ])
    """
    return {"params": _parameters(module, of)}


def get_other_param_group(module: nn.Module, excluding: Norms) -> dict:
    """Every parameter *not* covered by `get_param_group(module, excluding)`."""
    taken = {id(param) for param in _parameters(module, excluding)}
    return {"params": [p for p in module.parameters() if id(p) not in taken]}


# --- analysis -------------------------------------------------------------------------


def get_importance(module: nn.Module, per_layer: bool = False) -> Union[Tensor, List[Tensor]]:
    """The task-filter importance matrix `I[t, i] = sigmoid(gamma[t, i])`.

    Read straight off the trained scalers, at no cost beyond training. `[tasks, filters]`
    over the whole model, or one `[tasks, features]` matrix per layer.
    """
    layers = [n for n in module.modules() if isinstance(n, TSNorm)]
    if not layers:
        raise ValueError("no TSNorm layers found; did you taskify() the model?")
    if any(norm.tasks != layers[0].tasks for norm in layers):
        raise ValueError(
            "this model's TSNorm layers disagree on their tasks, so their scalers do not stack "
            "into one [tasks, filters] matrix — row t would mean a different task in different "
            f"layers. Taskify the whole model in one call. Found {layers[0].tasks} and "
            f"{next(n.tasks for n in layers if n.tasks != layers[0].tasks)}."
        )
    if any(not isinstance(norm, SIG_NORMS) for ts in layers for norm in ts.values()):
        raise TypeError("importance needs modulated norms: an unbounded scale is not an importance")
    if any(norm.weight.ndim != 1 for ts in layers for norm in ts.values()):
        raise ValueError(
            "importance is defined per feature, but this model normalizes over a "
            "multi-dimensional shape (e.g. nn.LayerNorm([4, 5])), so its scalers do not form "
            "a [tasks, filters] matrix"
        )

    importances = [norm.weights().sigmoid().detach() for norm in layers]
    return importances if per_layer else torch.cat(importances, dim=1)
