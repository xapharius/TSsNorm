"""
Model surgery, task dispatch, parameter groups, and readouts.

The method is three steps, applied to any model that already uses normalization layers:

    sigmoidify(model)                # BN/LN  ->  modulated sigma-BN/LN
    taskify(model, tasks)            # each norm  ->  one copy per task
    init_norms(model, how="const")   # initialise the scalers

`wrap` runs all three. Where the norms of a model need different treatment, the steps compose
per submodule.

Example
-------
>>> from tssnorm import wrap, forward_tasks, optim_factory, get_importance
>>>
>>> wrap(model.encoder, tasks=["seg", "depth", "normals"])
>>> optim = optim_factory(model, lr=1e-2, momentum=0.9)    # scalers at 100x the base rate
>>>
>>> out = forward_tasks(model, x)                           # {task: output}
>>> I = get_importance(model.encoder)                       # [tasks, filters]
"""

from contextlib import contextmanager
from fnmatch import fnmatch
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _NormBase

from .norms import (
    SigBatchNorm1d,
    SigBatchNorm2d,
    SigLayerNorm,
    TSModule,
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
    "forward_tasks",
    "get_importance",
    "get_mean_similarity",
    "get_norms",
    "get_optimizer",
    "get_other_param_group",
    "get_param_group",
    "get_similarity",
    "init_norms",
    "make_heads",
    "optim_factory",
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
Tasks = Union[int, str, Iterable]


def register_norm(norm_cls: type, sig_cls: type) -> None:
    """Teach `sigmoidify` to modulate `norm_cls` by replacing it with `sig_cls`.

    Dispatch is by exact type, because a subclass may fold extra behaviour into its forward
    that a swap would drop. timm's `BatchNormAct2d` fuses an activation, and its `LayerNorm2d`
    permutes axes. Register a subclass whose forward *is* equivalent to its base, and it will
    be modulated like the base:

    >>> import timm.layers
    >>> register_norm(timm.layers.LayerNorm, SigLayerNorm)   # a fast-path alias for nn.LayerNorm
    """
    _SIGMOID[norm_cls] = sig_cls


def _task_names(tasks: Tasks) -> List[str]:
    """Task names as strings. A number `n` names `n` anonymous tasks, `"0"` to `"n-1"`, such
    as ensemble members."""
    if isinstance(tasks, int):
        return [str(t) for t in range(tasks)]
    return [str(t) for t in ([tasks] if isinstance(tasks, str) else tasks)]


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

    A norm subclass may fold extra behaviour into its forward that replacing it would drop.
    timm's `BatchNormAct2d` fuses an activation, and its `LayerNorm2d` permutes axes. So a
    subclass is not assumed safe. Checking up front means a refused model is left exactly as it came in, and
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
                    "would drop, so it is not assumed safe. Call register_norm() if it is, "
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
    normalization layer, so one this cannot modulate is reported rather than skipped, which
    would leave part of the model unmodulated. Narrow `norms` to treat parts of a model
    differently, and the rest are then left alone deliberately. See `register_norm` to
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


def taskify(module: nn.Module, tasks: Tasks, norms: Optional[Norms] = None) -> nn.Module:
    """Give each task its own copy of every selected normalization layer, in place.

    `norms` defaults to every normalization layer; narrow it to duplicate only some. Modules
    that are already task-specific, such as heads from `make_heads`, are left alone.
    """
    norms = _ALL_NORMS if norms is None else norms
    tasks = _task_names(tasks)
    duplicated: dict = {}                         # id(old) -> TSNorm, for norms reused under two names

    def walk(parent: nn.Module) -> None:
        for name, child in _children(parent):
            if isinstance(child, TSModule):
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
    """(Re)initialise every task's scalers. Run after `taskify`, so a stochastic init draws
    per task."""
    for ts in get_norms(module):
        for norm in ts.values():
            if isinstance(norm, SIG_NORMS):
                init_weight(norm.weight, how)
    return module


def wrap(
    module: nn.Module,
    tasks: Optional[Tasks] = None,
    convert: Optional[str] = None,
    init: Union[str, None, "_Default"] = _DEFAULT,
    norms: Optional[Norms] = None,
) -> nn.Module:
    """Modulate the norms, give each task its own copy, and initialise them.

    `tasks` lists the task names, or gives their number: `wrap(model, 4)` makes the tasks
    `"0"` to `"3"`, as for the members of an ensemble. Leave it out when the model already
    has task-specific heads, such as from `make_heads`; `wrap` then uses their tasks.

    `init` runs last, so a stochastic init gives every task an independent draw. By default it
    is `"const"` for a from-scratch wrap, and `None` when `convert` is given, because initialising
    would otherwise overwrite the scalers `convert` just inherited. Pass `init` explicitly to
    force either behaviour. `norms` defaults to every normalization layer (see `sigmoidify`).
    """
    found = {tuple(m.tasks) for m in module.modules() if isinstance(m, TSModule)}
    if tasks is None:
        if not found:
            raise ValueError("wrap() needs `tasks`, or task-specific heads in the model to "
                             "take them from (see make_heads)")
        tasks = list(next(iter(found)))
    tasks = _task_names(tasks)
    if any(set(f) != set(tasks) for f in found):
        raise ValueError(
            f"this model already has task-specific modules for {sorted(map(list, found))}, "
            f"which disagree with {tasks}. Build a fresh model for a new task set."
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


def make_heads(in_features: int, out_features: Union[int, Dict], *, tasks: Tasks,
               norm: Optional[Type[nn.Module]] = SigBatchNorm1d) -> TSModule:
    """One head per task: a norm, then a linear layer.

    `tasks` lists the task names, or gives their number. `out_features` is an int, or a dict
    with one entry per task. Each head is built from scratch, so the tasks start from
    independent weights. Pass `norm=None` for a plain linear head. Assign the result where the
    model's classifier was, then `wrap` the model:

    >>> model.fc = make_heads(model.fc.in_features, 10, tasks=4)    # one head per member
    >>> wrap(model)                                                 # one norm per member
    """
    def head(task):
        out = out_features[task] if isinstance(out_features, dict) else out_features
        linear = nn.Linear(in_features, out)
        return linear if norm is None else nn.Sequential(norm(in_features), linear)

    return TSModule({task: head(task) for task in _task_names(tasks)})


# --- task dispatch --------------------------------------------------------------------


def set_active_task(module: nn.Module, task) -> nn.Module:
    """Point every task-specific module in the model at one task."""
    for m in module.modules():
        if isinstance(m, TSModule):
            m.active_task = str(task)
    return module


@contextmanager
def active_task(module: nn.Module, task):
    """Scope a forward pass to one task, restoring whatever was active before.

    >>> for task in model.tasks:
    ...     with active_task(model, task):
    ...         out[task] = model(x)
    """
    previous = [(m, m.active_task) for m in module.modules() if isinstance(m, TSModule)]
    set_active_task(module, task)
    try:
        yield module
    finally:
        for m, previous_task in previous:
            m.active_task = previous_task


def forward_tasks(module: nn.Module, x, tasks: Optional[Tasks] = None,
                  stack: bool = False) -> Union[dict, Tensor]:
    """Run the model once per task. Returns `{task: output}`, or with `stack=True` one
    `[tasks, ...]` tensor, such as the members' logits of an ensemble.

    `tasks` defaults to the model's tasks.
    """
    if tasks is None:
        ts = next((m for m in module.modules() if isinstance(m, TSModule)), None)
        if ts is None:
            raise ValueError("no task-specific modules found; did you wrap() the model?")
        tasks = ts.tasks
    out = {}
    for task in _task_names(tasks):
        with active_task(module, task):
            out[task] = module(x)
    return torch.stack(list(out.values())) if stack else out


# --- parameter groups and optimisers ----------------------------------------------------


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


def get_optimizer(groups: Iterable, optim=torch.optim.SGD, **kwargs):
    """Build `optim` over param groups. `kwargs` go to `optim`, the learning rate included.

    A group may carry `lr_mul`, which multiplies the learning rate it gets. Other group keys
    (`weight_decay`, ...) override `kwargs` as usual. Plain parameters form a single group.
    `optim` is anything that builds an optimizer from param groups: a class, or a partial.

    >>> get_optimizer([{**get_param_group(model, SIG_BNS), "lr_mul": 100},
    ...                get_other_param_group(model, excluding=SIG_BNS)], lr=1e-2)
    """
    groups = list(groups)
    if not all(isinstance(g, dict) for g in groups):
        groups = [{"params": groups}]
    multipliers = [g.get("lr_mul", 1.0) for g in groups]
    optimizer = optim([{k: v for k, v in g.items() if k != "lr_mul"} for g in groups], **kwargs)
    for group, multiplier in zip(optimizer.param_groups, multipliers):
        group["lr"] *= multiplier
    return optimizer


def optim_factory(module: nn.Module, optim=torch.optim.SGD, of: Norms = TSNorm,
                  lr_mul: float = 100.0, scaler_weight_decay: float = 0.0, **kwargs):
    """`optim` with the scalers at `lr_mul` times the learning rate of everything else.

    The scalers are the parameters of `of` layers. The default, `TSNorm`, takes every
    task-specific norm of the backbone, but not the norms inside heads from `make_heads`. Pass
    `of=SIG_NORMS` to include those. The scalers train without weight decay unless
    `scaler_weight_decay` says otherwise. `kwargs` go to
    `optim` and apply to the rest.

    >>> optim_factory(model, lr=0.1, momentum=0.9, weight_decay=5e-4)
    """
    groups = [
        {**get_param_group(module, of), "lr_mul": lr_mul, "weight_decay": scaler_weight_decay},
        get_other_param_group(module, excluding=of),
    ]
    return get_optimizer(groups, optim, **kwargs)


# --- readouts -------------------------------------------------------------------------


def get_norms(module: nn.Module, *patterns: str) -> List[TSNorm]:
    """The model's task-specific norms, optionally only those whose names match a pattern.

    Patterns are shell-style, matched against the module names of `named_modules()`:

    >>> get_norms(resnet, "layer*.bn2")      # the second norm of every residual block
    """
    return [m for name, m in module.named_modules() if isinstance(m, TSNorm)
            and (not patterns or any(fnmatch(name, p) for p in patterns))]


def _scalers(norms: List[TSNorm]) -> List[Tensor]:
    """`sigmoid(gamma)` of each layer, `[tasks, features]`, checked to be comparable."""
    if not norms:
        raise ValueError("no TSNorm layers found; did you taskify() the model?")
    if any(norm.tasks != norms[0].tasks for norm in norms):
        raise ValueError(
            "this model's TSNorm layers disagree on their tasks, so their scalers do not stack "
            "into one [tasks, filters] matrix: row t would mean a different task in different "
            f"layers. Taskify the whole model in one call. Found {norms[0].tasks} and "
            f"{next(n.tasks for n in norms if n.tasks != norms[0].tasks)}."
        )
    if any(not isinstance(norm, SIG_NORMS) for ts in norms for norm in ts.values()):
        raise TypeError("importance needs modulated norms: an unbounded scale is not an importance")
    if any(norm.weight.ndim != 1 for ts in norms for norm in ts.values()):
        raise ValueError(
            "importance is defined per feature, but this model normalizes over a "
            "multi-dimensional shape (e.g. nn.LayerNorm([4, 5])), so its scalers do not form "
            "a [tasks, filters] matrix"
        )
    return [norm.weights().sigmoid() for norm in norms]


def get_importance(module: nn.Module, per_layer: bool = False) -> Union[Tensor, List[Tensor]]:
    """The task-filter importance matrix `I[t, i] = sigmoid(gamma[t, i])`.

    Read from the trained scalers. `[tasks, filters]` over the whole model, or one
    `[tasks, features]` matrix per layer.
    """
    importances = [I.detach() for I in _scalers(get_norms(module))]
    return importances if per_layer else torch.cat(importances, dim=1)


def get_similarity(module: nn.Module, per_layer: bool = False) -> Union[Tensor, List[Tensor]]:
    """Cosine similarity between the tasks' importance vectors.

    `[tasks, tasks]` over the whole model, or one such matrix per layer. Averaging the
    per-layer matrices weighs every layer equally, however wide it is.
    """
    def cosine(I):
        I = F.normalize(I, dim=1)
        return I @ I.T

    importances = get_importance(module, per_layer=True)
    return [cosine(I) for I in importances] if per_layer else cosine(torch.cat(importances, 1))


def get_mean_similarity(module: nn.Module) -> float:
    """How alike the tasks are, in one number: the mean cosine between different tasks'
    importances, averaged over layers.

    It is 1 when every task weighs every feature alike, and lower as the tasks diverge. For an
    ensemble, it measures the members' diversity without data.
    """
    S = torch.stack(get_similarity(module, per_layer=True)).mean(0)
    if len(S) < 2:
        raise ValueError("similarity needs at least two tasks")
    return ((S.sum() - S.trace()) / (len(S) * (len(S) - 1))).item()
