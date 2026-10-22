"""Regularisers on the task-specific scalers."""

from typing import Iterable, Union

from torch import Tensor, nn
from torch.nn import functional as F

from .norms import TSNorm
from .utils import _scalers, get_norms

__all__ = ["softmax_loss"]


def softmax_loss(norms: Union[nn.Module, Iterable[TSNorm]], tau: float = 1.0) -> Tensor:
    """Concentrate each feature on fewer tasks.

    For every feature, this takes a softmax over the tasks' importances at temperature `tau`
    and sums the log probabilities. Minimising it raises the importance of the tasks that
    already lead on a feature and lowers the rest. A low `tau` drives each feature towards a
    single task. A high `tau` lets the tasks share it.

    `norms` is a model, or the layers to regularise:

    >>> loss = task_loss + 1e-2 * softmax_loss(get_norms(model, "layer*.bn2"), tau=0.1)
    """
    norms = get_norms(norms) if isinstance(norms, nn.Module) else list(norms)
    return sum(F.log_softmax(I / tau, dim=0).sum() for I in _scalers(norms))
