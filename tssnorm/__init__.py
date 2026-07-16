"""Task-Specific Sigmoid Normalization for multi-task learning."""

from .norms import (
    SigBatchNorm1d,
    SigBatchNorm2d,
    SigLayerNorm,
    TSNorm,
    convert_weights,
    init_weight,
)
from .utils import (
    NORMS,
    SIG_BNS,
    SIG_LNS,
    SIG_NORMS,
    active_task,
    get_importance,
    get_other_param_group,
    get_param_group,
    init_norms,
    register_norm,
    set_active_task,
    sigmoidify,
    taskify,
    wrap,
)

__all__ = [
    "NORMS",
    "SIG_BNS",
    "SIG_LNS",
    "SIG_NORMS",
    "SigBatchNorm1d",
    "SigBatchNorm2d",
    "SigLayerNorm",
    "TSNorm",
    "active_task",
    "convert_weights",
    "get_importance",
    "get_other_param_group",
    "get_param_group",
    "init_norms",
    "init_weight",
    "register_norm",
    "set_active_task",
    "sigmoidify",
    "taskify",
    "wrap",
]

__version__ = "1.0.0"
