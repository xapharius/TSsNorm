"""TSsBN for MLoRE's PascalContext benchmark.

A pretrained ViT-S backbone becomes task-specific through its normalization layers only:

- The patch embedding gets a LayerNorm, sigmoid-modulated for TSsBN and plain for TSBN.
- Every other backbone LayerNorm becomes task-specific but stays plain, because pretrained
  LayerNorm scales often exceed the sigmoid's range.
- A small multi-scale fusion module follows, whose BatchNorms become task-specific sigmoid
  BatchNorms.

Drop this file in as `MLoRE/models/TSsBN.py`. See this directory's README for the three
`common_config.py` edits that wire it in. Requires `tssnorm`, plus MLoRE's own dependencies
(`timm`, `einops`). Dropout and DropPath are disabled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from tssnorm import (
    SIG_BNS,
    SigLayerNorm,
    active_task,
    get_importance,
    get_other_param_group,
    get_param_group,
    taskify,
    wrap,
)


class TSsBNViT(nn.Module):
    """A pretrained ViT backbone + multi-scale fusion, made task-specific via normalization.

    `sigmoid=False` gives the plain TSBN baseline: task-specific norms without the sigmoid.
    """

    def __init__(self, backbone, heads, msf_hidden=None, msf_out=None, msf_depth=3,
                 img_size=512, select_layers=(3, 6, 9, 12), sigmoid=True):
        super().__init__()
        self.tasks = list(heads.keys())
        self.heads = nn.ModuleDict(heads)
        self.backbone = backbone
        self.img_size = img_size
        self.select_layers = list(select_layers)
        self.resolution = img_size // 16  # one token per 16x16 patch

        # Backbone: the patch embedding gets a norm, then every LN -> task-specific (taskify
        # matches SigLayerNorm too, since it subclasses nn.LayerNorm).
        self.backbone.patch_embed.norm = (SigLayerNorm(self.backbone.embed_dim) if sigmoid
                                          else nn.LayerNorm(self.backbone.embed_dim))
        taskify(self.backbone, self.tasks, norms=nn.LayerNorm)

        embed_dim = self.backbone.embed_dim
        self.msf = MultiScaleFusion(
            in_channels=embed_dim,
            hidden_channels=embed_dim if msf_hidden is None else msf_hidden,
            out_channels=embed_dim if msf_out is None else msf_out,
            n_scales=len(self.select_layers),
            depth=msf_depth,
        )
        # Fusion BatchNorms -> task-specific sigmoid-BN (const init), or plain per-task BN.
        if sigmoid:
            wrap(self.msf, self.tasks, convert=None, init="const", norms=nn.BatchNorm2d)
        else:
            taskify(self.msf, self.tasks, norms=nn.BatchNorm2d)

    def forward_features(self, x):
        """Backbone forward, collecting the selected layers as feature maps."""
        x = self.backbone.patch_embed(x)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed[:, 1:])
        feats = []
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i + 1 == len(self.backbone.blocks):
                x = self.backbone.norm(x)
            if i + 1 in self.select_layers:
                feats.append(x)
        return [rearrange(z, "b (h w) c -> b c h w", h=self.resolution, w=self.resolution)
                for z in feats]

    def forward_dec(self, z, task):
        z = F.interpolate(z, scale_factor=4, mode="bilinear")
        z = self.heads[task](z)
        return F.interpolate(z, (self.img_size, self.img_size), mode="bilinear")

    def forward(self, x, task=None):
        tasks = self.tasks if task is None else ([task] if isinstance(task, str) else task)
        out = {}
        for t in tasks:
            with active_task(self, t):          # points backbone + fusion norms at task t
                z = self.msf(self.forward_features(x))
            out[t] = self.forward_dec(z, t)
        return out

    def importance(self, per_layer=False):
        """The task-filter importance matrix of the fusion module's sigmoid-BN scalers.

        Only the sigmoid-modulated norms carry importances; the backbone's task-specific
        LayerNorms are un-modulated (unbounded) and are excluded.
        """
        return get_importance(self.msf, per_layer=per_layer)


class MultiScaleFusion(nn.Module):
    """Concat the selected scales, then 1x1 -> N x 3x3 (residual) -> 1x1, each Conv-BN-GELU."""

    def __init__(self, in_channels, hidden_channels, out_channels, n_scales=4, depth=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * n_scales, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.GELU(),
            ) for _ in range(depth)
        ])
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: list):
        x = self.conv1(torch.cat(x, dim=1))
        z = self.conv2(x)
        z = z + x
        return self.conv3(z)


def get_tssbn_optimizer(model, lr=2e-5, weight_decay=1e-6, lr_bn_mul=100, lr_ln_mul=10):
    """Adam with discriminative learning rates for the task-specific scalers.

    Sigmoid-BN scalers train at `lr * lr_bn_mul`, all LayerNorms (task-specific plain LNs and
    the sigmoid patch-embed LN) at `lr * lr_ln_mul`, both with weight decay off; everything
    else at `lr`. "Everything else" is the complement of the norm groups, so ViT parameters
    that live outside a Conv or Linear, such as `pos_embed` and `cls_token`, are included.
    """
    groups = [
        {**get_param_group(model, SIG_BNS), "lr": lr * lr_bn_mul, "weight_decay": 0.0},
        {**get_param_group(model, nn.LayerNorm), "lr": lr * lr_ln_mul, "weight_decay": 0.0},
        {**get_other_param_group(model, excluding=SIG_BNS + (nn.LayerNorm,)),
         "lr": lr, "weight_decay": weight_decay},
    ]
    groups = [g for g in groups if g["params"]]
    return torch.optim.Adam(groups)
