"""σN-Ens on pretrained ViTs, in TorchUncertainty's OodEval branch.

`SigmaNEns` is a TorchUncertainty classification routine. It converts a pretrained timm ViT
into an ensemble, adds the softmax regulariser to the loss, and gives the scalers their own
learning rate. `ViTCIFAR100` and `ViTImageNet` feed it images at the ViT's resolution, and test
on the OpenOOD near and far sets. Run it from the command line with `train.py`, or use it in
Python:

>>> from functools import partial
>>> routine = SigmaNEns(num_classes=100, reg_on=["blocks.*.norm2"],
...                     optimizer=partial(torch.optim.SGD, lr=0.01, momentum=0.9))
>>> TUTrainer(max_steps=10_000).fit(routine, ViTCIFAR100(root="./data", batch_size=256))
"""

from typing import List, Optional

import timm
import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import nn
from torchvision.datasets import ImageNet
from torchvision.transforms import v2
from torch_uncertainty.datamodules import CIFAR100DataModule, ImageNetDataModule
from torch_uncertainty.datasets.ood.utils import get_ood_datasets
from torch_uncertainty.routines import ClassificationRoutine
from torch_uncertainty.transforms import RepeatTarget

from tssnorm import (SIG_NORMS, SigLayerNorm, forward_tasks, get_mean_similarity, get_norms,
                     make_heads, optim_factory, register_norm, sigmoidify, softmax_loss, wrap)

register_norm(timm.layers.LayerNorm, SigLayerNorm)      # nn.LayerNorm with a fast path


def backbone(arch: str, num_classes: int) -> nn.Module:
    """A pretrained timm ViT, such as `vit_base_patch16_224.orig_in21k`."""
    return timm.create_model(arch, pretrained=True, num_classes=num_classes)


class Ensemble(nn.Module):
    """Runs every member and concatenates their logits into `[members * batch, classes]`,
    the layout TorchUncertainty expects."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return forward_tasks(self.model, x, stack=True).flatten(0, 1)


def sigma_n_ens(arch: str, num_classes: int, num_estimators: int = 4) -> Ensemble:
    """A pretrained ViT, converted into a σN-Ens.

    LayerNorm scales exceed 1, which `scale+bias` keeps. The heads' norms are new, with scales
    of 1, which `clip+bias` maps to 0.95.
    """
    model = backbone(arch, num_classes)
    model.head = make_heads(model.head.in_features, num_classes, tasks=num_estimators,
                            norm=nn.BatchNorm1d)
    sigmoidify(model.head, convert="clip+bias")
    wrap(model, convert="scale+bias")
    return Ensemble(model)


class SigmaNEns(ClassificationRoutine):
    """σN-Ens on a pretrained ViT, as a TorchUncertainty classification routine.

    The scalers and the heads' norms train at `lr_mul` times the base learning rate. The
    learning rate scheduler steps after every training step.

    Args:
        num_classes: number of classes.
        arch: a timm ViT, which starts pretrained.
        num_estimators: number of members.
        tau: temperature of the regulariser. A lower `tau` makes the members more diverse.
        reg_weight: weight of the regulariser in the loss.
        reg_on: name patterns of the regularised norms, such as `["blocks.*.norm2"]`. `None`
            regularises every norm.
        lr_mul: learning rate of the scalers, relative to the rest.
        optimizer: builds the optimizer from the parameter groups.
        lr_scheduler: builds the learning rate scheduler from the optimizer.
        eval_ood: evaluate out-of-distribution detection.
        eval_shift: evaluate under distribution shift.
    """

    def __init__(
        self,
        num_classes: int,
        arch: str = "vit_base_patch16_224.orig_in21k",
        num_estimators: int = 4,
        tau: float = 0.5,
        reg_weight: float = 1e-3,
        reg_on: Optional[List[str]] = None,
        lr_mul: float = 100.0,
        optimizer: OptimizerCallable = torch.optim.SGD,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
        eval_ood: bool = False,
        eval_shift: bool = False,
    ):
        super().__init__(
            model=sigma_n_ens(arch, num_classes, num_estimators),
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
        optimizer = optim_factory(self.model, self.optimizer, of=SIG_NORMS, lr_mul=self.lr_mul)
        if self.lr_scheduler is None:
            return optimizer
        scheduler = {"scheduler": self.lr_scheduler(optimizer), "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


BICUBIC = dict(interpolation=v2.InterpolationMode.BICUBIC, antialias=True)


def normalize() -> List[nn.Module]:
    """Image tensor to floats, normalised with ImageNet's statistics."""
    return [v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]


class ViTCIFAR100(CIFAR100DataModule):
    """CIFAR-100 for a pretrained ViT: bicubic resize to `image_size`, random crop and flip for
    training."""

    def __init__(self, root: str, batch_size: int, image_size: int = 224, **kwargs):
        train = v2.Compose([v2.ToImage(), v2.Resize(image_size, **BICUBIC),
                            v2.RandomCrop(image_size, padding=image_size // 8),
                            v2.RandomHorizontalFlip(), *normalize()])
        test = v2.Compose([v2.ToImage(), v2.Resize((image_size, image_size), **BICUBIC),
                           *normalize()])
        super().__init__(root, batch_size, train_transform=train, test_transform=test, **kwargs)


class ViTImageNet(ImageNetDataModule):
    """ImageNet for a pretrained ViT, from a local folder with `train/`, `val/` and the devkit.

    Training uses random resized crops and flips. Evaluation uses the official validation
    images, resized and center-cropped. The OpenOOD sets download on the first test.
    """

    def __init__(self, root: str, batch_size: int, image_size: int = 224, **kwargs):
        train = v2.Compose([v2.ToImage(),
                            v2.RandomResizedCrop(image_size, scale=(0.08, 1.0), **BICUBIC),
                            v2.RandomHorizontalFlip(), *normalize()])
        test = v2.Compose([v2.ToImage(), v2.Resize(image_size * 8 // 7, **BICUBIC),
                           v2.CenterCrop(image_size), *normalize()])
        super().__init__(root, batch_size, train_transform=train, test_transform=test, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train = ImageNet(self.root, split="train", transform=self.train_transform)
            self.val = ImageNet(self.root, split="val", transform=self.test_transform)
        if stage in ("test", None):
            self.test = ImageNet(self.root, split="val", transform=self.test_transform)
            if self.eval_ood:
                self.test_ood, self.val_ood, near, far = get_ood_datasets(
                    root=self.root, dataset_id="imagenet1k", transform=self.test_transform)
                self.near_oods, self.far_oods = list(near.values()), list(far.values())
                for ds in [self.val_ood, *self.near_oods, *self.far_oods]:
                    ds.dataset_name = getattr(ds, "dataset_name", type(ds).__name__.lower())
                self.near_ood_names = [ds.dataset_name for ds in self.near_oods]
                self.far_ood_names = [ds.dataset_name for ds in self.far_oods]

    def train_dataloader(self):
        return self._data_loader(self.train, training=True, shuffle=True)
