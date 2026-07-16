"""Reproduce Table 2 (NYUv2) — TSsBN in LibMTL.

A standalone LibMTL training script: a sibling to LibMTL's own `examples/nyu/main.py`, with
TSsBN wired in from the start. It modifies no LibMTL file — LibMTL is a library you write a
training script against, and this is one. Drop it into `LibMTL/examples/nyu/` (so its
`utils`, `aspp`, `create_dataset` resolve) and run:

    python reproduce_nyuv2.py --arch TSsBN --dataset_path PATH/nyuv2 --gpu_id 0 \
        --weighting EW --scheduler step --mode train --save_path ./tssbn \
        --epochs 200 --train_bs 8 --test_bs 8

`--arch TSBN` runs the baseline. `--dry-run` builds the model and optimizer and exits (no
dataset needed) — a fast sanity check before committing to the full run. Requires `tssnorm`
and `tssnorm_libmtl.py` on the path, and LibMTL GitHub main.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import SegMetric, SegLoss, DepthMetric, DepthLoss, NormalMetric, NormalLoss
from aspp import DeepLabHead
from create_dataset import NYUv2

from LibMTL.model import resnet_dilated
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args

from tssnorm_libmtl import register, TSsBNTrainer

register()   # make architecture="TSsBN" / "TSBN" resolvable


def build_args(params):
    """LibMTL's kwargs / optim_param / scheduler_param, without `prepare_args`' architecture
    whitelist (which upstream hardcodes and does not include TSsBN). Covers the EW + Adam/SGD +
    step/cos/exp setup this script uses."""
    kwargs = {"weight_args": {}, "arch_args": {}}
    optim_param = {"optim": params.optim, "lr": params.lr, "weight_decay": params.weight_decay}
    if params.optim == "sgd":
        optim_param["momentum"] = params.momentum
    scheduler_param = None
    if params.scheduler is not None:
        scheduler_param = {"scheduler": params.scheduler}
        if params.scheduler == "step":
            scheduler_param.update(step_size=params.step_size, gamma=params.gamma)
    return kwargs, optim_param, scheduler_param


def parse_args(parser):
    parser.add_argument("--train_bs", default=8, type=int, help="training batch size (LibMTL NYUv2 default)")
    parser.add_argument("--test_bs", default=8, type=int, help="test batch size")
    parser.add_argument("--epochs", default=200, type=int, help="training epochs")
    parser.add_argument("--dataset_path", default="/", type=str, help="NYUv2 path")
    parser.add_argument("--aug", action="store_true", default=False, help="data augmentation")
    parser.add_argument("--bn_lr_mul", default=100, type=float, help="alpha_sBN (scaler LR multiplier)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="build model + optimizer and exit; no dataset needed")
    return parser.parse_args()


def build_task_dict():
    return {
        "segmentation": {"metrics": ["mIoU", "pixAcc"], "metrics_fn": SegMetric(num_classes=13),
                         "loss_fn": SegLoss(), "weight": [1, 1]},
        "depth": {"metrics": ["abs_err", "rel_err"], "metrics_fn": DepthMetric(),
                  "loss_fn": DepthLoss(), "weight": [0, 0]},
        "normal": {"metrics": ["mean", "median", "<11.25", "<22.5", "<30"],
                   "metrics_fn": NormalMetric(), "loss_fn": NormalLoss(), "weight": [0, 0, 1, 1, 1]},
    }


class NYUtrainer(TSsBNTrainer):
    """LibMTL's NYUv2 trainer (the example's `process_preds`), on `TSsBNTrainer`."""

    def process_preds(self, preds):
        for task in self.task_name:
            preds[task] = F.interpolate(preds[task], (288, 384), mode="bilinear", align_corners=True)
        return preds


def main(params):
    kwargs, optim_param, scheduler_param = build_args(params)
    if params.arch == "TSsBN":
        kwargs["arch_args"]["convert"] = "clip+bias"      # represent pretrained ResNet-50 scales (App. F.3)

    task_dict = build_task_dict()

    def encoder_class():
        return resnet_dilated("resnet50")                  # pretrained ImageNet backbone

    num_out = {"segmentation": 13, "depth": 1, "normal": 3}
    decoders = nn.ModuleDict({t: DeepLabHead(2048, num_out[t]) for t in task_dict})

    model = NYUtrainer(
        task_dict=task_dict, weighting=params.weighting, architecture=params.arch,
        encoder_class=encoder_class, decoders=decoders, rep_grad=params.rep_grad,
        multi_input=params.multi_input, optim_param=optim_param, scheduler_param=scheduler_param,
        save_path=params.save_path, load_path=params.load_path,
        bn_lr_mul=params.bn_lr_mul, device="cpu" if params.dry_run else "cuda:0",
        **kwargs,
    )

    if params.dry_run:
        print("dry run — model and optimizer built, no training.")
        for i, g in enumerate(model.optimizer.param_groups):
            print(f"  group {i}: lr={g['lr']:g}  wd={g['weight_decay']:g}  "
                  f"params={sum(p.numel() for p in g['params']):,}")
        if params.arch == "TSsBN":
            I = model.model.importance()
            print(f"  importance matrix {tuple(I.shape)} = [tasks, filters]")
        return

    train_set = NYUv2(root=params.dataset_path, mode="train", augmentation=params.aug)
    test_set = NYUv2(root=params.dataset_path, mode="test", augmentation=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=params.train_bs, shuffle=True,
                                               num_workers=2, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=params.test_bs, shuffle=False,
                                              num_workers=2, pin_memory=True)
    if params.mode == "train":
        model.train(train_loader, test_loader, params.epochs)
    elif params.mode == "test":
        model.test(test_loader)
    else:
        raise ValueError(f"unknown mode {params.mode}")


if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    if not params.dry_run:
        set_device(params.gpu_id)
    set_random_seed(params.seed)
    main(params)
