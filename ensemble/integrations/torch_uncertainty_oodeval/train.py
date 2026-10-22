"""Train and evaluate σN-Ens on a pretrained ViT with TorchUncertainty.

    python train.py fit --config configs/cifar100/vit_b16.yaml --data.root ./data

From Python or a notebook, pass the same arguments as a list:

    from train import main
    main(["fit", "--config", "configs/cifar100/vit_b16.yaml"])
"""

from typing import Optional, Union

import torch
from torch_uncertainty import TULightningCLI, TUTrainer

from tssnorm_tu import SigmaNEns


class Trainer(TUTrainer):
    """`TUTrainer`, whose `val_check_interval` also takes a number of steps. `TUTrainer` types it
    as a float, which would turn 2000 steps into 2000 epochs."""

    def __init__(self, *args, val_check_interval: Optional[Union[int, float]] = None, **kwargs):
        super().__init__(*args, val_check_interval=val_check_interval, **kwargs)


def main(args=None) -> TULightningCLI:
    torch.set_float32_matmul_precision("medium")
    cli = TULightningCLI(SigmaNEns, subclass_mode_data=True, trainer_class=Trainer, args=args)
    if cli.subcommand == "fit" and cli._get(cli.config, "eval_after_fit"):
        # A fast_dev_run saves no checkpoint.
        best = None if cli.trainer.fast_dev_run else "best"
        cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path=best)
    return cli


if __name__ == "__main__":
    main()
