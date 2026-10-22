"""Train and evaluate σN-Ens with TorchUncertainty.

    python train.py fit --config configs/cifar10/resnet18.yaml --data.root ./data

From Python or a notebook, pass the same arguments as a list:

    from train import main
    main(["fit", "--config", "configs/cifar10/resnet18.yaml"])
"""

import torch
from torch_uncertainty import TULightningCLI

from tssnorm_tu import SigmaNEns


def main(args=None) -> TULightningCLI:
    torch.set_float32_matmul_precision("medium")
    cli = TULightningCLI(SigmaNEns, subclass_mode_data=True, args=args)
    if cli.subcommand == "fit" and cli._get(cli.config, "eval_after_fit"):
        # A fast_dev_run saves no checkpoint.
        best = None if cli.trainer.fast_dev_run else "best"
        cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path=best)
    return cli


if __name__ == "__main__":
    main()
