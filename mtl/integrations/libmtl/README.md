# TSsBN in LibMTL

Runs TSsBN in [LibMTL](https://github.com/median-research-group/LibMTL). This is the setup of
Table 2: DeepLabV3 with a pretrained ResNet-50, on NYUv2 with 3 tasks and Cityscapes with 2.

LibMTL is used as a library. None of its files change. `register()` adds the `TSsBN` and `TSBN`
architectures to LibMTL, which then finds them by name. `TSsBNTrainer` is a LibMTL `Trainer`
that gives the scalers their higher learning rate.

## Install

From this folder:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e ../../..             # tssnorm, from the root of this repo
```

This installs LibMTL from GitHub, pinned to one commit of `main`. The PyPI release, 1.1.5, is
older and cannot find architectures by name. `tssnorm_libmtl.py` is in this folder. Put it on
your path.

## Usage

```python
from tssnorm_libmtl import TSsBNTrainer, register

register()

trainer = TSsBNTrainer(
    task_dict=task_dict,
    weighting="EW",
    architecture="TSsBN",
    encoder_class=encoder_class,               # any backbone with BatchNorm
    decoders=decoders,
    rep_grad=False, multi_input=False,
    optim_param={"optim": "adam", "lr": 1e-4},
    scheduler_param=None,
    arch_args={"convert": "clip+bias"},        # keep the pretrained scales
    bn_lr_mul=100,                             # the scalers' learning rate, relative to lr
    device="cuda:0",
)
trainer.train(train_loader, test_loader, epochs)

I = trainer.model.importance()                 # [tasks, filters]
```

`arch_args` go to `tssnorm.wrap`. `convert="clip+bias"` keeps the pretrained scales and
shifts. For an encoder trained from scratch, pass `{"convert": None, "init": "const"}`.

The scalers train at `lr * bn_lr_mul`, without weight decay. LibMTL's own `Trainer` always uses
`cuda:0`. `TSsBNTrainer` uses the `device` you pass, so it also runs on a CPU.

The `TSBN` baseline gives each task its own BatchNorm, without the sigmoid. It has no scalers,
so every parameter trains at `lr`:

```python
trainer = TSsBNTrainer(..., architecture="TSBN", arch_args={})
```

## Reproduce Table 2 on NYUv2

[`reproduce_nyuv2.py`](reproduce_nyuv2.py) is a LibMTL training script, like LibMTL's own
`examples/nyu/main.py`. It reuses that example's data and decoder code, so it runs from there:

```bash
git clone https://github.com/median-research-group/LibMTL.git
git -C LibMTL checkout 4336804      # the commit requirements.txt installs
cp reproduce_nyuv2.py tssnorm_libmtl.py LibMTL/examples/nyu/
cd LibMTL/examples/nyu
```

Check the setup without data. `--dry-run` builds the model and the optimizer, prints the
learning rate groups and the shape of the importance matrix, and exits:

```bash
python reproduce_nyuv2.py --arch TSsBN --weighting EW --scheduler step --mode train --dry-run
```

Download NYUv2 as LibMTL's `examples/nyu/README.md` describes, then train:

```bash
python reproduce_nyuv2.py --arch TSsBN --weighting EW \
    --dataset_path PATH/nyuv2 --gpu_id 0 --scheduler step --mode train \
    --save_path ./tssbn --epochs 200 --train_bs 8 --test_bs 8
```

The defaults are the paper's: Adam at `1e-4`, weight decay `1e-5`, the learning rate halved at
epoch 100, 200 epochs and batch size 8. `--arch TSBN` runs the baseline.

The script covers NYUv2 only. For Cityscapes, make the same changes to LibMTL's Cityscapes
example: call `register()`, subclass `TSsBNTrainer`, set `arch_args`, and use batch size 16.
