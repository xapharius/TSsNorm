# TSsBN for LibMTL

Run **Task-Specific Sigmoid BatchNorm** inside
[LibMTL](https://github.com/median-research-group/LibMTL). This is the setup behind **Table 2**
(DeepLabV3 + pretrained ResNet-50 on NYUv2 3-task / Cityscapes 2-task).

**Zero edits to LibMTL.** `register()` injects the `TSsBN` and `TSBN` architectures into
`LibMTL.architecture` so LibMTL resolves them by name, and `TSsBNTrainer` (a `Trainer`
subclass) gives the task-specific scalers their higher learning rate (Algorithm 1) — it
overrides only `_prepare_optimizer` and the device.

## Which LibMTL

Target **LibMTL GitHub main**, not the PyPI `1.1.5` release. Main is what LibMTL's own
`examples/` are written for: it resolves the architecture by name (so `register()` works) and
runs on modern PyTorch. The PyPI release is older, class-based, and can't run those examples.

## Install

```bash
pip install "git+https://github.com/median-research-group/LibMTL@4336804"   # main, pinned
pip install -e ../..                                                       # this repo, for `tssnorm`
```

LibMTL main's trainer/weightings also need `scipy`, `cvxpy`, `qpsolvers`, `torch_geometric`
(it imports them at load time). Then put `tssnorm_libmtl.py` on your path (it lives here).

## Usage

```python
from tssnorm_libmtl import TSsBNTrainer, register
register()                                       # make architecture="TSsBN" resolvable

trainer = TSsBNTrainer(
    task_dict=task_dict,                         # your usual LibMTL task_dict
    weighting="EW",
    architecture="TSsBN",
    encoder_class=encoder_class,                 # any backbone that uses BatchNorm
    decoders=decoders,
    rep_grad=False, multi_input=False,
    optim_param={"optim": "adam", "lr": 1e-4},
    scheduler_param=None,
    arch_args={"convert": "clip+bias"},          # pretrained backbone (Table 2)
    bn_lr_mul=100,                               # alpha_sBN
    device="cuda:0",                             # or "cpu"
)
trainer.train(train_loader, test_loader, epochs)

I = trainer.model.importance()                   # [tasks, filters] importance matrix
```

For a data-free tour of the `tssnorm` API itself, see the repo's `examples/basic_usage.ipynb`.

### `arch_args`

Forwarded to `tssnorm.wrap`:

| key | default | meaning |
|-----|---------|---------|
| `convert` | `"clip+bias"` | how to inherit a pretrained norm's scale. `"clip+bias"` represents it with a sigmoid and copies the frozen bias (App. F.3); `"clip"` omits the bias; `None` discards the pretrained weights. |
| `init` | `None` | scaler init when not converting. Use `convert=None, init="const"` for a **from-scratch** encoder (trains from `sigmoid(0)=0.5`). |

### `TSsBNTrainer` arguments

| arg | default | meaning |
|-----|---------|---------|
| `bn_lr_mul` | `100` | learning-rate multiplier for the sigmoid-BN scalers (`alpha_sBN`) |
| `device` | `"cuda:0"` | device to train on. Upstream `Trainer` hardcodes `cuda:0`; this respects the argument, so `"cpu"` works. |

The scalers train at `lr * bn_lr_mul` with weight decay off; everything else at `lr`.

## The `TSBN` baseline

`TSBN` — per-task BatchNorm without the sigmoid modulation — is the paper's minimal baseline.
It has no scalers to accelerate, so run it through `TSsBNTrainer` too: the multiplier groups
come out empty and every parameter trains at the base learning rate.

```python
register()
trainer = TSsBNTrainer(..., architecture="TSBN", arch_args={})
```

## Reproducing Table 2 (NYUv2)

[`reproduce_nyuv2.py`](reproduce_nyuv2.py) is a standalone LibMTL training script — a sibling
to LibMTL's own `examples/nyu/main.py`, with TSsBN wired in from the start. **It modifies no
LibMTL file**: LibMTL is a library you write a training script against, and this is one.

**1. Clone LibMTL and drop the script into the NYUv2 example.** It reuses the example's
data/decoder helpers (`utils`, `aspp`, `create_dataset`), so it runs from `examples/nyu/`:

```bash
git clone https://github.com/median-research-group/LibMTL.git
cp reproduce_nyuv2.py tssnorm_libmtl.py  LibMTL/examples/nyu/
cd LibMTL/examples/nyu
```

**2. Sanity-check — no dataset needed.** `--dry-run` builds the real DeepLabV3 + pretrained
ResNet-50 model and the optimizer, prints the discriminative-LR groups and the importance
matrix, and exits:

```bash
python reproduce_nyuv2.py --arch TSsBN --weighting EW --scheduler step --mode train --dry-run
```

**3. Get NYUv2 and train.** Download NYUv2 per LibMTL's `examples/nyu/README.md`, then:

```bash
python reproduce_nyuv2.py --arch TSsBN --weighting EW \
    --dataset_path PATH/nyuv2 --gpu_id 0 --scheduler step --mode train \
    --save_path ./tssbn --epochs 200 --train_bs 8 --test_bs 8
```

- **TSBN baseline:** `--arch TSBN` (trains at the base learning rate — no scaler acceleration).
- **Cityscapes** (the 2-task half of Table 2): the same script pattern on LibMTL's Cityscapes
  example, with `--train_bs 16`.
- **Config:** Adam `lr=1e-4`, weight decay `1e-5`, StepLR `gamma=0.5` at epoch 100, 200 epochs,
  batch size 8 — all LibMTL argparser defaults, so the command above already uses them.
  `--arch TSsBN` applies `convert="clip+bias"` automatically.

The script's only TSsBN touches — all inside it, none in LibMTL — are `register()`,
`NYUtrainer` subclassing `TSsBNTrainer`, and setting `arch_args["convert"]`.

## Note on device

LibMTL's `Trainer` hardcodes `cuda:0`. `TSsBNTrainer` respects a `device` argument, so it also
runs on `"cpu"`.
