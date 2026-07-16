# TSsBN for MLoRE (ViT-S, PascalContext)

Run **Task-Specific Sigmoid Normalization** on a pretrained ViT-S under the
[MLoRE](https://github.com/YuqiYang213/MLoRE) protocol — the setup behind **Table 3**
(5-task PascalContext).

MLoRE is a research codebase, not a plugin framework: its `get_backbone` / `get_model` /
`get_optimizer` are `if/elif` chains, so this integration is a **model file plus four small
edits** — three to `utils/common_config.py` and one to `main.py` — against a clone of MLoRE.
(Contrast the LibMTL add-on, which needs no upstream edits.)

Requires `tssnorm` (this repo) plus MLoRE's own deps (`timm`, `einops`) — see the repo
`requirements.txt`. The training run needs the PascalContext dataset and a GPU (the paper
trains 60k iterations).

## 1. Clone MLoRE

```bash
git clone https://github.com/YuqiYang213/MLoRE.git
cd MLoRE
# this integration was written against commit 5817007
```

## 2. Copy the integration files

| file | goes to | what |
|------|---------|------|
| `TSsBN.py` | `MLoRE/models/TSsBN.py` | `TSsBNViT`, `MultiScaleFusion`, `get_tssbn_optimizer` |
| `configs/pascal/pascal_vitSp16_TSsBN.yml` | `MLoRE/configs/pascal/` | the Table 3 config |
| `run_TSsBN_pascal_vitS.sh` | `MLoRE/` | launch command |

```bash
cp integrations/mlore/TSsBN.py                                 MLoRE/models/
cp integrations/mlore/configs/pascal/pascal_vitSp16_TSsBN.yml  MLoRE/configs/pascal/
cp integrations/mlore/run_TSsBN_pascal_vitS.sh                 MLoRE/
```

## 3. Make the three `common_config.py` edits, and one to `main.py`

The edits are additive. TSsBN uses a **plain** pretrained ViT-S — not MLoRE's
mixture-of-experts backbone — so the first edit adds that backbone.

**a. `get_backbone` — add a plain ViT-S branch** (alongside the existing `elif p['backbone']
== 'MLoRE_vitS'`):

```python
elif p['backbone'] == 'vitS':
    import timm
    backbone = timm.create_model(
        'vit_small_patch16_384',
        pretrained=True,
        img_size=p.TRAIN.SCALE,
        num_classes=0,       # no classification head
        global_pool=None,    # no global pooling
    )
    backbone_channels = backbone.embed_dim
```

**b. `get_model` — add a branch** (alongside the existing `elif p['model'] == ...`):

```python
elif p['model'] in ['TSsBN', 'TSBN']:
    from models.TSsBN import TSsBNViT
    heads = torch.nn.ModuleDict({task: get_head(p, p.msf.out, task) for task in p.TASKS.NAMES})
    model = TSsBNViT(backbone, heads,
                     msf_hidden=p.msf.hidden, msf_out=p.msf.out, msf_depth=p.msf.depth,
                     img_size=p.TRAIN.SCALE[0], sigmoid=p['model'] == 'TSsBN')
```

**c. `get_optimizer` — add a branch** (before the `sgd`/`adam` cases):

```python
if p['model'] == 'TSsBN':
    from models.TSsBN import get_tssbn_optimizer
    optimizer = get_tssbn_optimizer(model, **p['optimizer_kwargs'])
elif p['optimizer'] == 'sgd':
    ...   # existing code unchanged from here
```

The `TSBN` baseline uses the regular `adam` optimizer — it has no scalers to accelerate. The
`pascal_vitSp16_TSsBN.yml` config sets `backbone: vitS` and `model: TSsBN`, so these three
branches are what it exercises.

**d. `main.py` — drop the `SyncBatchNorm` conversion.** Line 91 is

```python
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
```

Change it to `model = model.cuda()`. `SyncBatchNorm` is not compatible with sigmoid-modulated
norms — the conversion replaces them with plain ones and drops the modulation. This integration
therefore runs **single-GPU**, which is what the launch command does (`--nproc_per_node=1`);
`SyncBatchNorm` only does work across processes, so nothing is lost. Multi-GPU support is not
implemented.

## 4. Get PascalContext

Follow MLoRE's own README to download and place PascalContext — the integration changes
nothing about the data pipeline. Point the config / dataset path as MLoRE documents.

## 5. Train

```bash
bash run_TSsBN_pascal_vitS.sh
# = python -m torch.distributed.launch --nproc_per_node=1 main.py \
#     --config_exp ./configs/pascal/pascal_vitSp16_TSsBN.yml --run_mode train
```

Config (`pascal_vitSp16_TSsBN.yml`): `backbone: vitS`, `model: TSsBN`, Adam `lr=2e-5`,
`lr_bn_mul=100`, `lr_ln_mul=10`, `weight_decay=1e-6`, poly schedule, `max_iter=60000`, MSF
`hidden=512 / out=192 / depth=2`; `optimizer_kwargs` are passed straight to
`get_tssbn_optimizer`.

For the **TSBN** baseline, set `model: TSBN` and drop `lr_bn_mul` / `lr_ln_mul` from
`optimizer_kwargs` — TSBN goes through MLoRE's plain `adam` branch, which forwards those keys
straight to `torch.optim.Adam` and would reject them.
