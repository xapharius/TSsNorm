# TSsBN in MLoRE

Runs TSsBN on a pretrained ViT-S with [MLoRE](https://github.com/YuqiYang213/MLoRE). This is the
setup of Table 3: PascalContext with 5 tasks.

The ViT's LayerNorms become task-specific but keep their plain pretrained scales. The sigmoid
applies to a new LayerNorm after the patch embedding, and to the BatchNorms of a small fusion
module on top of the ViT. The importance matrix comes from the fusion module.

MLoRE has no plugin mechanism. Its `get_backbone`, `get_model` and `get_optimizer` are chains of
`if` statements. So this integration is one model file, plus three small edits to
`utils/common_config.py` and one to `main.py`.

Training needs PascalContext and a GPU. The paper trains for 60,000 iterations.

## 1. Clone MLoRE and install

Set up MLoRE's environment as its README describes, then add this integration's packages:

```bash
git clone https://github.com/YuqiYang213/MLoRE.git    # written against commit 5817007
cd MLoRE
pip install -r <this repo>/mtl/integrations/mlore/requirements.txt
pip install -e <this repo>                             # tssnorm
```

## 2. Copy the files

| file | goes to | content |
|------|---------|---------|
| `TSsBN.py` | `MLoRE/models/` | `TSsBNViT`, `MultiScaleFusion`, `get_tssbn_optimizer` |
| `configs/pascal/pascal_vitSp16_TSsBN.yml` | `MLoRE/configs/pascal/` | the Table 3 config |
| `run_TSsBN_pascal_vitS.sh` | `MLoRE/` | the launch command |

From inside `MLoRE/`:

```bash
cp <this repo>/mtl/integrations/mlore/TSsBN.py                                 models/
cp <this repo>/mtl/integrations/mlore/configs/pascal/pascal_vitSp16_TSsBN.yml  configs/pascal/
cp <this repo>/mtl/integrations/mlore/run_TSsBN_pascal_vitS.sh                 .
```

## 3. Edit MLoRE

The `common_config.py` edits only add code. The `main.py` edit changes one line. TSsBN uses a
plain pretrained ViT-S, not MLoRE's mixture-of-experts backbone, so the first edit adds that
backbone.

In `utils/common_config.py`, add a branch to `get_backbone`, next to
`elif p['backbone'] == 'MLoRE_vitS'`:

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

Add a branch to `get_model`, next to the other models:

```python
elif p['model'] in ['TSsBN', 'TSBN']:
    from models.TSsBN import TSsBNViT
    heads = torch.nn.ModuleDict({task: get_head(p, p.msf.out, task) for task in p.TASKS.NAMES})
    model = TSsBNViT(backbone, heads,
                     msf_hidden=p.msf.hidden, msf_out=p.msf.out, msf_depth=p.msf.depth,
                     img_size=p.TRAIN.SCALE[0], sigmoid=p['model'] == 'TSsBN')
```

Add a branch to `get_optimizer`, before the `sgd` and `adam` cases:

```python
if p['model'] == 'TSsBN':
    from models.TSsBN import get_tssbn_optimizer
    optimizer = get_tssbn_optimizer(model, **p['optimizer_kwargs'])
elif p['optimizer'] == 'sgd':
    ...   # the existing code
```

In `main.py`, one line converts the model to `SyncBatchNorm`:

```python
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
```

Change it to `model = model.cuda()`. The conversion would replace the sigmoid norms with plain
ones. Without it, training runs on one GPU, which is what the launch command does. Multi-GPU
training is not supported.

## 4. Get PascalContext

Download it as MLoRE's README describes. The integration does not change the data pipeline.

## 5. Train

```bash
bash run_TSsBN_pascal_vitS.sh
```

The config uses Adam at `2e-5`, with the BatchNorm scalers at 100 times that rate and the
LayerNorms at 10 times. It also sets weight decay `1e-6`, a poly schedule, 60,000 iterations,
and a fusion module with `hidden=512`, `out=192` and `depth=2`. `optimizer_kwargs` go straight
to `get_tssbn_optimizer`.

For the `TSBN` baseline, set `model: TSBN` and remove `lr_bn_mul` and `lr_ln_mul` from
`optimizer_kwargs`. `TSBN` uses MLoRE's plain `adam` branch, which passes those keys to
`torch.optim.Adam`, and Adam rejects them.
