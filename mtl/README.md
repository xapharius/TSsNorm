# TSσNorm: multi-task learning

Code for *Simplifying Multi-Task Architectures Through Task-Specific Normalization* (TMLR).

Each task gets its own sigmoid norms. The rest of the backbone is shared. To train a model, use
the code in the [root README](../README.md#usage) with your tasks as task names.

The paper calls the method TSsBN when it modulates BatchNorm. Its baseline, TSBN, gives each task
its own BatchNorm without the sigmoid.

## Analysis (Section 5)

The trained scales form an importance matrix `I` of shape `[tasks, filters]`.
`tssnorm.get_importance(model)` returns it. The functions in
[`tssnorm.mtl`](../tssnorm/mtl.py) compute the paper's analysis from `I`. They need numpy and
scipy. Install them with `pip install -e ".[analysis]"`.

| function | paper | result |
|----------|-------|--------|
| `capacity(I)` | 5.1 | capacity of each task |
| `capacity_decomposition(I)` | 5.1, Eq. 4 and 5 | shared and independent capacity |
| `task_similarity(I)` | 5.2 | cosine similarity between tasks |
| `task_clusters(I)` | 5.2 | hierarchical clustering of tasks |
| `specialized_filters(I, tau)` | 5.3 | filters used by a single task |
| `similarity_stability([...])` | 5.2 | Spearman stability of task similarity across runs |

[`examples/analysis.ipynb`](examples/analysis.ipynb) runs the first four functions on the NYUv2
model from Table 2, and reproduces Figure 3. The other two need many tasks. The notebook is
saved with its outputs.

The notebook loads the importance matrix from `examples/models/nyuv2_tssbn.npy`, which is not
in git. To analyse your own model, save its matrix and point `MATRIX` in the notebook at it:

```python
numpy.save("examples/models/my_model.npy", get_importance(model).cpu().numpy())
```

## Integrations

| path | framework | paper |
|------|-----------|-------|
| [`integrations/libmtl/`](integrations/libmtl/) | [LibMTL](https://github.com/median-research-group/LibMTL), DeepLabV3 and ResNet-50 on NYUv2 and Cityscapes | Table 2 |
| [`integrations/mlore/`](integrations/mlore/) | [MLoRE](https://github.com/YuqiYang213/MLoRE), ViT-S on PascalContext | Table 3 |

## Citation

```bibtex
@article{suteu2026tssnorm,
  title   = {Simplifying Multi-Task Architectures Through Task-Specific Normalization},
  author  = {Suteu, Mihai and Serban, Ovidiu},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=QNO893OXrS}
}
```
