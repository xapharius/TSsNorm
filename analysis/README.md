# MTL analysis (Section 5)

The paper's interpretability framework: read multi-task structure — capacity allocation, task
relationships, filter specialization — straight off a trained model's TSsBN scalers, at no cost
beyond training.

The functions live in the package, [`tssnorm.analysis`](../tssnorm/analysis.py), and take the
task-filter importance matrix `I = sigmoid(gamma)` (shape `[tasks, filters]`) that
`tssnorm.get_importance(model)` returns:

| function | paper | what |
|----------|-------|------|
| `capacity(I)` | 5.1 | per-task total capacity `C_t` |
| `capacity_decomposition(I)` | 5.1 (Eq. 4–5) | shared vs independent capacity |
| `task_similarity(I)` | 5.2 | pairwise cosine similarity of task importance vectors |
| `task_clusters(I)` | 5.2 | hierarchical task clustering |
| `specialized_filters(I, tau)` | 5.3 | which filters specialize to a single task |
| `similarity_stability([...])` | 5.2 | cross-run Spearman stability of the relationships |

They are pure `numpy`/`scipy` — no model, data, or training needed once you have `I`. Install
the extra deps with `pip install -e ".[analysis]"`.

## Demo

[`analysis.ipynb`](analysis.ipynb) runs the toolkit on a **real TSsBN model** — the NYUv2
three-task DeepLabV3 + ResNet-50 from Table 2 — and recovers the paper's Figure 3 result (most
capacity shared, no task dominating). It ships with its outputs so it reads without the data.

The importance matrix is not committed. `models/` (git-ignored) is where it goes — the demo
loads `models/nyuv2_tssbn.npy`. To analyse your own model, save `get_importance(model)` (a
`[tasks, filters]` tensor) into `models/` and point the loader at it. A many-task model (e.g.
40-task CelebA) additionally makes the `task_clusters` / `similarity_stability` analysis of 5.2
rich.
