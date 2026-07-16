"""Reading multi-task structure off the TSsBN scalers — the paper's Section 5 analysis.

These read the task-filter importance matrix ``I = sigmoid(gamma)`` of shape
``[tasks, filters]`` — exactly what :func:`tssnorm.get_importance` returns for a trained model.
(:func:`similarity_stability` is the exception: it compares the *similarity* matrices of
several runs.) The importance matrix is read straight from the trained parameters, so the whole
analysis is free: no extra forward passes, no ablation. Pure numpy/scipy; no model or data
needed here.

    from tssnorm import get_importance
    from tssnorm.analysis import capacity, capacity_decomposition, task_similarity

    I = get_importance(model)                  # [tasks, filters]
    shared, indep = capacity_decomposition(I)  # Section 5.1, Fig. 3
    S = task_similarity(I)                     # Section 5.2, Table 5 clusters
"""

import warnings
from itertools import combinations
from typing import Any, Iterable, Optional, Tuple

import numpy as np

# `I` is the importance matrix — a numpy array or a torch tensor (accepted via _asarray).
Matrix = Any

__all__ = [
    "capacity",
    "capacity_decomposition",
    "task_similarity",
    "task_clusters",
    "specialized_filters",
    "similarity_stability",
]


def _asarray(I: Matrix) -> np.ndarray:
    if hasattr(I, "detach"):          # torch tensor, without importing torch
        I = I.detach().cpu().float()  # a model trained in half precision still analyses in full
    return np.asarray(I, dtype=float)


def capacity(I: Matrix) -> np.ndarray:
    """Total capacity of each task, ``C_t = mean_i I[t, i]`` (Section 5.1). Shape ``[tasks]``."""
    return _asarray(I).mean(axis=1)


def capacity_decomposition(I: Matrix) -> Tuple[np.ndarray, np.ndarray]:
    """Split each task's capacity into shared and independent parts (Eq. 4-5, Fig. 3).

    Each task vector ``I_t`` is projected onto the span of the *other* task vectors; the
    projection is the shared component and the residual the independent one. Scaled to the total
    capacity so that ``shared_t**2 + indep_t**2 == C_t**2`` (Pythagoras, the components being
    orthogonal). Returns ``(shared, independent)``, each shape ``[tasks]``.
    """
    I = _asarray(I)
    C = I.mean(axis=1)
    shared = np.zeros(len(I))
    indep = np.zeros(len(I))
    for t in range(len(I)):
        A = np.delete(I, t, axis=0).T                # [filters, tasks-1]: the other task vectors
        # Associate right: A @ (pinv(A) @ I_t) is O(filters * tasks); the left-to-right
        # A @ pinv(A) would first build a filters x filters matrix.
        projection = A @ (np.linalg.pinv(A) @ I[t])  # P_A I_t — the shared component
        residual = I[t] - projection                 # I_t^perp — the independent component
        norm = np.linalg.norm(I[t])
        if norm > 0:
            shared[t] = np.linalg.norm(projection) / norm * C[t]
            indep[t] = np.linalg.norm(residual) / norm * C[t]
    return shared, indep


def task_similarity(I: Matrix) -> np.ndarray:
    """Pairwise cosine similarity between task importance vectors (Section 5.2). ``[tasks, tasks]``.

    1 means identical filter usage, 0 orthogonal.
    """
    I = _asarray(I)
    unit = I / np.clip(np.linalg.norm(I, axis=1, keepdims=True), 1e-12, None)
    return np.clip(unit @ unit.T, -1.0, 1.0)


def task_clusters(I: Matrix, n_clusters: Optional[int] = None, threshold: Optional[float] = None,
                  method: str = "average"):
    """Hierarchically cluster tasks by importance-vector similarity (Section 5.2, Table 5).

    Without ``n_clusters`` or ``threshold`` returns the SciPy linkage matrix (feed it to
    ``dendrogram``). With either, returns a flat cluster label per task.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    distance = 1.0 - task_similarity(I)
    distance = np.clip((distance + distance.T) / 2, 0, None)
    np.fill_diagonal(distance, 0.0)
    Z = linkage(squareform(distance, checks=False), method=method)
    if n_clusters is not None:
        return fcluster(Z, n_clusters, criterion="maxclust")
    if threshold is not None:
        return fcluster(Z, threshold, criterion="distance")
    return Z


def specialized_filters(I: Matrix, tau: float = 0.5) -> np.ndarray:
    """Boolean ``[tasks, filters]``: filter ``i`` is specialized for task ``t`` if it contributes
    mostly to that task, ``I[t, i] / sum_t I[t, i] > tau`` (Section 5.3). ``tau=0.5`` means a
    single task claims the majority of the filter's usage.
    """
    I = _asarray(I)
    share = I / np.clip(I.sum(axis=0, keepdims=True), 1e-12, None)
    return share > tau


def similarity_stability(similarities: Iterable[Matrix]) -> float:
    """Mean pairwise Spearman rank correlation between task-similarity matrices from independent
    runs (Section 5.2 — how stable the recovered task relationships are across seeds). Pass the
    ``task_similarity`` matrices of several trained models; returns a scalar in ``[-1, 1]``.

    The correlation is taken over the off-diagonal task pairs, so it is only meaningful with
    enough of them — a handful of tasks gives a rank correlation over a handful of points.
    """
    from scipy.stats import spearmanr

    mats = [_asarray(S) for S in similarities]
    if len(mats) < 2:
        raise ValueError("need at least two similarity matrices to measure stability")
    if any(S.ndim != 2 or S.shape[0] != S.shape[1] for S in mats):
        raise ValueError(
            "expected square [tasks, tasks] similarity matrices — pass task_similarity(I), "
            "not the importance matrix I itself"
        )
    if len({S.shape for S in mats}) != 1:
        raise ValueError(
            f"the runs describe different task counts {sorted({S.shape[0] for S in mats})}; "
            "their task relationships are not comparable"
        )
    n_tasks = len(mats[0])
    n_pairs = n_tasks * (n_tasks - 1) // 2
    if n_pairs < 3:
        raise ValueError(
            f"{n_tasks} tasks give only {n_pairs} task pair(s) — too few for a rank "
            "correlation; stability is meant for many-task models"
        )
    uppers = [S[np.triu_indices(n_tasks, k=1)] for S in mats]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")   # a constant input is reported below, not warned about
        rs = [spearmanr(a, b).correlation for a, b in combinations(uppers, 2)]
    if not np.isfinite(rs).all():
        raise ValueError(
            "task similarities are constant across pairs, so their rank correlation is "
            "undefined — the runs carry no relationship structure to compare"
        )
    return float(np.mean(rs))
