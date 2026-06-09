"""choan.md §2.1 — "is the attack even separable in the internal representation?"

The §2.2 scalarizers reduce the hidden state to ONE number per layer before
thresholding. This module answers the prior, coarser question: feed the FULL
per-layer hidden vector to a logistic-regression probe and see whether attack
(B,D) vs benign (C,E,F,G) tokens are linearly separable at all. choan.md's
finding ("internal representation 을 logistic regression 했더니 잘 나왔다") is the
per-layer probe AUC reported here.

Pure analysis over the saved stage-03 artifacts (hidden cube + token rows); the
heavy lifting is :func:`core.metrics.probe_layer` (logistic regression with a
numpy nearest-centroid fallback, GroupKFold by prompt for a leakage-free AUC).
"""

from __future__ import annotations

import numpy as np

from . import metrics
from .cascade import binary_labels


def per_layer_separability(rows: list[dict], hidden: np.ndarray,
                           folds: int = 5, seed: int = 0) -> dict:
    """Cross-validated probe AUC of attack-vs-benign from the full hidden vector,
    one row per layer. ``rows`` and ``hidden`` come from stage 03; ``hidden`` is
    indexed by ``row_id`` so a balanced row subset stays aligned.

    Returns ``{n_layers, n_pos, n_neg, method, split, per_layer:[...],
    best_layer, best_auc}``.
    """
    y = binary_labels(rows)
    keep = y >= 0
    rows_k = [r for r, m in zip(rows, keep) if m]
    yk = y[keep]
    if not rows_k or not getattr(hidden, "size", 0):
        return {"n_layers": 0, "per_layer": [], "best_layer": None, "best_auc": None,
                "note": "no hidden cube or no labelled rows"}

    ridx = np.array([r["row_id"] for r in rows_k], dtype=int)
    Hsub = hidden[ridx].astype(np.float32)              # [n, L+1, dim]
    groups = np.array([int(r["sample_index"]) for r in rows_k], dtype=int)
    n_layers = Hsub.shape[1]

    per_layer = []
    best_layer, best_auc = None, -1.0
    method = split = None
    for l in range(n_layers):
        res = metrics.probe_layer(Hsub[:, l, :], yk, folds=folds, seed=seed, groups=groups)
        res["layer"] = l
        per_layer.append(res)
        method, split = res.get("method"), res.get("split")
        a = res.get("auc")
        if a is not None and a == a and a > best_auc:
            best_auc, best_layer = a, l
    return {
        "n_layers": n_layers,
        "n_pos": int((yk == 1).sum()),
        "n_neg": int((yk == 0).sum()),
        "method": method,
        "split": split,
        "best_layer": best_layer,
        "best_auc": None if best_layer is None else round(best_auc, 5),
        "per_layer": per_layer,
    }
