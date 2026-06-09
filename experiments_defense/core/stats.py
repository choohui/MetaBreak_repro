"""Tiny detector-stats helpers (numpy only): ROC-AUC, best layer, thresholds.

All scorers follow the convention *higher score = more attack-like*, so a token /
prompt is flagged when ``score >= threshold``.
"""

from __future__ import annotations

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC-AUC for binary labels (1 = positive). Returns 0.5 on a
    degenerate single-class input."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = csum - (counts - 1) / 2.0
    ranks = avg[inv]
    sum_pos = ranks[y == 1].sum()
    auc = (sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def best_layer(score_matrix: np.ndarray, labels: np.ndarray) -> tuple[int, float]:
    """Pick the layer (column of ``[n, L]``) with the highest |AUC-0.5|, returning
    a signed orientation so higher-score = attack. Returns (layer, signed_auc)."""
    n, L = score_matrix.shape
    best_l, best_sep, best_auc = 0, -1.0, 0.5
    for l in range(L):
        col = score_matrix[:, l]
        if not np.isfinite(col).any():
            continue
        auc = roc_auc(col, labels)
        sep = abs(auc - 0.5)
        if sep > best_sep:
            best_l, best_sep, best_auc = l, sep, auc
    return best_l, best_auc


def threshold_fpr(neg_scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """Threshold so that at most ``target_fpr`` of the negatives are flagged
    (score >= threshold)."""
    neg = np.asarray(neg_scores, dtype=np.float64)
    if neg.size == 0:
        return 0.0
    return float(np.quantile(neg, 1.0 - target_fpr))


def threshold_youden(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising TPR - FPR over candidate cut points."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float(np.median(s))
    cands = np.unique(s)
    best_t, best_j = cands[0], -1.0
    for t in cands:
        tpr = float((pos >= t).mean())
        fpr = float((neg >= t).mean())
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)
