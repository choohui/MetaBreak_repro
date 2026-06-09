"""ROC / DET / PR / calibration point generators (numpy-only, no plotting dep).

Each returns a small list of points (down-sampled to <= ``max_points``) so the
reports stay light and the figures can be drawn later from JSON. Scores are
oriented so 'higher = more attack-like' before sweeping thresholds.
"""

from __future__ import annotations

import numpy as np

from . import metrics


def _oriented(scores: np.ndarray, y: np.ndarray):
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    mask = (y >= 0) & ~np.isnan(scores)
    s, yy = scores[mask], y[mask]
    if len(s) == 0 or yy.min() == yy.max():
        return s, yy, "higher_is_attack"
    if metrics.roc_auc(s, yy) < 0.5:
        return -s, yy, "lower_is_attack"
    return s, yy, "higher_is_attack"


def _thresholds(s: np.ndarray, max_points: int) -> np.ndarray:
    u = np.unique(s)
    if len(u) <= max_points:
        return u
    qs = np.linspace(0, 100, max_points)
    return np.unique(np.percentile(u, qs))


def roc_det_pr(scores: np.ndarray, y: np.ndarray, max_points: int = 100) -> dict:
    """ROC (fpr,tpr), DET (fpr,fnr) and PR (recall,precision) point lists."""
    s, yy, direction = _oriented(scores, y)
    if len(s) == 0 or yy.min() == yy.max():
        return {"direction": direction, "roc": [], "det": [], "pr": [], "auc": None}
    pos = s[yy == 1]; neg = s[yy == 0]
    P = len(pos); N = len(neg)
    roc, det, pr = [], [], []
    for t in _thresholds(s, max_points):
        tp = int((pos >= t).sum()); fp = int((neg >= t).sum())
        tpr = tp / P if P else 0.0
        fpr = fp / N if N else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        roc.append({"threshold": float(t), "fpr": round(fpr, 5), "tpr": round(tpr, 5)})
        det.append({"threshold": float(t), "fpr": round(fpr, 5), "fnr": round(1 - tpr, 5)})
        pr.append({"threshold": float(t), "recall": round(tpr, 5), "precision": round(prec, 5)})
    return {"direction": direction,
            "auc": round(float(metrics.roc_auc(s, yy)), 5),
            "roc": roc, "det": det, "pr": pr}


def calibration(scores: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Reliability bins: min-max scaled oriented score vs empirical attack rate."""
    s, yy, _ = _oriented(scores, y)
    if len(s) == 0 or yy.min() == yy.max():
        return []
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return []
    z = (s - lo) / (hi - lo)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for b in range(n_bins):
        m = (z >= edges[b]) & (z <= edges[b + 1] if b == n_bins - 1 else z < edges[b + 1])
        if m.sum() == 0:
            continue
        out.append({"bin": b, "n": int(m.sum()),
                    "mean_score": round(float(z[m].mean()), 5),
                    "attack_rate": round(float(yy[m].mean()), 5)})
    return out
