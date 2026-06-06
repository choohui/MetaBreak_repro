"""Threshold selectors + threshold-stability — the decision rule on top of a
scalar (hc_4_claude). Every selector is fit on the **TRAIN** scores only and
returns a threshold on the *oriented* score (``metrics.binary_metrics`` flips the
score when lower=attack, so callers must orient identically; see
:func:`core.cascade.predict`).

The hc_2 lesson encoded here: a threshold that is unstable across resamples is the
early warning that it will collapse on held-out data. :func:`threshold_stability`
measures that instability (mean / std / cv across CV folds + bootstraps) so stage
05 can *prefer a stable threshold*, not merely the highest in-sample AUC.
"""

from __future__ import annotations

import numpy as np

from . import metrics

# Selector keys understood by :func:`select_threshold`. Parameterised forms:
#   fpr@1 fpr@5 fpr@10  ;  pct_benign@95 pct_benign@99  ;  cost (uses fn_fp_cost)
KNOWN_PREFIXES = ("youden", "fpr@", "eer", "pct_benign@", "cost")


def _orient(scores: np.ndarray, y: np.ndarray):
    """Return (oriented_scores, direction) using the AUC-based orientation."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    mask = (y >= 0) & ~np.isnan(scores)
    if mask.sum() == 0 or y[mask].min() == y[mask].max():
        return scores, "higher_is_attack"
    auc = metrics.roc_auc(scores[mask], y[mask])
    if auc < 0.5:
        return -scores, "lower_is_attack"
    return scores, "higher_is_attack"


def select_threshold(scores: np.ndarray, y: np.ndarray, method: str,
                     fn_fp_cost: float = 1.0) -> dict:
    """Fit one threshold on TRAIN ``scores`` (labels ``y``: 1/0/-1).

    Returns ``{method, threshold, direction, auc}`` with the threshold on the
    oriented score (``None`` if degenerate)."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    mask = (y >= 0) & ~np.isnan(scores)
    s_all, direction = _orient(scores, y)
    s = s_all[mask]
    yy = y[mask]
    base = {"method": method, "direction": direction, "threshold": None, "auc": None}
    if len(s) == 0 or yy.min() == yy.max():
        return base
    base["auc"] = round(float(metrics.roc_auc(s, yy)), 5)
    pos = s[yy == 1]
    neg = s[yy == 0]
    if len(pos) == 0 or len(neg) == 0:
        return base

    thr = None
    if method == "youden":
        thresholds = np.unique(s)
        best_j = -1.0
        for t in thresholds:
            j = (pos >= t).mean() - (neg >= t).mean()
            if j > best_j:
                best_j, thr = j, float(t)
    elif method.startswith("fpr@"):
        target = float(method.split("@", 1)[1]) / 100.0
        thresholds = np.unique(s)
        best_tpr = -1.0
        for t in thresholds:
            fpr = (neg >= t).mean()
            tpr = (pos >= t).mean()
            if fpr <= target and tpr > best_tpr:
                best_tpr, thr = tpr, float(t)
        if thr is None:                       # nothing meets the budget -> Youden
            return select_threshold(scores, y, "youden", fn_fp_cost) | {"method": method}
    elif method == "eer":
        thresholds = np.unique(s)
        best_gap = np.inf
        for t in thresholds:
            fpr = (neg >= t).mean()
            fnr = 1.0 - (pos >= t).mean()
            gap = abs(fpr - fnr)
            if gap < best_gap:
                best_gap, thr = gap, float(t)
    elif method.startswith("pct_benign@"):
        p = float(method.split("@", 1)[1])
        thr = float(np.percentile(neg, p))
    elif method == "cost":
        thresholds = np.unique(s)
        best_cost = np.inf
        for t in thresholds:
            fpr = (neg >= t).mean()
            fnr = 1.0 - (pos >= t).mean()
            cost = fn_fp_cost * fnr + fpr
            if cost < best_cost:
                best_cost, thr = cost, float(t)
    else:
        # unknown -> Youden fallback
        return select_threshold(scores, y, "youden", fn_fp_cost) | {"method": method}

    base["threshold"] = thr
    return base


def predict(scores: np.ndarray, threshold: float | None, direction: str | None) -> np.ndarray:
    """Boolean flag for an oriented threshold (mirrors core.cascade.predict)."""
    if threshold is None:
        return np.zeros(len(scores), dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    s = s if direction != "lower_is_attack" else -s
    return s >= threshold


def _group_folds(groups: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Indices of each of k group-disjoint folds (the TEST side of each split)."""
    uniq = np.array(sorted(set(int(g) for g in groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    k = max(2, min(k, len(uniq)))
    fold_of = {int(g): i % k for i, g in enumerate(uniq)}
    folds = []
    for f in range(k):
        folds.append(np.array([i for i, g in enumerate(groups) if fold_of[int(g)] == f], dtype=int))
    return folds


def threshold_stability(scores: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        method: str, fn_fp_cost: float = 1.0,
                        n_folds: int = 5, n_bootstrap: int = 200, seed: int = 0) -> dict:
    """Refit the threshold across CV folds + bootstraps; report drift.

    A low ``cv`` (= std/|mean|) signals a threshold that should transfer to
    held-out; a high ``cv`` is the hc_2 collapse warning."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    base = select_threshold(scores, y, method, fn_fp_cost)

    fold_thrs: list[float] = []
    folds = _group_folds(groups, n_folds, seed)
    for test_idx in folds:                       # fit on the complement (train)
        mask = np.ones(len(scores), dtype=bool)
        mask[test_idx] = False
        r = select_threshold(scores[mask], y[mask], method, fn_fp_cost)
        if r["threshold"] is not None:
            fold_thrs.append(float(r["threshold"]))

    boot_thrs: list[float] = []
    uniq = np.array(sorted(set(int(g) for g in groups)))
    rng = np.random.default_rng(seed + 1)
    by_group = {int(g): np.where(groups == g)[0] for g in uniq}
    for _ in range(max(0, n_bootstrap)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[int(g)] for g in pick]) if len(pick) else np.array([], int)
        if len(idx) == 0:
            continue
        r = select_threshold(scores[idx], y[idx], method, fn_fp_cost)
        if r["threshold"] is not None:
            boot_thrs.append(float(r["threshold"]))

    arr = np.array(boot_thrs) if boot_thrs else np.array(fold_thrs)
    if arr.size:
        mean = float(np.mean(arr)); std = float(np.std(arr))
        cv = float(std / abs(mean)) if mean != 0 else None
        lo = float(np.percentile(arr, 2.5)); hi = float(np.percentile(arr, 97.5))
    else:
        mean = std = cv = lo = hi = None
    return {
        "method": method,
        "direction": base["direction"],
        "threshold": base["threshold"],
        "auc": base["auc"],
        "threshold_mean": None if mean is None else round(mean, 6),
        "threshold_std": None if std is None else round(std, 6),
        "threshold_cv": None if cv is None else round(cv, 6),
        "threshold_ci95": [None, None] if arr.size == 0 else [round(lo, 6), round(hi, 6)],
        "n_fold_fits": len(fold_thrs),
        "n_bootstrap_fits": len(boot_thrs),
    }
