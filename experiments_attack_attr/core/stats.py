"""Prompt-grouped resampling statistics: bootstrap CIs and permutation tests.

Resampling and label-shuffling are done at the **prompt (group) level** so the
within-prompt token correlation does not inflate significance — the same honesty
discipline as the GroupKFold AUC used elsewhere.
"""

from __future__ import annotations

import numpy as np

from . import metrics


def _clean(scores, y):
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    mask = (y >= 0) & ~np.isnan(scores)
    return scores[mask], y[mask], mask


def bootstrap_auc_ci(scores: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """Group-bootstrap 95% CI for ROC-AUC (resample whole prompts)."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    valid = (y >= 0) & ~np.isnan(scores)
    if valid.sum() == 0 or y[valid].min() == y[valid].max():
        return {"auc": None, "lo": None, "hi": None, "n_boot": 0}
    obs = float(metrics.roc_auc(scores[valid], y[valid]))
    uniq = np.array(sorted(set(int(g) for g in groups)))
    by_group = {int(g): np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(max(1, n_boot)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[int(g)] for g in pick])
        idx = idx[(y[idx] >= 0) & ~np.isnan(scores[idx])]
        if len(idx) == 0 or y[idx].min() == y[idx].max():
            continue
        aucs.append(metrics.roc_auc(scores[idx], y[idx]))
    if not aucs:
        return {"auc": round(obs, 5), "lo": None, "hi": None, "n_boot": 0}
    return {"auc": round(obs, 5),
            "lo": round(float(np.percentile(aucs, 2.5)), 5),
            "hi": round(float(np.percentile(aucs, 97.5)), 5),
            "n_boot": len(aucs)}


def permutation_pvalue(scores: np.ndarray, y: np.ndarray, groups: np.ndarray,
                       n_perm: int = 1000, seed: int = 0) -> dict:
    """Permutation p-value for separation, shuffling labels WITHIN each prompt.

    Statistic = |AUC - 0.5|. p = (1 + #{perm >= observed}) / (n_perm + 1)."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    valid = (y >= 0) & ~np.isnan(scores)
    if valid.sum() == 0 or y[valid].min() == y[valid].max():
        return {"observed_auc": None, "p_value": None, "n_perm": 0}
    s = scores[valid]; yy = y[valid]; gg = groups[valid]
    obs = abs(metrics.roc_auc(s, yy) - 0.5)
    by_group = {int(g): np.where(gg == g)[0] for g in sorted(set(int(g) for g in gg))}
    rng = np.random.default_rng(seed)
    count = 0
    done = 0
    for _ in range(max(1, n_perm)):
        yp = yy.copy()
        for idx in by_group.values():
            yp[idx] = rng.permutation(yy[idx])
        if yp.min() == yp.max():
            continue
        stat = abs(metrics.roc_auc(s, yp) - 0.5)
        count += int(stat >= obs)
        done += 1
    p = (1 + count) / (done + 1) if done else None
    return {"observed_auc": round(float(metrics.roc_auc(s, yy)), 5),
            "p_value": None if p is None else round(float(p), 5),
            "n_perm": done}
