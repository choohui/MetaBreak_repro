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


# --------------------------------------------------------------------------- #
# hc_7 additions — prompt-level rate (ASR / over-refusal) resampling
# --------------------------------------------------------------------------- #
def bootstrap_rate_ci(success: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap 95% CI for a per-prompt binary RATE (e.g. ASR, over-refusal).

    ``success`` is one bool/0-1 per prompt (the analysis unit is already the
    prompt, so resampling prompts == resampling the rate's samples)."""
    s = np.asarray(success, dtype=np.float64)
    s = s[~np.isnan(s)]
    n = len(s)
    if n == 0:
        return {"rate": None, "lo": None, "hi": None, "n": 0, "n_boot": 0}
    obs = float(s.mean())
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(max(1, n_boot)):
        pick = rng.integers(0, n, size=n)
        rates.append(float(s[pick].mean()))
    return {"rate": round(obs, 5),
            "lo": round(float(np.percentile(rates, 2.5)), 5),
            "hi": round(float(np.percentile(rates, 97.5)), 5),
            "n": n, "n_boot": len(rates)}


def permutation_delta_pvalue(success_a: np.ndarray, success_b: np.ndarray,
                             n_perm: int = 1000, seed: int = 0) -> dict:
    """Paired permutation test that rate(a) > rate(b) — i.e. steering REDUCED the
    rate (baseline ``a`` = alpha 0, treatment ``b`` = steered).

    ``success_a`` and ``success_b`` are aligned per-prompt binaries. Under the
    null (steering has no effect) the two outcomes of a prompt are exchangeable,
    so each permutation flips each prompt's (a,b) pair with prob 0.5. Statistic =
    ``mean(a) - mean(b)`` (positive = reduction). One-sided p."""
    a = np.asarray(success_a, dtype=np.float64)
    b = np.asarray(success_b, dtype=np.float64)
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    n = len(a)
    if n == 0:
        return {"delta": None, "p_value": None, "n_perm": 0, "n": 0}
    obs = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(max(1, n_perm)):
        flip = rng.random(n) < 0.5
        pa = np.where(flip, b, a)
        pb = np.where(flip, a, b)
        stat = float(pa.mean() - pb.mean())
        count += int(stat >= obs)
    p = (1 + count) / (max(1, n_perm) + 1)
    return {"delta": round(obs, 5), "p_value": round(float(p), 5),
            "n_perm": int(max(1, n_perm)), "n": n}
