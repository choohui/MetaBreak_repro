"""Numpy-only analysis helpers: ROC-AUC, single-threshold metrics, cosine,
a logistic-regression probe (with a nearest-centroid fallback), and Spearman.

Kept dependency-light so the analysis stages run even without scikit-learn.
"""

from __future__ import annotations

import numpy as np

try:  # optional; we degrade gracefully when unavailable
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - exercised only when sklearn missing
    _HAVE_SKLEARN = False


# --------------------------------------------------------------------------- #
# ROC-AUC + single-threshold metrics
# --------------------------------------------------------------------------- #


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based, tie-aware ROC-AUC (== Mann-Whitney U / (n_pos*n_neg))."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for ties
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    fpr_targets: tuple[float, ...] = (0.01, 0.05),
) -> dict:
    """ROC-AUC, Youden-optimal threshold, and TPR at fixed FPR budgets.

    Auto-orients the score: if AUC < 0.5 the score is flipped (lower = attack)
    and ``direction`` is reported as ``"lower_is_attack"``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = ~np.isnan(scores)
    scores, labels = scores[valid], labels[valid]
    if len(scores) == 0 or labels.sum() == 0 or (labels == 0).sum() == 0:
        return {"auc": float("nan"), "n": int(len(scores)), "direction": None}

    auc = roc_auc(scores, labels)
    direction = "higher_is_attack"
    if auc < 0.5:
        scores = -scores
        auc = 1.0 - auc
        direction = "lower_is_attack"

    pos = scores[labels == 1]
    neg = scores[labels == 0]
    thresholds = np.unique(scores)
    best_j, best_t, best_tpr, best_fpr = -1.0, None, 0.0, 0.0
    tpr_at = {t: 0.0 for t in fpr_targets}
    for t in thresholds:
        tpr = float((pos >= t).mean())
        fpr = float((neg >= t).mean())
        j = tpr - fpr
        if j > best_j:
            best_j, best_t, best_tpr, best_fpr = j, float(t), tpr, fpr
        for target in fpr_targets:
            if fpr <= target and tpr > tpr_at[target]:
                tpr_at[target] = tpr
    return {
        "auc": round(float(auc), 5),
        "n": int(len(scores)),
        "n_pos": int(labels.sum()),
        "n_neg": int((labels == 0).sum()),
        "direction": direction,
        "youden_threshold": best_t,
        "youden_tpr": round(best_tpr, 5),
        "youden_fpr": round(best_fpr, 5),
        "tpr_at_fpr": {f"{int(t*100)}pct": round(tpr_at[t], 5) for t in fpr_targets},
    }


def tpr_fpr_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float,
                         higher_is_attack: bool = True) -> tuple[float, float]:
    """Given a fixed threshold, return (TPR, FPR)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    s = scores if higher_is_attack else -scores
    t = threshold if higher_is_attack else -threshold
    pred = s >= t
    pos = labels == 1
    neg = labels == 0
    tpr = float((pred & pos).sum() / max(1, pos.sum()))
    fpr = float((pred & neg).sum() / max(1, neg.sum()))
    return tpr, fpr


# --------------------------------------------------------------------------- #
# Cosine + Spearman
# --------------------------------------------------------------------------- #


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def cosine_rowwise(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Cosine of every row of ``mat`` [n, d] against ``vec`` [d] -> [n]."""
    mat = np.asarray(mat, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    num = mat @ vec
    den = np.linalg.norm(mat, axis=1) * (np.linalg.norm(vec) + 1e-12)
    out = np.full(mat.shape[0], np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (numpy-only)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i, n = 0, len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


# --------------------------------------------------------------------------- #
# Logistic-regression probe (full hidden vector) with nearest-centroid fallback
# --------------------------------------------------------------------------- #


def have_sklearn() -> bool:
    return _HAVE_SKLEARN


def probe_layer(x: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> dict:
    """Cross-validated separability of attack(1) vs benign(0) from ``x`` [n, d].

    Uses logistic regression if sklearn is available, else a nearest-centroid
    classifier. Returns mean ROC-AUC + balanced accuracy across folds.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos < 2 or n_neg < 2:
        return {"auc": float("nan"), "balanced_acc": float("nan"),
                "method": "none", "n_pos": n_pos, "n_neg": n_neg}
    k = max(2, min(folds, n_pos, n_neg))

    if _HAVE_SKLEARN:
        aucs, baccs = [], []
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for tr, te in skf.split(x, y):
            scaler = StandardScaler().fit(x[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(scaler.transform(x[tr]), y[tr])
            prob = clf.predict_proba(scaler.transform(x[te]))[:, 1]
            aucs.append(roc_auc(prob, y[te]))
            pred = (prob >= 0.5).astype(int)
            baccs.append(_balanced_acc(y[te], pred))
        return {"auc": round(float(np.nanmean(aucs)), 5),
                "balanced_acc": round(float(np.nanmean(baccs)), 5),
                "method": "logreg", "folds": k, "n_pos": n_pos, "n_neg": n_neg}

    # Fallback: nearest-centroid in standardized space.
    aucs, baccs = [], []
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    splits = np.array_split(idx, k)
    for fi in range(k):
        te = splits[fi]
        tr = np.concatenate([splits[j] for j in range(k) if j != fi])
        mu = x[tr].mean(axis=0)
        sd = x[tr].std(axis=0) + 1e-8
        xt = (x[tr] - mu) / sd
        xe = (x[te] - mu) / sd
        c1 = xt[y[tr] == 1].mean(axis=0)
        c0 = xt[y[tr] == 0].mean(axis=0)
        # score = closeness to attack centroid minus closeness to benign centroid
        score = (-np.linalg.norm(xe - c1, axis=1)) - (-np.linalg.norm(xe - c0, axis=1))
        aucs.append(roc_auc(score, y[te]))
        pred = (score >= 0).astype(int)
        baccs.append(_balanced_acc(y[te], pred))
    return {"auc": round(float(np.nanmean(aucs)), 5),
            "balanced_acc": round(float(np.nanmean(baccs)), 5),
            "method": "nearest_centroid", "folds": k, "n_pos": n_pos, "n_neg": n_neg}


def _balanced_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    p = int((y_true == 1).sum())
    n = int((y_true == 0).sum())
    sens = tp / p if p else 0.0
    spec = tn / n if n else 0.0
    return 0.5 * (sens + spec)
