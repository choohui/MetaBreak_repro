"""Small modeling helpers for hc_3."""

from __future__ import annotations

import warnings

import numpy as np

from . import metrics

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False

warnings.filterwarnings("ignore", message="'penalty' was deprecated.*")
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1.*")


def valid_binary_mask(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    return (y == 0) | (y == 1)


def fit_linear_model(x: np.ndarray, y: np.ndarray, c: float = 0.25):
    if not HAVE_SKLEARN:
        return _CentroidModel().fit(x, y)
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=c, penalty="l1", solver="liblinear"),
    ).fit(x, y)


def predict_score(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.score_samples(x)


def grouped_cv_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      seed: int = 0, folds: int = 5) -> dict:
    mask = valid_binary_mask(y)
    x, y, groups = x[mask], y[mask], groups[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"auc": float("nan"), "balanced_acc": float("nan"),
                "scores": np.array([]), "indices": np.array([], dtype=int)}

    original_idx = np.where(mask)[0]
    scores = np.zeros(len(y), dtype=np.float64)
    preds = np.zeros(len(y), dtype=int)

    if HAVE_SKLEARN and len(np.unique(groups)) >= 2:
        k = min(folds, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=k).split(x, y, groups)
        split_name = "group"
    else:
        k = max(2, min(folds, int((y == 0).sum()), int((y == 1).sum())))
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(x, y)
        split_name = "stratified"

    for tr, te in splitter:
        if len(np.unique(y[tr])) < 2:
            continue
        model = fit_linear_model(x[tr], y[tr])
        scores[te] = predict_score(model, x[te])
        preds[te] = (scores[te] >= 0.5).astype(int)

    auc = metrics.roc_auc(scores, y)
    bacc = _balanced_acc(y, preds)
    return {
        "auc": round(float(auc), 5),
        "balanced_acc": round(float(bacc), 5),
        "split": split_name,
        "folds": int(k),
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "scores": scores,
        "indices": original_idx,
    }


def threshold_for_fpr(scores: np.ndarray, y: np.ndarray, fpr: float) -> float:
    neg = np.asarray(scores)[np.asarray(y) == 0]
    if len(neg) == 0:
        return float("inf")
    return float(np.quantile(neg, max(0.0, min(1.0, 1.0 - fpr))))


def threshold_for_recall(scores: np.ndarray, y: np.ndarray, recall: float) -> float:
    pos = np.asarray(scores)[np.asarray(y) == 1]
    if len(pos) == 0:
        return float("inf")
    return float(np.quantile(pos, max(0.0, min(1.0, 1.0 - recall))))


def top_coefficients(model, feature_names: list[str], k: int = 30) -> list[dict]:
    if not HAVE_SKLEARN or not hasattr(model, "named_steps"):
        return []
    clf = model.named_steps.get("logisticregression")
    if clf is None:
        return []
    coef = clf.coef_[0]
    order = np.argsort(-np.abs(coef))[:k]
    return [{"feature": feature_names[i], "coef": round(float(coef[i]), 6)}
            for i in order if coef[i] != 0]


def single_feature_auc_table(x: np.ndarray, y: np.ndarray,
                             names: list[str], k: int = 50) -> list[dict]:
    mask = valid_binary_mask(y)
    out = []
    for i, name in enumerate(names):
        m = metrics.binary_metrics(x[mask, i], y[mask])
        auc = m.get("auc")
        if auc is not None and auc == auc:
            out.append({"feature": name, "auc": auc, "direction": m.get("direction")})
    out.sort(key=lambda r: r["auc"], reverse=True)
    return out[:k]


def _balanced_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    vals = []
    for cls in (0, 1):
        mask = y_true == cls
        vals.append(float((y_pred[mask] == cls).mean()) if mask.any() else 0.0)
    return float(np.mean(vals))


class _CentroidModel:
    def fit(self, x: np.ndarray, y: np.ndarray):
        self.mu = x.mean(axis=0)
        self.sd = x.std(axis=0) + 1e-8
        z = (x - self.mu) / self.sd
        self.c0 = z[y == 0].mean(axis=0)
        self.c1 = z[y == 1].mean(axis=0)
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mu) / self.sd
        d0 = np.linalg.norm(z - self.c0, axis=1)
        d1 = np.linalg.norm(z - self.c1, axis=1)
        return d0 - d1
