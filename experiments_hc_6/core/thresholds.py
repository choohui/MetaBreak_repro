from __future__ import annotations

import numpy as np

from .labels import binary_label
from .metrics import binary_eval, oriented_auc, threshold_at_fpr


def labels_from_rows(rows: list[dict]) -> np.ndarray:
    return np.asarray([binary_label(r["letter"]) for r in rows], dtype=int)


def orient_scores(scores: np.ndarray, direction: str) -> np.ndarray:
    return -scores if direction == "lower" else scores


def predict_rule(x: np.ndarray, feature_names: list[str], rule: dict) -> np.ndarray:
    idx_by_name = {n: i for i, n in enumerate(feature_names)}
    preds = []
    for term in rule.get("terms", []):
        j = idx_by_name[term["feature"]]
        s = orient_scores(x[:, j], term["direction"])
        preds.append(s >= float(term["threshold"]))
    if not preds:
        return np.zeros(x.shape[0], dtype=bool)
    mat = np.vstack(preds)
    kind = rule.get("kind", "single")
    if kind == "and2" or kind == "two_of_k":
        return mat.sum(axis=0) >= min(2, mat.shape[0])
    return mat.any(axis=0)


def _fit_single_terms(x: np.ndarray, names: list[str], y: np.ndarray, train: np.ndarray,
                      fpr: float, max_features: int = 80) -> list[dict]:
    terms = []
    for j, name in enumerate(names):
        yy = y[train]
        if len(np.unique(yy[yy >= 0])) < 2:
            continue
        auc, direction = oriented_auc(x[train, j], yy)
        if np.isnan(auc):
            continue
        os = orient_scores(x[:, j], direction)
        thr = threshold_at_fpr(os[train], yy, fpr)
        terms.append({
            "feature": name,
            "direction": direction,
            "threshold": thr,
            "train_auc": round(float(auc), 5),
        })
    terms.sort(key=lambda r: r["train_auc"], reverse=True)
    return terms[:max_features]


def select_threshold_rules(x: np.ndarray, names: list[str], rows: list[dict], fpr_targets: list[float],
                           max_terms: int = 3) -> dict:
    y = labels_from_rows(rows)
    split = np.asarray([r["split"] for r in rows])
    train = (split == "train") & (y >= 0)
    val = (split == "val") & (y >= 0)
    out = {"selected": {}, "candidates": {}}
    for fpr in fpr_targets:
        terms = _fit_single_terms(x, names, y, train, fpr)
        candidates = []
        for term in terms:
            rule = {"kind": "single", "terms": [term], "fpr_target": fpr}
            pred = predict_rule(x[val], names, rule)
            ev = binary_eval(pred, y[val])
            candidates.append({**rule, "val_eval": ev})
        for k in range(2, max_terms + 1):
            chosen = terms[:k]
            if len(chosen) == k:
                for kind in ("or", "two_of_k"):
                    rule = {"kind": kind, "terms": chosen, "fpr_target": fpr}
                    pred = predict_rule(x[val], names, rule)
                    ev = binary_eval(pred, y[val])
                    candidates.append({**rule, "val_eval": ev})
        # Branch proxy: best B-like + best D-like feature against benign.
        for attack_letter in ("B", "D"):
            yb = np.asarray([1 if r["letter"] == attack_letter else (0 if r["letter"] in "CEFG" else -1)
                             for r in rows], dtype=int)
            bterms = _fit_single_terms(x, names, yb, train & (yb >= 0), fpr, max_features=5)
            if bterms:
                rule = {"kind": f"branch_{attack_letter}", "terms": [bterms[0]], "fpr_target": fpr}
                pred = predict_rule(x[val], names, rule)
                ev = binary_eval(pred, y[val])
                candidates.append({**rule, "val_eval": ev})
        # Choose highest recall under budget, then lower FPR, then higher F1.
        candidates.sort(key=lambda r: (
            r["val_eval"]["fpr"] <= fpr,
            r["val_eval"]["recall"],
            -r["val_eval"]["fpr"],
            r["val_eval"]["f1"],
        ), reverse=True)
        out["candidates"][str(fpr)] = candidates[:50]
        out["selected"][str(fpr)] = candidates[0] if candidates else None
    return out


