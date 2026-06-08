from __future__ import annotations

import numpy as np


def roc_auc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    mask = np.isfinite(scores) & (labels >= 0)
    scores, labels = scores[mask], labels[mask]
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    sr = ranks[labels == 1].sum()
    return float((sr - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def oriented_auc(scores, labels) -> tuple[float, str]:
    auc = roc_auc(scores, labels)
    if np.isnan(auc):
        return auc, "higher"
    if auc < 0.5:
        return 1.0 - auc, "lower"
    return auc, "higher"


def threshold_at_fpr(scores, labels, target_fpr: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    neg = scores[labels == 0]
    if len(neg) == 0:
        return float("inf")
    return float(np.quantile(neg, max(0.0, min(1.0, 1.0 - target_fpr))))


def binary_eval(pred, labels) -> dict:
    pred = np.asarray(pred, dtype=bool)
    labels = np.asarray(labels, dtype=np.int64)
    pos = labels == 1
    neg = labels == 0
    tp = int((pred & pos).sum())
    fp = int((pred & neg).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & neg).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": round(recall, 5),
        "fpr": round(fpr, 5),
        "precision": round(precision, 5),
        "f1": round(2 * precision * recall / max(1e-12, precision + recall), 5),
    }


def per_letter_rates(pred, letters) -> dict:
    pred = np.asarray(pred, dtype=bool)
    letters = np.asarray(letters)
    out = {}
    for letter in "ABCDEFG":
        m = letters == letter
        if m.any():
            out[letter] = {
                "n": int(m.sum()),
                "rate": round(float(pred[m].mean()), 5),
            }
    return out


def prompt_eval(rows: list[dict], row_pred: np.ndarray) -> dict:
    by_prompt: dict[int, list[int]] = {}
    attack_prompt: dict[int, bool] = {}
    for i, row in enumerate(rows):
        sid = int(row["sample_index"])
        by_prompt.setdefault(sid, []).append(i)
        attack_prompt[sid] = attack_prompt.get(sid, False) or row["letter"] in ("B", "D")
    blocked = {sid for sid, idxs in by_prompt.items() if bool(np.asarray(row_pred)[idxs].any())}
    attack_ids = {sid for sid, is_attack in attack_prompt.items() if is_attack}
    benign_ids = set(attack_prompt) - attack_ids
    return {
        "n_prompt": len(by_prompt),
        "n_attack_prompt": len(attack_ids),
        "n_benign_prompt": len(benign_ids),
        "block_rate": round(len(blocked & attack_ids) / max(1, len(attack_ids)), 5),
        "prompt_fpr": round(len(blocked & benign_ids) / max(1, len(benign_ids)), 5),
    }


