"""Threshold-based defense feasibility study.

Positive class = attack tokens (A mimicry-regular  u  B malicious-special).
Negative class = benign-special (C)  u  ordinary regular (E).

For every candidate scalar feature, at every layer, we ask: can a single
threshold separate attack tokens from benign ones?  We report ROC-AUC,
the Youden-optimal threshold, and TPR at fixed FPR (1% / 5%).

Candidate features
  * ``sink``         attention sink score (Attention Sinks as Internal Signals)
  * ``hidden_norm``  hidden-state L2 norm (massive-activation signal)
  * ``value_norm``   ||V|| value-vector norm (computationally-active sink)
  * ``output_norm``  ||O|| attention-output norm
  * ``cos_to_D``     cosine of the token's hidden state to the system-special (D)
                     centroid at that layer ("does this token look like a real
                     special token internally?")

All metrics are computed in pure NumPy (no sklearn dependency).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments_hwichan.analyze_representations import load_features
from experiments_hwichan.common import (
    ATTACK_CATS,
    CAT_SYSTEM,
    NEGATIVE_CATS,
    write_json,
)

SCALAR_FEATURES = ["sink", "hidden_norm", "value_norm", "output_norm"]


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), tie-aware."""
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s = scores[order]
    # average ranks for ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    sum_pos = ranks[pos].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """AUC + best Youden threshold + TPR@FPR. Auto-orients so higher=attack."""
    valid = ~np.isnan(scores)
    scores, labels = scores[valid], labels[valid]
    if len(np.unique(labels)) < 2:
        return {"auc": None, "n": int(len(labels))}
    auc = roc_auc(scores, labels)
    direction = "higher_is_attack"
    if auc < 0.5:
        scores = -scores
        auc = 1.0 - auc
        direction = "lower_is_attack"

    pos = scores[labels == 1]
    neg = scores[labels == 0]
    thresholds = np.unique(scores)
    best_j, best_t, best_tpr, best_fpr = -1.0, None, None, None
    tpr_at = {0.01: 0.0, 0.05: 0.0}
    thr_at = {0.01: None, 0.05: None}
    for t in thresholds:
        tpr = float((pos >= t).mean())
        fpr = float((neg >= t).mean())
        j = tpr - fpr
        if j > best_j:
            best_j, best_t, best_tpr, best_fpr = j, float(t), tpr, fpr
        for target in (0.01, 0.05):
            if fpr <= target and tpr > tpr_at[target]:
                tpr_at[target] = tpr
                thr_at[target] = float(t)
    return {
        "auc": round(float(auc), 5),
        "direction": direction,
        "youden_threshold": best_t,
        "youden_tpr": round(float(best_tpr), 5),
        "youden_fpr": round(float(best_fpr), 5),
        "tpr_at_fpr_0.01": round(tpr_at[0.01], 5),
        "thr_at_fpr_0.01": thr_at[0.01],
        "tpr_at_fpr_0.05": round(tpr_at[0.05], 5),
        "thr_at_fpr_0.05": thr_at[0.05],
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
    }


def _labels_from_rows(rows) -> np.ndarray:
    y = np.full(len(rows), -1, dtype=int)
    for i, r in enumerate(rows):
        if r["category"] in ATTACK_CATS:
            y[i] = 1
        elif r["category"] in NEGATIVE_CATS:
            y[i] = 0
    return y


def evaluate(out_dir: Path, pos_offset: int = 0) -> dict:
    rows, hidden = load_features(out_dir, pos_offset=pos_offset)
    y = _labels_from_rows(rows)
    mask = y >= 0
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        raise SystemExit("Need both attack and negative tokens for the defense study.")

    results: dict[str, list[dict]] = {}

    # scalar per-layer features from tokens.jsonl lists
    for feat in SCALAR_FEATURES:
        lengths = {len(r[feat]) for r in rows}
        n_layers = min(lengths)
        feat_rows = []
        for l in range(n_layers):
            scores = np.array([r[feat][l] for r in rows], dtype=np.float64)
            m = binary_metrics(scores[mask], y[mask])
            m["layer"] = l
            feat_rows.append(m)
        results[feat] = feat_rows

    # cos-to-D centroid feature (needs hidden cube + D centroid per layer)
    d_idx = [i for i, r in enumerate(rows) if r["category"] == CAT_SYSTEM]
    if d_idx:
        n_hl = hidden.shape[1]
        d_centroids = np.stack(
            [hidden[d_idx, l, :].mean(axis=0) for l in range(n_hl)], axis=0
        )
        d_norm = np.linalg.norm(d_centroids, axis=1)  # [n_hl]
        cos_rows = []
        for l in range(n_hl):
            h = hidden[:, l, :]
            hn = np.linalg.norm(h, axis=1)
            denom = hn * d_norm[l]
            cos = np.where(denom > 0, (h @ d_centroids[l]) / np.where(denom > 0, denom, 1), np.nan)
            m = binary_metrics(cos[mask], y[mask])
            m["layer"] = l
            cos_rows.append(m)
        results["cos_to_D"] = cos_rows

    # best (feature, layer) overall and per feature
    def best_of(feat_rows):
        scored = [r for r in feat_rows if r.get("auc") is not None]
        return max(scored, key=lambda r: r["auc"]) if scored else None

    best_per_feature = {f: best_of(rs) for f, rs in results.items()}
    overall = None
    for f, b in best_per_feature.items():
        if b and (overall is None or b["auc"] > overall[1]["auc"]):
            overall = (f, b)

    report = {
        "pos_offset": pos_offset,
        "n_attack": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "per_feature": results,
        "best_per_feature": best_per_feature,
        "best_overall": {"feature": overall[0], **overall[1]} if overall else None,
    }
    write_json(out_dir / "defense_report.json", report)
    _write_markdown(out_dir / "defense_report.md", report)
    print(
        "[defense] best overall:",
        (overall[0], overall[1]["layer"], overall[1]["auc"]) if overall else None,
    )
    return report


def _write_markdown(path: Path, report: dict) -> None:
    lines = ["# Threshold-defense feasibility report", ""]
    lines.append(f"- pos_offset: `{report['pos_offset']}`")
    lines.append(f"- attack tokens (A u B): **{report['n_attack']}**")
    lines.append(f"- negative tokens (C u E): **{report['n_negative']}**")
    lines.append("")
    bo = report.get("best_overall")
    if bo:
        lines.append(
            f"**Best single threshold:** feature=`{bo['feature']}` "
            f"layer=`{bo['layer']}` AUC=`{bo['auc']}` "
            f"({bo['direction']}); TPR@FPR1%=`{bo['tpr_at_fpr_0.01']}`, "
            f"TPR@FPR5%=`{bo['tpr_at_fpr_0.05']}`."
        )
        lines.append("")
    lines.append("## Best layer per feature")
    lines.append("")
    lines.append("| feature | layer | AUC | direction | TPR@FPR1% | TPR@FPR5% | Youden thr |")
    lines.append("|---|---|---|---|---|---|---|")
    for f, b in report["best_per_feature"].items():
        if not b:
            lines.append(f"| {f} | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {f} | {b['layer']} | {b['auc']} | {b['direction']} | "
            f"{b['tpr_at_fpr_0.01']} | {b['tpr_at_fpr_0.05']} | "
            f"{b.get('youden_threshold')} |"
        )
    lines.append("")
    lines.append(
        "Interpretation: an AUC near 1.0 means that feature, at that layer, "
        "cleanly separates attack tokens (mimicked/malicious) from benign special "
        "tokens and ordinary text — i.e. a deployable threshold defense. Compare "
        "`sink` (attention-only) against the hidden-state features "
        "(`hidden_norm`, `value_norm`, `cos_to_D`) to judge which signal is stronger."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pos_offset", type=int, default=0, choices=[0, 1])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(Path(args.out_dir), pos_offset=args.pos_offset)


if __name__ == "__main__":
    main()
