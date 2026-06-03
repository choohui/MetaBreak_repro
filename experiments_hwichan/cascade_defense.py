"""Tier-1 cascade (funnel) feasibility study — post-hoc on existing results.

Idea: use the cheap attention ``sink`` score not as a standalone detector but as a
**recall-oriented first-stage gate** that drops easy negatives, then apply a strong
hidden-state feature (``cos_to_D`` / ``hidden_norm`` / ``value_norm`` / ``output_norm``)
only to the survivors.

Why a funnel can help even though a pure AND-cascade can never *raise* recall above
the second stage alone: by removing many negatives for free at stage 1, the global
FPR budget (e.g. 1%) is spent over fewer surviving negatives, so stage 2 can run at a
more permissive threshold and recover TPR. We therefore measure, per (exp, pos):

  * stage-1 attack recall and **negative-removal rate** (workload reduction),
  * **end-to-end TPR@FPR=1%/5%** of the cascade vs the same feature standalone
    (FPR/TPR always computed over the *full* attack/negative pools), and
  * **Spearman correlation** between the sink gate and each stage-2 feature
    (low correlation ⇒ more cascade benefit).

This reads the already-extracted ``tokens.jsonl`` + ``features.npz`` and needs no GPU.

Run:
    cd repro_mb
    python experiments_hwichan/cascade_defense.py            # all exp{1,2} x pos{0,1}
    python experiments_hwichan/cascade_defense.py --out_dir experiments_hwichan/results/exp2_llama31_8b --pos_offset 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_hwichan.analyze_representations import load_features  # noqa: E402
from experiments_hwichan.common import CAT_SYSTEM, write_json  # noqa: E402
from experiments_hwichan.defense_thresholds import (  # noqa: E402
    _labels_from_rows,
    binary_metrics,
    roc_auc,
)

STAGE2_SCALARS = ["hidden_norm", "value_norm", "output_norm"]
TARGET_RECALLS = [0.99, 0.95]
HERE = Path(__file__).resolve().parent
DEFAULT_DIRS = [
    HERE / "results" / "exp1_llama31_8b",
    HERE / "results" / "exp2_llama31_8b",
]


# --------------------------------------------------------------------------- #
# small numpy helpers
# --------------------------------------------------------------------------- #


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank (tie-aware), mirroring the AUC ranker in defense_thresholds."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    s = x[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    v = ~(np.isnan(a) | np.isnan(b))
    a, b = a[v], b[v]
    if len(a) < 3:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def feature_matrix(rows, feat: str) -> np.ndarray:
    """[N, n_layers] for a per-layer scalar list feature in tokens.jsonl."""
    n_layers = min(len(r[feat]) for r in rows)
    return np.array(
        [[r[feat][l] for l in range(n_layers)] for r in rows], dtype=np.float64
    )


def cos_to_D_matrix(rows, hidden: np.ndarray) -> np.ndarray:
    """[N, n_hidden_layers] cosine of each token to the per-layer D centroid."""
    d_idx = [i for i, r in enumerate(rows) if r["category"] == CAT_SYSTEM]
    n_hl = hidden.shape[1]
    out = np.full((hidden.shape[0], n_hl), np.nan, dtype=np.float64)
    if not d_idx:
        return out
    d_cent = np.stack([hidden[d_idx, l, :].mean(axis=0) for l in range(n_hl)], axis=0)
    d_norm = np.linalg.norm(d_cent, axis=1)
    for l in range(n_hl):
        h = hidden[:, l, :]
        hn = np.linalg.norm(h, axis=1)
        denom = hn * d_norm[l]
        out[:, l] = np.where(denom > 0, (h @ d_cent[l]) / np.where(denom > 0, denom, 1), np.nan)
    return out


def best_layer_oriented(mat: np.ndarray, y: np.ndarray):
    """Pick the layer with highest |AUC-0.5|; return (layer, auc, sign).

    sign = +1 means higher value -> attack (already oriented); -1 means lower.
    """
    best = None
    for l in range(mat.shape[1]):
        s = mat[:, l]
        valid = ~np.isnan(s)
        if valid.sum() == 0 or len(np.unique(y[valid])) < 2:
            continue
        auc = roc_auc(s[valid], y[valid])
        oriented = auc if auc >= 0.5 else 1.0 - auc
        sign = 1 if auc >= 0.5 else -1
        if best is None or oriented > best[1]:
            best = (l, round(oriented, 5), sign)
    return best


def sink_layeragg(sink_mat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-layer z-score + own-sign orientation, then max over layers.

    A token-level "does any layer look attack-like" gate that is scale-free across
    depth. Returned scores are oriented so higher == more attack-like.
    """
    n_layers = sink_mat.shape[1]
    oriented = np.full_like(sink_mat, -np.inf, dtype=np.float64)
    for l in range(n_layers):
        s = sink_mat[:, l]
        mu, sd = np.nanmean(s), np.nanstd(s)
        if sd == 0 or np.isnan(sd):
            continue
        z = (s - mu) / sd
        auc = roc_auc(s, y)
        sign = 1 if auc >= 0.5 else -1
        oriented[:, l] = sign * z
    return oriented.max(axis=1)


# --------------------------------------------------------------------------- #
# cascade evaluation
# --------------------------------------------------------------------------- #


def cascade_tpr_at_fpr(
    s1: np.ndarray,
    s2: np.ndarray,
    y: np.ndarray,
    recall_target: float,
    fpr_targets=(0.01, 0.05),
):
    """Recall-gate on s1 (higher=attack), then threshold s2 (higher=attack).

    TPR/FPR are over the full attack/negative pools; tokens dropped at stage 1 are
    counted as predicted-negative. Returns gate stats + TPR at each fpr target.
    """
    pos = y == 1
    neg = y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())

    att1 = s1[pos]
    t1 = float(np.quantile(att1, 1.0 - recall_target))
    passed = s1 >= t1
    recall1 = float((att1 >= t1).mean())
    neg_removed = float((s1[neg] < t1).mean())

    # sweep stage-2 threshold over surviving, non-nan scores
    cand = s2[passed & ~np.isnan(s2)]
    out = {
        "stage1_threshold": round(t1, 6),
        "stage1_recall": round(recall1, 5),
        "neg_removed": round(neg_removed, 5),
        "frac_passed": round(float(passed.mean()), 5),
    }
    tpr_at = {ft: 0.0 for ft in fpr_targets}
    if cand.size:
        for t2 in np.unique(cand):
            pred = passed & ~np.isnan(s2) & (s2 >= t2)
            tp = int((pred & pos).sum())
            fp = int((pred & neg).sum())
            tpr = tp / n_pos if n_pos else 0.0
            fpr = fp / n_neg if n_neg else 0.0
            for ft in fpr_targets:
                if fpr <= ft and tpr > tpr_at[ft]:
                    tpr_at[ft] = tpr
    for ft in fpr_targets:
        out[f"tpr_at_fpr_{ft}"] = round(tpr_at[ft], 5)
    return out


def evaluate(out_dir: Path, pos_offset: int = 0) -> dict:
    rows, hidden = load_features(out_dir, pos_offset=pos_offset)
    if hidden.size == 0:
        raise SystemExit(f"No features for pos_offset={pos_offset} in {out_dir}")
    y_all = _labels_from_rows(rows)
    mask = y_all >= 0
    y = y_all[mask]
    if len(np.unique(y)) < 2:
        raise SystemExit("Need both attack and negative tokens.")

    # ---- stage-1 sink gate (best single layer + cross-layer aggregate) ----
    sink_mat = feature_matrix(rows, "sink")[mask]
    s_layer, s_auc, s_sign = best_layer_oriented(sink_mat, y)
    sink_best = s_sign * sink_mat[:, s_layer]
    sink_agg = sink_layeragg(sink_mat, y)
    sink_agg_auc = round(roc_auc(sink_agg, y), 5)

    stage1_options = {
        f"sink_layer{s_layer}": {"score": sink_best, "auc": s_auc, "desc": f"sink @layer {s_layer}"},
        "sink_layeragg": {"score": sink_agg, "auc": sink_agg_auc, "desc": "sink max-over-layers (z, oriented)"},
    }

    # ---- stage-2 feature matrices (masked), oriented + best layer ----
    feat_mats = {f: feature_matrix(rows, f)[mask] for f in STAGE2_SCALARS}
    feat_mats["cos_to_D"] = cos_to_D_matrix(rows, hidden)[mask]
    stage2 = {}
    for f, mat in feat_mats.items():
        bl = best_layer_oriented(mat, y)
        if bl is None:
            continue
        layer, auc, sign = bl
        score = sign * mat[:, layer]
        standalone = binary_metrics(mat[:, layer], y)  # auto-orients internally
        stage2[f] = {
            "layer": layer,
            "auc": auc,
            "score": score,
            "standalone_tpr_at_fpr_0.01": standalone.get("tpr_at_fpr_0.01"),
            "standalone_tpr_at_fpr_0.05": standalone.get("tpr_at_fpr_0.05"),
        }

    # ---- cascade grid: each stage1 option x each stage2 feature x recall ----
    grid = []
    for s1_name, s1 in stage1_options.items():
        for f, st2 in stage2.items():
            corr = spearman(s1["score"], st2["score"])
            for r in TARGET_RECALLS:
                casc = cascade_tpr_at_fpr(s1["score"], st2["score"], y, r)
                grid.append(
                    {
                        "stage1": s1_name,
                        "stage1_auc": s1["auc"],
                        "stage2": f,
                        "stage2_layer": st2["layer"],
                        "stage2_auc": st2["auc"],
                        "recall_target": r,
                        "corr_sink_feat": corr,
                        "standalone_tpr_at_fpr_0.01": st2["standalone_tpr_at_fpr_0.01"],
                        "standalone_tpr_at_fpr_0.05": st2["standalone_tpr_at_fpr_0.05"],
                        **casc,
                    }
                )

    report = {
        "out_dir": str(out_dir),
        "pos_offset": pos_offset,
        "n_attack": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "stage1_options": {k: {"auc": v["auc"], "desc": v["desc"]} for k, v in stage1_options.items()},
        "stage2_standalone": {
            f: {k: v[k] for k in ("layer", "auc", "standalone_tpr_at_fpr_0.01", "standalone_tpr_at_fpr_0.05")}
            for f, v in stage2.items()
        },
        "grid": grid,
    }
    write_json(out_dir / f"cascade_report_pos{pos_offset}.json", report)
    _write_markdown(out_dir / f"cascade_report_pos{pos_offset}.md", report)
    return report


def _write_markdown(path: Path, report: dict) -> None:
    L = ["# Cascade (sink-gate funnel) feasibility report", ""]
    L.append(f"- dir: `{report['out_dir']}`  pos_offset: `{report['pos_offset']}`")
    L.append(f"- attack tokens (A∪B): **{report['n_attack']}**  |  negative (C∪E): **{report['n_negative']}**")
    L.append("")
    L.append("**Stage-1 gate options (oriented AUC):** "
             + ", ".join(f"`{k}`={v['auc']} ({v['desc']})" for k, v in report["stage1_options"].items()))
    L.append("")
    L.append("Columns: standalone = stage-2 feature alone (no gate). cascade = sink-gate@recall then stage-2, "
             "TPR/FPR over the full pools. neg_removed = fraction of negatives dropped by the gate. "
             "corr = Spearman(sink-gate, stage-2 feature).")
    L.append("")
    L.append("| stage1 | stage2 (layer) | recall | neg_removed | standalone TPR@1% / 5% | cascade TPR@1% / 5% | corr |")
    L.append("|---|---|---|---|---|---|---|")
    for g in report["grid"]:
        L.append(
            f"| {g['stage1']} (AUC {g['stage1_auc']}) | {g['stage2']} (L{g['stage2_layer']}, AUC {g['stage2_auc']}) | "
            f"{g['recall_target']} | {g['neg_removed']} | "
            f"{g['standalone_tpr_at_fpr_0.01']} / {g['standalone_tpr_at_fpr_0.05']} | "
            f"**{g['tpr_at_fpr_0.01']} / {g['tpr_at_fpr_0.05']}** | {g['corr_sink_feat']} |"
        )
    L.append("")
    L.append("Read: cascade helps when **cascade TPR ≥ standalone TPR** at the same FPR while "
             "**neg_removed** is large and **stage-1 recall ≈ target** (gate isn't dropping attacks). "
             "Low **corr** indicates the two signals are independent ⇒ more headroom for the funnel.")
    path.write_text("\n".join(L), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=None, help="single results dir; default = both exp dirs")
    p.add_argument("--pos_offset", type=int, default=None, choices=[0, 1],
                   help="single pos_offset; default = both 0 and 1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dirs = [Path(args.out_dir)] if args.out_dir else DEFAULT_DIRS
    offsets = [args.pos_offset] if args.pos_offset is not None else [0, 1]
    for d in dirs:
        for off in offsets:
            try:
                rep = evaluate(d, pos_offset=off)
            except SystemExit as e:
                print(f"[cascade] {d} pos{off} skipped: {e}")
                continue
            print(f"\n===== {d.name} pos{off}  (attack={rep['n_attack']} neg={rep['n_negative']}) =====")
            print(f"  stage1: " + ", ".join(f"{k} AUC={v['auc']}" for k, v in rep["stage1_options"].items()))
            for g in rep["grid"]:
                print(
                    f"  {g['stage1']:>14} -> {g['stage2']:<11} L{g['stage2_layer']:<2} "
                    f"r={g['recall_target']} negrm={g['neg_removed']:.3f} "
                    f"base@1/5%={g['standalone_tpr_at_fpr_0.01']}/{g['standalone_tpr_at_fpr_0.05']} "
                    f"casc@1/5%={g['tpr_at_fpr_0.01']}/{g['tpr_at_fpr_0.05']} corr={g['corr_sink_feat']}"
                )


if __name__ == "__main__":
    main()
