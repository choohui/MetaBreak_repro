"""Tier-2 honest cascade evaluation — full-sequence E coverage + held-out thresholds.

Unlike the Tier-1 study (``cascade_defense.py``), this script:

  1. expects a **full-coverage** extraction where *every* ordinary content token is
     labelled E (run ``extract_representations.py --ordinary -1``), so the negative
     pool reflects a realistic "scan the whole prompt" deployment rather than a
     sparse 4-6 token sample; and
  2. fits **all** thresholds (and selects the sink layer + stage-2 feature/layer) on
     a **train split**, then reports TPR@FPR on a disjoint **test split**. The split
     is *grouped by base prompt* so the malicious / mimicked / benign variants of the
     same question never straddle train and test (they are near-duplicates).

Stage 1 = sink recall-gate (drop easy negatives). Stage 2 = best hidden-state scalar
feature. End-to-end TPR/FPR are always over the full attack / negative pools on test.
``cos_to_D`` is included only if a ``features.npz`` hidden cube is present (its D
centroid is then estimated on the **train** split only).

Run (after the Tier-2 extraction, see the module docstring of extract step):
    cd repro_mb
    python experiments_hwichan/cascade_eval_tier2.py \
        --out_dir experiments_hwichan/results/exp2_llama31_8b_tier2 --pos_offset 0
    # or omit --out_dir to scan the default *_tier2 dirs / both pos offsets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_hwichan.common import (  # noqa: E402
    ATTACK_CATS,
    NEGATIVE_CATS,
    read_jsonl,
    write_json,
)
from experiments_hwichan.defense_thresholds import roc_auc  # noqa: E402

STAGE2_SCALARS = ["hidden_norm", "value_norm", "output_norm"]
TARGET_RECALLS = [0.99, 0.95]
FPR_BUDGETS = [0.01, 0.05]
HERE = Path(__file__).resolve().parent
DEFAULT_DIRS = [
    HERE / "results" / "exp1_llama31_8b_tier2",
    HERE / "results" / "exp2_llama31_8b_tier2",
]


# --------------------------------------------------------------------------- #
# data loading / split
# --------------------------------------------------------------------------- #


def load_rows(out_dir: Path, pos_offset: int):
    rows = [r for r in read_jsonl(out_dir / "tokens.jsonl") if r["pos_offset"] == pos_offset]
    if not rows:
        raise SystemExit(f"No tokens for pos_offset={pos_offset} in {out_dir}")
    return rows


def labels_of(rows) -> np.ndarray:
    y = np.full(len(rows), -1, dtype=int)
    for i, r in enumerate(rows):
        if r["category"] in ATTACK_CATS:
            y[i] = 1
        elif r["category"] in NEGATIVE_CATS:
            y[i] = 0
    return y


def grouped_split(rows, test_frac: float, seed: int):
    """Train/test boolean masks, grouped by base prompt id (no variant leakage)."""
    groups = sorted({str(r.get("prompt_idx")) for r in rows})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    n_test = max(1, int(round(test_frac * len(groups))))
    test_groups = {groups[i] for i in perm[:n_test]}
    is_test = np.array([str(r.get("prompt_idx")) in test_groups for r in rows])
    return ~is_test, is_test


def feature_matrix(rows, feat: str) -> np.ndarray:
    n_layers = min(len(r[feat]) for r in rows)
    return np.array([[r[feat][l] for l in range(n_layers)] for r in rows], dtype=np.float64)


# --------------------------------------------------------------------------- #
# threshold fitting (TRAIN) and evaluation (TEST)
# --------------------------------------------------------------------------- #


def orient_and_pick_layer(mat: np.ndarray, y: np.ndarray, train: np.ndarray):
    """Choose the layer with best train AUC; return (layer, sign, train_auc).

    sign=+1 means higher value -> attack. Selection uses TRAIN only.
    """
    best = None
    for l in range(mat.shape[1]):
        s = mat[train, l]
        valid = ~np.isnan(s)
        ytr = y[train][valid]
        if valid.sum() == 0 or len(np.unique(ytr)) < 2:
            continue
        auc = roc_auc(s[valid], ytr)
        oriented = auc if auc >= 0.5 else 1.0 - auc
        sign = 1 if auc >= 0.5 else -1
        if best is None or oriented > best[2]:
            best = (l, sign, round(oriented, 5))
    return best


def fit_stage1_threshold(s1_tr: np.ndarray, ytr: np.ndarray, recall_target: float) -> float:
    att = s1_tr[ytr == 1]
    return float(np.quantile(att, 1.0 - recall_target))


def fit_stage2_threshold(s2_tr, ytr, fpr_budget, passed_tr=None) -> float:
    """Largest-TPR threshold whose FPR (over ALL train negatives) <= budget.

    If ``passed_tr`` (stage-1 survival mask) is given, only survivors can be flagged
    positive — this is the cascade fit; otherwise it is the standalone fit.
    """
    pos = ytr == 1
    neg = ytr == 0
    n_neg = max(1, int(neg.sum()))
    gate = passed_tr if passed_tr is not None else np.ones(len(ytr), dtype=bool)
    usable = gate & ~np.isnan(s2_tr)
    cand = np.unique(s2_tr[usable])
    best_t, best_tpr = None, -1.0
    for t in cand:
        pred = usable & (s2_tr >= t)
        fpr = (pred & neg).sum() / n_neg
        if fpr <= fpr_budget:
            tpr = (pred & pos).sum() / max(1, int(pos.sum()))
            if tpr > best_tpr:
                best_tpr, best_t = tpr, float(t)
    if best_t is None:  # nothing meets budget -> strictest threshold
        best_t = float(cand.max()) if cand.size else float("inf")
    return best_t


def eval_on(s1, s2, y, t1, t2, cascade: bool):
    pos = y == 1
    neg = y == 0
    n_pos, n_neg = max(1, int(pos.sum())), max(1, int(neg.sum()))
    gate = (s1 >= t1) if cascade else np.ones(len(y), dtype=bool)
    pred = gate & ~np.isnan(s2) & (s2 >= t2)
    res = {
        "tpr": round(float((pred & pos).sum() / n_pos), 5),
        "fpr": round(float((pred & neg).sum() / n_neg), 5),
    }
    if cascade:
        res["stage1_recall"] = round(float((s1[pos] >= t1).mean()), 5)
        res["neg_removed"] = round(float((s1[neg] < t1).mean()), 5)
    return res


def _rankdata(x):
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


def spearman(a, b) -> float:
    v = ~(np.isnan(a) | np.isnan(b))
    a, b = a[v], b[v]
    if len(a) < 3:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


# --------------------------------------------------------------------------- #
# main evaluation
# --------------------------------------------------------------------------- #


def evaluate(out_dir: Path, pos_offset: int, test_frac: float, seed: int) -> dict:
    rows = load_rows(out_dir, pos_offset)
    y = labels_of(rows)
    keep = y >= 0
    rows = [r for i, r in enumerate(rows) if keep[i]]
    y = y[keep]
    train, test = grouped_split(rows, test_frac, seed)
    if y[train].sum() == 0 or y[test].sum() == 0 or (y[train] == 0).sum() == 0 or (y[test] == 0).sum() == 0:
        raise SystemExit("Split left a side without both classes; adjust test_frac/seed.")

    # ---- stage-2 candidate features ----
    # Tier-2 uses the scalar features only (sink / hidden_norm / value_norm /
    # output_norm). cos_to_D needs the [N, L+1, dim] hidden cube, which the full-E
    # extraction deliberately omits (--no_hidden); evaluate cos_to_D via Tier-1
    # (cascade_defense.py), which keeps the cube on the sparse-E sample.
    feat_mats = {f: feature_matrix(rows, f) for f in STAGE2_SCALARS}

    # ---- stage-1 sink: pick layer on TRAIN ----
    sink_mat = feature_matrix(rows, "sink")
    s1_pick = orient_and_pick_layer(sink_mat, y, train)
    s1_layer, s1_sign, s1_auc = s1_pick
    s1 = s1_sign * sink_mat[:, s1_layer]

    # ---- per stage-2 feature: pick layer on TRAIN, fit thresholds on TRAIN ----
    results = []
    for f, mat in feat_mats.items():
        pick = orient_and_pick_layer(mat, y, train)
        if pick is None:
            continue
        f_layer, f_sign, f_auc = pick
        s2 = f_sign * mat[:, f_layer]
        corr = spearman(s1[test], s2[test])
        for budget in FPR_BUDGETS:
            # standalone (no gate): threshold fit on train, eval on test
            t2_std = fit_stage2_threshold(s2[train], y[train], budget, passed_tr=None)
            std = eval_on(s1, s2, y, -np.inf, t2_std, cascade=False)
            # take std restricted to test
            std_test = eval_on(s1[test], s2[test], y[test], -np.inf, t2_std, cascade=False)
            for recall in TARGET_RECALLS:
                t1 = fit_stage1_threshold(s1[train], y[train], recall)
                passed_tr = s1[train] >= t1
                t2_casc = fit_stage2_threshold(s2[train], y[train], budget, passed_tr=passed_tr)
                casc_test = eval_on(s1[test], s2[test], y[test], t1, t2_casc, cascade=True)
                results.append(
                    {
                        "stage2": f,
                        "stage2_layer": f_layer,
                        "stage2_train_auc": f_auc,
                        "fpr_budget": budget,
                        "recall_target": recall,
                        "corr_sink_feat_test": corr,
                        "standalone_test_tpr": std_test["tpr"],
                        "standalone_test_fpr": std_test["fpr"],
                        "cascade_test_tpr": casc_test["tpr"],
                        "cascade_test_fpr": casc_test["fpr"],
                        "test_stage1_recall": casc_test["stage1_recall"],
                        "test_neg_removed": casc_test["neg_removed"],
                        "delta_tpr": round(casc_test["tpr"] - std_test["tpr"], 5),
                    }
                )

    report = {
        "out_dir": str(out_dir),
        "pos_offset": pos_offset,
        "test_frac": test_frac,
        "seed": seed,
        "n_total": len(rows),
        "n_attack": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_train_groups": int(np.unique([str(r.get("prompt_idx")) for r, m in zip(rows, train) if m]).size),
        "stage1_sink": {"layer": s1_layer, "sign": s1_sign, "train_auc": s1_auc},
        "results": results,
    }
    write_json(out_dir / f"cascade_tier2_pos{pos_offset}.json", report)
    _write_md(out_dir / f"cascade_tier2_pos{pos_offset}.md", report)
    return report


def _write_md(path: Path, rep: dict) -> None:
    L = ["# Tier-2 honest cascade report (held-out thresholds, full-E coverage)", ""]
    L.append(f"- dir: `{rep['out_dir']}`  pos_offset: `{rep['pos_offset']}`  "
             f"(test_frac={rep['test_frac']}, seed={rep['seed']})")
    L.append(f"- tokens: attack(A∪B)=**{rep['n_attack']}**, negative(C∪E)=**{rep['n_negative']}** "
             f"(full content-token coverage)")
    s1 = rep["stage1_sink"]
    L.append(f"- stage-1 sink gate: layer `{s1['layer']}` (sign {s1['sign']}, train AUC {s1['train_auc']})")
    L.append("")
    L.append("All thresholds + layer/feature selection fit on TRAIN; numbers below are on the held-out TEST split. "
             "TPR/FPR over the full attack/negative pools.")
    L.append("")
    L.append("| stage2 (layer) | FPR budget | recall | neg_removed | standalone TPR (fpr) | cascade TPR (fpr) | ΔTPR | corr |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rep["results"]:
        flag = " ✅" if r["delta_tpr"] > 0 else ""
        L.append(
            f"| {r['stage2']} (L{r['stage2_layer']}, AUC {r['stage2_train_auc']}) | {r['fpr_budget']} | "
            f"{r['recall_target']} | {r['test_neg_removed']} | "
            f"{r['standalone_test_tpr']} ({r['standalone_test_fpr']}) | "
            f"**{r['cascade_test_tpr']}** ({r['cascade_test_fpr']}) | {r['delta_tpr']}{flag} | "
            f"{r['corr_sink_feat_test']} |"
        )
    L.append("")
    L.append("ΔTPR>0 means the sink funnel beats the stage-2 feature alone at the same FPR budget, on held-out data. "
             "Watch `cascade_test_fpr` ≤ budget (else the train threshold did not transfer) and "
             "`test_stage1_recall` ≈ target (else the gate is dropping real attacks).")
    path.write_text("\n".join(L), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=None, help="single *_tier2 results dir; default = both")
    p.add_argument("--pos_offset", type=int, default=None, choices=[0, 1])
    p.add_argument("--test_frac", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dirs = [Path(args.out_dir)] if args.out_dir else DEFAULT_DIRS
    offsets = [args.pos_offset] if args.pos_offset is not None else [0, 1]
    for d in dirs:
        for off in offsets:
            try:
                rep = evaluate(d, off, args.test_frac, args.seed)
            except SystemExit as e:
                print(f"[tier2] {d} pos{off} skipped: {e}")
                continue
            print(f"\n===== {d.name} pos{off}  attack={rep['n_attack']} neg={rep['n_negative']} "
                  f"sink L{rep['stage1_sink']['layer']} (AUC {rep['stage1_sink']['train_auc']}) =====")
            for r in rep["results"]:
                flag = "  <== gain" if r["delta_tpr"] > 0 else ""
                print(
                    f"  {r['stage2']:<11} L{r['stage2_layer']:<2} budget={r['fpr_budget']} r={r['recall_target']} "
                    f"negrm={r['test_neg_removed']:.3f} std={r['standalone_test_tpr']:.3f}({r['standalone_test_fpr']:.3f}) "
                    f"casc={r['cascade_test_tpr']:.3f}({r['cascade_test_fpr']:.3f}) dTPR={r['delta_tpr']:+.3f} "
                    f"corr={r['corr_sink_feat_test']}{flag}"
                )


if __name__ == "__main__":
    main()
