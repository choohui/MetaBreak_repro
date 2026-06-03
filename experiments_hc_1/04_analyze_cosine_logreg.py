"""Stage 04 (Main.md §2.3) — logistic-regression probe + cosine analysis.

For each ``pos_offset`` and each layer:
  * logistic-regression probe on the FULL hidden vector
    (attack B u D = 1 vs benign C u E u F u G = 0), 5-fold CV ROC-AUC + bacc;
  * cosine for the pairs (A,B),(A,D),(A,G),(B,C),(B,D),(B,F) computed BOTH
    (a) centroid-vs-centroid and (b) as a per-prompt distribution
        (per-prompt mean of X vs the global centroid of Y).

Also persists the A (reference) centroid per layer -> consumed by stages 05/06
for the ``cos_to_ref`` signal.

Outputs (under ``out_dir/pos{offset}/``):
    representation_metrics.json / .csv
    cosine_pairs.json
    ref_centroids.npz          - A-centroid per layer (for cos_to_ref)
    pca_coords.npz             - 2D PCA at the best probe layer
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import ExpConfig, config_from_args, make_stage_parser  # noqa: E402
from core import io, metrics  # noqa: E402
from core.labels import (  # noqa: E402
    CAT_A, LETTER_TO_CAT, NEGATIVE_CATS, POSITIVE_CATS, ALL_CATEGORIES,
)

PAIRS = [("A", "B"), ("A", "D"), ("A", "G"), ("B", "C"), ("B", "D"), ("B", "F")]
_Qs = [0.1, 0.25, 0.5, 0.75, 0.9]


def _dist_stats(vals: np.ndarray) -> dict:
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return {"n": 0, "mean": None}
    return {
        "n": int(vals.size),
        "mean": round(float(vals.mean()), 5),
        "std": round(float(vals.std()), 5),
        "quantiles": {str(q): round(float(np.quantile(vals, q)), 5) for q in _Qs},
    }


def _analyze_offset(cfg: ExpConfig, rows: list[dict], hidden: np.ndarray, offset: int) -> dict:
    sub = [r for r in rows if r["pos_offset"] == offset]
    if not sub:
        print(f"[04] pos_offset={offset}: no rows, skipping")
        return {}
    ridx = np.array([r["row_id"] for r in sub])
    H = hidden[ridx].astype(np.float32)            # [n, L+1, dim]
    n, n_layers, dim = H.shape
    cats = [r["category"] for r in sub]
    samples = [r["sample_index"] for r in sub]

    cat_rows: dict[str, list[int]] = {c: [] for c in ALL_CATEGORIES}
    for i, c in enumerate(cats):
        cat_rows[c].append(i)
    # per-prompt groups for the cosine "distribution" method
    group_rows: dict[tuple, list[int]] = {}
    for i, (c, s) in enumerate(zip(cats, samples)):
        group_rows.setdefault((c, s), []).append(i)

    # defense labels for the probe
    y = np.array([1 if c in POSITIVE_CATS else (0 if c in NEGATIVE_CATS else -1)
                  for c in cats])
    probe_mask = y >= 0

    per_layer: list[dict] = []
    cosine_pairs: dict[str, dict] = {f"{a}-{b}": {"centroid": [], "per_prompt": []}
                                     for a, b in PAIRS}
    ref_centroids = np.zeros((n_layers, dim), dtype=np.float32)
    best_layer, best_auc = 0, -1.0

    for l in range(n_layers):
        Hl = H[:, l, :]
        # reference (A) centroid
        if cat_rows[CAT_A]:
            ref_centroids[l] = Hl[cat_rows[CAT_A]].mean(axis=0)

        row = {"layer": l}
        # per-category mean norm
        for c in ALL_CATEGORIES:
            if cat_rows[c]:
                row[f"{c[0]}__mean_norm"] = round(
                    float(np.linalg.norm(Hl[cat_rows[c]], axis=1).mean()), 5)

        # logreg probe
        if probe_mask.sum() > 4:
            pr = metrics.probe_layer(Hl[probe_mask], y[probe_mask])
            row["probe_auc"] = pr["auc"]
            row["probe_balanced_acc"] = pr["balanced_acc"]
            row["probe_method"] = pr["method"]
            if pr["auc"] == pr["auc"] and pr["auc"] > best_auc:
                best_auc, best_layer = pr["auc"], l

        # cosine pairs
        for a, b in PAIRS:
            ca, cb = LETTER_TO_CAT[a], LETTER_TO_CAT[b]
            key = f"{a}-{b}"
            if cat_rows[ca] and cat_rows[cb]:
                va = Hl[cat_rows[ca]].mean(axis=0)
                vb = Hl[cat_rows[cb]].mean(axis=0)
                cval = round(metrics.cosine(va, vb), 5)
                row[f"cos__{a}_{b}"] = cval
                cosine_pairs[key]["centroid"].append(cval)
                # per-prompt: per-prompt mean of X vs global centroid of Y
                xmeans = np.stack([Hl[g].mean(axis=0)
                                   for (c, s), g in group_rows.items() if c == ca], axis=0)
                cos_dist = metrics.cosine_rowwise(xmeans, vb)
                cosine_pairs[key]["per_prompt"].append(_dist_stats(cos_dist))
            else:
                cosine_pairs[key]["centroid"].append(None)
                cosine_pairs[key]["per_prompt"].append({"n": 0, "mean": None})
        per_layer.append(row)

    # PCA (2D) at the best probe layer for visualization
    Hb = H[:, best_layer, :]
    coords = _pca_2d(Hb)

    metrics_out = {
        "pos_offset": offset,
        "n_rows": n,
        "n_layers": n_layers,
        "dim": dim,
        "best_probe_layer": best_layer,
        "best_probe_auc": best_auc,
        "probe_labels": "attack(B,D)=1 vs benign(C,E,F,G)=0",
        "per_layer": per_layer,
    }
    pos_dir = cfg.pos_dir(offset)
    io.write_json(pos_dir / "representation_metrics.json", metrics_out)
    io.write_csv(pos_dir / "representation_metrics.csv", per_layer)
    io.write_json(pos_dir / "cosine_pairs.json",
                  {"pairs": list(cosine_pairs.keys()),
                   "method_a": "centroid-vs-centroid (per layer)",
                   "method_b": "per-prompt mean(X) vs global centroid(Y) (per layer)",
                   "by_pair": cosine_pairs})
    np.savez_compressed(pos_dir / "ref_centroids.npz", ref_centroids=ref_centroids)
    np.savez_compressed(pos_dir / "pca_coords.npz", coords=coords,
                        categories=np.array(cats), layer=best_layer)
    print(f"[04] pos{offset}: best probe layer={best_layer} auc={best_auc} -> {pos_dir}")
    return metrics_out


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        return x @ vt[:2].T
    except np.linalg.LinAlgError:
        return np.zeros((len(x), 2))


def run(cfg: ExpConfig, lm=None) -> dict:  # lm unused (model-free stage)
    rows = io.read_jsonl(cfg.out_dir / "tokens.jsonl")
    hidden = np.load(cfg.out_dir / "features.npz")["hidden"]
    if hidden.size == 0:
        raise SystemExit("[04] features.npz has no hidden cube (was --no_hidden used?).")
    out = {}
    for off in cfg.pos_offsets:
        out[f"pos{off}"] = _analyze_offset(cfg, rows, hidden, off)
    return out


def main() -> None:
    args = make_stage_parser(__doc__).parse_args()
    run(config_from_args(args))


if __name__ == "__main__":
    main()
