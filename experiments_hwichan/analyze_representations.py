"""Representation-geometry analysis of the four token categories.

Reads ``tokens.jsonl`` + ``features.npz`` produced by ``extract_representations``
and answers two questions per layer:

  1. How do A/B/C/D/E differ geometrically?
       - per-category mean hidden-state norm
       - centroid pairwise cosine / L2 (esp. A-D, B-D, A-B, C-D)
       - A->D convergence trajectory: cos(centroid_A, centroid_D) vs depth
  2. Is the attack class (A u B) linearly separable from the negatives
     (C u E)?  -> per-layer logistic-regression probe (sklearn if available,
     nearest-centroid fallback otherwise) + best-layer PCA 2D coordinates.

Outputs: ``representation_metrics.json``, ``representation_metrics.csv``,
``pca_coords.npz``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments_hwichan.common import (
    ATTACK_CATS,
    CAT_BENIGN_SPECIAL,
    CAT_MALICIOUS,
    CAT_MIMICRY,
    CAT_ORDINARY,
    CAT_SYSTEM,
    NEGATIVE_CATS,
    read_jsonl,
    write_json,
)

CATS = [CAT_MIMICRY, CAT_MALICIOUS, CAT_BENIGN_SPECIAL, CAT_SYSTEM, CAT_ORDINARY]


def _try_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        from sklearn.model_selection import cross_val_score  # noqa: F401
        return True
    except Exception:
        return False


def load_features(out_dir: Path, pos_offset: int = 0):
    rows = read_jsonl(out_dir / "tokens.jsonl")
    hidden = np.load(out_dir / "features.npz")["hidden"].astype(np.float32)
    keep = [i for i, r in enumerate(rows) if r["pos_offset"] == pos_offset]
    rows = [rows[i] for i in keep]
    hidden = hidden[keep] if hidden.size else hidden
    return rows, hidden


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def cat_indices(rows) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {c: [] for c in CATS}
    for i, r in enumerate(rows):
        out.setdefault(r["category"], []).append(i)
    return out


def numpy_pca_2d(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    # economy SVD; columns of Vt are principal directions
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:2].T


def nearest_centroid_cv_acc(x: np.ndarray, y: np.ndarray, folds: int = 5) -> float:
    """Dependency-free fallback: 5-fold nearest-centroid balanced accuracy."""
    n = len(y)
    if n < folds or len(np.unique(y)) < 2:
        return float("nan")
    idx = np.arange(n)
    rng_order = idx  # deterministic (no shuffling -> reproducible without RNG)
    fold_sizes = np.full(folds, n // folds)
    fold_sizes[: n % folds] += 1
    accs = []
    start = 0
    for fs in fold_sizes:
        test = rng_order[start : start + fs]
        train = np.concatenate([rng_order[:start], rng_order[start + fs :]])
        start += fs
        if len(np.unique(y[train])) < 2:
            continue
        c1 = x[train][y[train] == 1].mean(axis=0)
        c0 = x[train][y[train] == 0].mean(axis=0)
        d1 = np.linalg.norm(x[test] - c1, axis=1)
        d0 = np.linalg.norm(x[test] - c0, axis=1)
        pred = (d1 < d0).astype(int)
        # balanced accuracy
        accs.append(_balanced_acc(y[test], pred))
    return float(np.mean(accs)) if accs else float("nan")


def _balanced_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    accs = []
    for cls in (0, 1):
        m = y_true == cls
        if m.sum() == 0:
            continue
        accs.append((y_pred[m] == cls).mean())
    return float(np.mean(accs)) if accs else float("nan")


def probe_per_layer(hidden: np.ndarray, y: np.ndarray, have_sklearn: bool):
    """Return list of dicts with per-layer separability of attack vs negative."""
    n_layers = hidden.shape[1]
    results = []
    if have_sklearn:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        for l in range(n_layers):
            x = hidden[:, l, :]
            try:
                clf = LogisticRegression(max_iter=2000, C=1.0)
                auc = cross_val_score(clf, x, y, cv=5, scoring="roc_auc").mean()
                acc = cross_val_score(clf, x, y, cv=5, scoring="balanced_accuracy").mean()
            except Exception as e:  # pragma: no cover
                auc, acc = float("nan"), float("nan")
                print(f"[probe] layer {l} sklearn error: {e}")
            results.append({"layer": l, "roc_auc": round(float(auc), 5),
                            "balanced_acc": round(float(acc), 5), "method": "logreg"})
    else:
        for l in range(n_layers):
            acc = nearest_centroid_cv_acc(hidden[:, l, :], y)
            results.append({"layer": l, "roc_auc": None,
                            "balanced_acc": round(float(acc), 5),
                            "method": "nearest_centroid"})
    return results


def analyze(out_dir: Path, pos_offset: int = 0) -> dict:
    rows, hidden = load_features(out_dir, pos_offset=pos_offset)
    if hidden.size == 0:
        raise SystemExit("No features for the requested pos_offset.")
    n_layers = hidden.shape[1]
    idx = cat_indices(rows)
    present = {c: v for c, v in idx.items() if v}

    # per-layer centroids per category
    centroids = {
        c: np.stack([hidden[v, l, :].mean(axis=0) for l in range(n_layers)], axis=0)
        for c, v in present.items()
    }  # each [n_layers, dim]

    per_layer = []
    pairs = [
        (CAT_MIMICRY, CAT_SYSTEM),
        (CAT_MALICIOUS, CAT_SYSTEM),
        (CAT_MIMICRY, CAT_MALICIOUS),
        (CAT_BENIGN_SPECIAL, CAT_SYSTEM),
        (CAT_MIMICRY, CAT_ORDINARY),
        (CAT_BENIGN_SPECIAL, CAT_ORDINARY),
    ]
    for l in range(n_layers):
        row = {"layer": l}
        for c in CATS:
            if c in present:
                row[f"{c}__mean_norm"] = round(
                    float(np.linalg.norm(hidden[present[c], l, :], axis=1).mean()), 5
                )
        for a, b in pairs:
            if a in centroids and b in centroids:
                row[f"cos__{a}__{b}"] = round(_cos(centroids[a][l], centroids[b][l]), 5)
                row[f"l2__{a}__{b}"] = round(
                    float(np.linalg.norm(centroids[a][l] - centroids[b][l])), 5
                )
        per_layer.append(row)

    # attack (A u B) vs negative (C u E) probe
    y_list, keep = [], []
    for i, r in enumerate(rows):
        if r["category"] in ATTACK_CATS:
            y_list.append(1); keep.append(i)
        elif r["category"] in NEGATIVE_CATS:
            y_list.append(0); keep.append(i)
    probe = []
    best_layer = None
    if len(set(y_list)) == 2:
        y = np.array(y_list)
        hsub = hidden[keep]
        probe = probe_per_layer(hsub, y, _try_sklearn())
        scored = [p for p in probe if p.get("roc_auc") is not None]
        key = "roc_auc" if scored else "balanced_acc"
        best = max(probe, key=lambda p: (p.get(key) or -1))
        best_layer = best["layer"]
    else:
        print("[analyze] not enough class diversity for probe (need A/B and C/E).")

    # PCA coords at best layer (+ first/mid/last) for visualisation
    pca = {}
    layers_for_pca = sorted({0, n_layers // 2, n_layers - 1,
                             best_layer if best_layer is not None else n_layers - 1})
    for l in layers_for_pca:
        coords = numpy_pca_2d(hidden[:, l, :])
        pca[f"layer_{l}"] = coords.astype(np.float32)
    np.savez_compressed(
        out_dir / "pca_coords.npz",
        categories=np.array([r["category"] for r in rows]),
        variants=np.array([r["variant"] for r in rows]),
        **pca,
    )

    metrics = {
        "pos_offset": pos_offset,
        "n_layers": n_layers,
        "n_rows": len(rows),
        "category_counts": {c: len(v) for c, v in present.items()},
        "per_layer": per_layer,
        "attack_vs_negative_probe": probe,
        "best_probe_layer": best_layer,
        "A_to_D_convergence": [
            {"layer": r["layer"], "cos": r.get(f"cos__{CAT_MIMICRY}__{CAT_SYSTEM}")}
            for r in per_layer
        ],
        "pca_layers": layers_for_pca,
    }
    write_json(out_dir / "representation_metrics.json", metrics)

    # csv of per-layer table
    if per_layer:
        keys = sorted({k for r in per_layer for k in r})
        keys = ["layer"] + [k for k in keys if k != "layer"]
        with open(out_dir / "representation_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_layer:
                w.writerow(r)
    print(f"[analyze] best probe layer={best_layer}; wrote representation_metrics.*")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pos_offset", type=int, default=0, choices=[0, 1])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    analyze(Path(args.out_dir), pos_offset=args.pos_offset)


if __name__ == "__main__":
    main()
