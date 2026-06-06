"""Shared helpers for the scalar-defense analysis stages (04-09).

Centralises: loading the balanced stage-03 artifacts per pos_offset, the
prompt-level train/held-out split, the production scalar fit (fit on ALL train,
score every row), the HONEST train-side AUC (out-of-fold cross-fitting for the
fitted scalarizers so their train AUC is not optimistic), and the on-disk format
shared between stages (``scalar_scores.npz`` + ``scalar_scores_meta.json``).

Every stage reloads these from disk so it composes standalone (the hc_2/hc_3
pattern that the smoke test re-exercises without a model).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_4_claude.core import io, metrics            # noqa: E402
from experiments_hc_4_claude.core import scalarizers as SZ      # noqa: E402
from experiments_hc_4_claude.core.cascade import binary_labels  # noqa: E402
from experiments_hc_4_claude.core.labels import CAT_TO_LETTER   # noqa: E402
from experiments_hc_4_claude.core.splits import holdout_mask    # noqa: E402

import experiments_hc_4_claude.stages.analysis_common as ac      # noqa: E402  (success_set/load_artifacts)


# --------------------------------------------------------------------------- #
# loading + splitting
# --------------------------------------------------------------------------- #
def load_pos(cfg, off: int, balanced: bool = True):
    """Balanced stage-03 rows for one pos_offset + the full hidden cube + ASR set."""
    rows, hidden, success = ac.load_artifacts(cfg.out_dir, cfg.asr_judge, balanced=balanced)
    rows = [r for r in rows if int(r["pos_offset"]) == off]
    return rows, hidden, success


def split_masks(cfg, rows):
    is_train, is_test = holdout_mask(rows, cfg.seed, cfg.holdout_frac)
    groups = np.array([int(r["sample_index"]) for r in rows], dtype=int)
    eval_mode = "in_sample" if bool(np.any(is_train & is_test)) else "holdout"
    return is_train, is_test, groups, eval_mode


def labels(rows, success=None):
    return binary_labels(rows, success)


# --------------------------------------------------------------------------- #
# honest train-side AUC (out-of-fold for fitted scalarizers)
# --------------------------------------------------------------------------- #
def _gather_hidden(rows, hidden):
    if hidden is None or not getattr(hidden, "size", 0):
        return None
    ridx = np.array([r["row_id"] for r in rows], dtype=int)
    return hidden[ridx].astype(np.float32)


def _group_fold_assign(groups, is_train, k, seed):
    uniq = np.array(sorted({int(g) for g, t in zip(groups, is_train) if t}))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    k = max(2, min(k, len(uniq))) if len(uniq) >= 2 else 1
    return {int(g): (i % k) for i, g in enumerate(uniq)}, k


def honest_train_layer_auc(cfg, rows, hidden, key, mat_prod, is_train, y, groups):
    """Per-layer AUC on the TRAIN rows + the score matrix it was measured on.

    For non-fitted scalarizers this is the plain train AUC of the production
    matrix; for fitted (hidden-based) ones it is the out-of-fold AUC (fit on
    train-folds, score the held fold) so the estimate used for model selection is
    not optimistic. Returns ``(auc_per_layer, score_matrix)`` where score_matrix is
    the production matrix (non-fitted) or the out-of-fold matrix (fitted)."""
    sc = SZ.REGISTRY[key]
    tr = is_train & (y >= 0)
    n_layers = mat_prod.shape[1]
    if not sc.needs_hidden or tr.sum() < 4 or len(set(int(g) for g in groups[tr])) < 2:
        return [_safe_auc(mat_prod[tr, l], y[tr]) for l in range(n_layers)], mat_prod

    Hsub = _gather_hidden(rows, hidden)
    fold_of, k = _group_fold_assign(groups, is_train, cfg.cv_folds, cfg.seed)
    oof = np.full(mat_prod.shape, np.nan)
    for f in range(k):
        inner_train = np.array([bool(is_train[i] and fold_of.get(int(groups[i]), -1) != f)
                                for i in range(len(rows))])
        held = np.array([bool(is_train[i] and fold_of.get(int(groups[i]), -1) == f)
                         for i in range(len(rows))])
        if inner_train.sum() < 2 or held.sum() == 0:
            continue
        m = sc.compute(rows, Hsub, inner_train, y, {})
        m = SZ.normalize_matrix(m, rows, cfg.normalize)
        oof[held] = m[held]
    return [_safe_auc(oof[tr, l], y[tr]) for l in range(n_layers)], oof


def _safe_auc(col, y):
    col = np.asarray(col, dtype=np.float64)
    yy = np.asarray(y, dtype=np.int64)
    mask = (yy >= 0) & ~np.isnan(col)
    if mask.sum() == 0 or yy[mask].min() == yy[mask].max():
        return float("nan")
    return round(float(metrics.roc_auc(col[mask], yy[mask])), 5)


# --------------------------------------------------------------------------- #
# production scalars + persistence
# --------------------------------------------------------------------------- #
def compute_production(cfg, rows, hidden, is_train, success=None, keys=None):
    """Fit on ALL train rows, score every row. Returns (mats, aux, y, keys)."""
    mats, aux, y = SZ.compute_scalars(cfg, rows, hidden, is_train, keys=keys, success=success)
    keys = list(mats.keys())
    return mats, aux, y, keys


def save_scalar_scores(cfg, off, rows, mats, aux, y, is_train, is_test, groups, eval_mode):
    pdir = cfg.pos_dir(off)
    pdir.mkdir(parents=True, exist_ok=True)
    npz: dict[str, np.ndarray] = {
        "row_id": np.array([r["row_id"] for r in rows], dtype=int),
        "sample_index": np.array([int(r["sample_index"]) for r in rows], dtype=int),
        "y": np.asarray(y, dtype=int),
        "is_train": np.asarray(is_train, dtype=bool),
        "is_test": np.asarray(is_test, dtype=bool),
        "groups": np.asarray(groups, dtype=int),
    }
    for k, m in mats.items():
        npz["mat__" + k] = np.asarray(m, dtype=np.float32)
    np.savez_compressed(pdir / "scalar_scores.npz", **npz)
    # light fitted directions (NOT covariances) for interpretability
    if aux:
        np.savez_compressed(pdir / "scalarizer_fit.npz",
                            **{("dir__" + k): np.asarray(v, dtype=np.float32) for k, v in aux.items()})
    letters = [r["letter"] for r in rows]
    meta = {
        "pos_offset": off,
        "keys": list(mats.keys()),
        "borderline": {k: SZ.is_borderline(k) for k in mats.keys()},
        "layer_counts": {k: int(m.shape[1]) for k, m in mats.items()},
        "n_rows": len(rows),
        "n_train": int(np.sum(is_train)),
        "n_test": int(np.sum(is_test)),
        "eval_mode": eval_mode,
        "normalize": cfg.normalize,
        "scalarizer_set": cfg.scalarizer_set,
        "fit_on": "train",
        "letters": letters,
        "prompt_idx": [str(r.get("prompt_idx", "")) for r in rows],
    }
    io.write_json(pdir / "scalar_scores_meta.json", meta)
    return meta


def load_scalar_scores(cfg, off):
    """Reload what stage 04 saved. Returns (rows, mats, meta, arrays)."""
    pdir = cfg.pos_dir(off)
    meta = io.read_json(pdir / "scalar_scores_meta.json")
    z = np.load(pdir / "scalar_scores.npz")
    keys = meta["keys"]
    mats = {k: z["mat__" + k] for k in keys}
    prompt_idx = meta.get("prompt_idx", [""] * int(meta["n_rows"]))
    rows = []
    for i in range(int(meta["n_rows"])):
        rows.append({
            "row_id": int(z["row_id"][i]),
            "sample_index": int(z["sample_index"][i]),
            "prompt_idx": prompt_idx[i],
            "letter": meta["letters"][i],
        })
    arrays = {"y": z["y"], "is_train": z["is_train"], "is_test": z["is_test"],
              "groups": z["groups"]}
    return rows, mats, meta, arrays
