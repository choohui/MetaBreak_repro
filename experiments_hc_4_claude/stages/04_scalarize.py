"""Stage 04 — scalarize internal representations (fit on TRAIN, score every row).

For each pos_offset: split prompts into train/held-out, fit every selected
scalarizer's geometry on the TRAIN rows only, reduce each token to one scalar per
layer, and persist the scores. Reports the HONEST train-side AUC per
(scalarizer, layer) — out-of-fold for the fitted scalarizers so model selection
in stage 05 is not optimistic — with a group-bootstrap CI.

Outputs (per ``pos{off}/``):
    scalar_scores.npz        - per-scalarizer [n, n_layers] + y / masks / groups
    scalar_scores_meta.json  - key list, layer counts, borderline tags, eval_mode
    scalarizer_fit.npz       - light fitted directions/centroids (no covariances)
    scalarizer_auc.json      - per-scalarizer per-layer AUC + best layer + CI
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

from experiments_hc_4_claude.config import ExpConfig, config_from_args, make_stage_parser  # noqa: E402
from experiments_hc_4_claude.core import io, stats              # noqa: E402
from experiments_hc_4_claude.stages import scalar_common as sc  # noqa: E402


def _best_layer(auc_list):
    best_l, best_a = None, -1.0
    for l, a in enumerate(auc_list):
        if a is not None and a == a and a > best_a:
            best_a, best_l = a, l
    return best_l, (None if best_l is None else best_a)


def _run_offset(cfg: ExpConfig, off: int) -> dict:
    rows, hidden, success = sc.load_pos(cfg, off, balanced=True)
    if not rows:
        return {"pos_offset": off, "skipped": "no rows"}
    is_train, is_test, groups, eval_mode = sc.split_masks(cfg, rows)
    mats, aux, y, keys = sc.compute_production(cfg, rows, hidden, is_train)
    sc.save_scalar_scores(cfg, off, rows, mats, aux, y, is_train, is_test, groups, eval_mode)

    report = {"pos_offset": off, "eval_mode": eval_mode, "fit_on": "train",
              "normalize": cfg.normalize, "scalarizer_set": cfg.scalarizer_set,
              "n_rows": len(rows), "n_train": int(is_train.sum()),
              "n_test": int(is_test.sum()), "per_scalarizer": {}}
    for k in keys:
        auc_list, score_mat = sc.honest_train_layer_auc(
            cfg, rows, hidden, k, mats[k], is_train, y, groups)
        bl, ba = _best_layer(auc_list)
        ci = {"auc": None, "lo": None, "hi": None, "n_boot": 0}
        if bl is not None:
            ci = stats.bootstrap_auc_ci(score_mat[is_train, bl], y[is_train],
                                        groups[is_train], cfg.n_bootstrap, cfg.seed)
        report["per_scalarizer"][k] = {
            "borderline": sc.SZ.is_borderline(k),
            "needs_hidden": sc.SZ.needs_hidden(k),
            "n_layers": len(auc_list),
            "best_layer": bl,
            "best_train_auc": ba,
            "best_train_auc_ci95": [ci["lo"], ci["hi"]],
            "auc_estimator": "out_of_fold" if sc.SZ.needs_hidden(k) else "plain_train",
            "per_layer": [{"layer": l, "auc": a} for l, a in enumerate(auc_list)],
        }
    io.write_json(cfg.pos_dir(off) / "scalarizer_auc.json", report)
    clean_best = sorted(
        ((v["best_train_auc"] or -1, k) for k, v in report["per_scalarizer"].items()
         if not v["borderline"]), reverse=True)[:3]
    print(f"[04] pos{off}: {len(keys)} scalarizers, eval_mode={eval_mode}; "
          f"top clean by train-AUC: {[(k, round(a,4)) for a, k in clean_best]}")
    return report


def run(cfg: ExpConfig, lm=None) -> dict:
    out = {}
    for off in cfg.pos_offsets:
        out[f"pos{off}"] = _run_offset(cfg, off)
    return out


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
