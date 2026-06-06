"""Stage 06 — evaluate the selected operating point on the HELD-OUT test split.

This is the only stage that reads TEST scores, and only to REPORT (selection
happened on train in stages 04/05). It is exactly the scenario hc_2's cascade
failed: a threshold fixed on train, applied to unseen prompts. We report TEST
AUC, aggregated TPR / benign-FPR, the per-type breakdown, ROC/DET/PR +
calibration curves, and a prompt-grouped permutation p-value for the selected
scalarizer.

Outputs (per ``pos{off}/``):
    holdout_eval.json   - clean + borderline op points, train vs test metrics
    curves.json         - ROC/DET/PR + calibration for the clean op scalarizer (TEST)
    permutation_test.json - label-permutation p-value on TEST for the op scalarizer
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
from experiments_hc_4_claude.core import io, metrics, curves, stats  # noqa: E402
from experiments_hc_4_claude.core import thresholds as TH            # noqa: E402
from experiments_hc_4_claude.core.cascade import per_type_rates      # noqa: E402
from experiments_hc_4_claude.stages import scalar_common as sc       # noqa: E402


def _agg(pred, y, mask):
    sub = mask & (y >= 0)
    pos = sub & (y == 1)
    neg = sub & (y == 0)
    tpr = float(pred[pos].mean()) if pos.sum() else None
    fpr = float(pred[neg].mean()) if neg.sum() else None
    return {"n": int(sub.sum()), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
            "tpr": None if tpr is None else round(tpr, 5),
            "benign_fpr": None if fpr is None else round(fpr, 5)}


def _eval_family(sel, rows, mats, y, is_train, is_test):
    if not sel or sel["scalarizer"] not in mats:
        return None
    col = mats[sel["scalarizer"]][:, sel["layer"]]
    pred = TH.predict(col, sel["threshold"], sel["direction"])
    out = dict(sel)
    for name, mask in (("train", is_train), ("test", is_test)):
        rate = _agg(pred, y, mask)
        rate["auc"] = sc._safe_auc(col[mask], y[mask])
        rate["per_type"] = per_type_rates(
            [rows[i] for i in np.where(mask)[0]], pred[mask])
        out[name] = rate
    return out


def _run_offset(cfg: ExpConfig, off: int) -> dict:
    rows, mats, meta, arr = sc.load_scalar_scores(cfg, off)
    y, is_train, is_test, groups = arr["y"], arr["is_train"], arr["is_test"], arr["groups"]
    op = io.read_json(cfg.pos_dir(off) / "operating_points.json")

    report = {"pos_offset": off, "eval_mode": meta["eval_mode"], "selected_on": "train",
              "clean": _eval_family(op.get("clean"), rows, mats, y, is_train, is_test),
              "borderline": _eval_family(op.get("borderline"), rows, mats, y, is_train, is_test)}
    io.write_json(cfg.pos_dir(off) / "holdout_eval.json", report)

    sel = op.get("clean")
    if sel and sel["scalarizer"] in mats:
        col = mats[sel["scalarizer"]][:, sel["layer"]]
        cur = curves.roc_det_pr(col[is_test], y[is_test])
        cur["calibration"] = curves.calibration(col[is_test], y[is_test])
        cur["scalarizer"] = sel["scalarizer"]; cur["layer"] = sel["layer"]
        io.write_json(cfg.pos_dir(off) / "curves.json", cur)
        perm = stats.permutation_pvalue(col[is_test], y[is_test], groups[is_test],
                                        cfg.n_perm, cfg.seed)
        perm["scalarizer"] = sel["scalarizer"]; perm["layer"] = sel["layer"]
        io.write_json(cfg.pos_dir(off) / "permutation_test.json", perm)

    ct = report["clean"]["test"] if report["clean"] else None
    print(f"[06] pos{off}: eval_mode={meta['eval_mode']} "
          f"clean held-out AUC={ct['auc'] if ct else None} TPR={ct['tpr'] if ct else None} "
          f"FPR={ct['benign_fpr'] if ct else None}")
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
