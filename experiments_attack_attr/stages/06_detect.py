"""Stage 06 (choan.md §2.2 / §3.4) — fit the detector on TRAIN, report on HELD-OUT.

Two things, the way choan.md §2.2 frames detection:

  1. **Fit (train only).** For each signal at its stage-05 best layer, fit every
     threshold selector on the TRAIN rows and refit across CV folds + bootstraps
     to measure drift (``threshold_cv``). Pick ONE operating point per family on
     TRAIN: the **clean** headline (cos_to_attack et al.) and the **borderline**
     token-detector (**diff_means** — choan: the signal that best catches
     malicious tokens at low benign FPR, and the one the §3 defenses flag with).

  2. **Report (held-out test only).** Apply each fixed operating point to the
     unseen TEST prompts and report AUC / TPR / benign-FPR + the per-type flagged
     rate, plus ROC/DET/PR + calibration curves and a prompt-grouped permutation
     p-value for the clean headline. This is the generalisation check (choan: a
     single threshold that worked only on train is the failure to avoid).

Outputs (per ``pos{off}/``):
    threshold_stability.json - per signal/method: threshold, cv, CI, AUC (TRAIN)
    operating_points.json    - selected clean + borderline op points (selected_on=train)
    holdout_eval.json        - clean + borderline: train vs test AUC/TPR/FPR + per-type
    detect_summary.json      - the headline diff_means / cos_to_attack held-out numbers
    threshold_per_type.csv   - flagged rate per letter at the clean op point
    curves.json              - ROC/DET/PR + calibration (clean, TEST)
    permutation_test.json    - label-permutation p-value (clean, TEST)
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

from experiments_attack_attr.config import ExpConfig, config_from_args, make_stage_parser  # noqa: E402
from experiments_attack_attr.core import io, curves, stats  # noqa: E402
from experiments_attack_attr.core import thresholds as TH  # noqa: E402
from experiments_attack_attr.core.cascade import per_type_rates  # noqa: E402
from experiments_attack_attr.stages import scalar_common as sc  # noqa: E402


# --------------------------------------------------------------------------- #
# 1) threshold fit (train) + operating-point selection
# --------------------------------------------------------------------------- #
def _best_layers(cfg, off) -> dict:
    rep = io.read_json(cfg.pos_dir(off) / "scalarizer_auc.json")
    return {k: (v.get("best_layer"), v.get("best_train_auc"), v.get("borderline"))
            for k, v in rep.get("per_scalarizer", {}).items()}


def _pick_threshold(v: dict):
    """Most stable threshold (lowest cv) for one signal's fitted methods."""
    methods = v["methods"]
    best_m, best_cv = None, np.inf
    for m, info in methods.items():
        if info.get("threshold") is None:
            continue
        cv = info.get("threshold_cv")
        cv = np.inf if cv is None else cv
        if cv < best_cv:
            best_cv, best_m = cv, m
    if best_m is None:
        best_m = "youden" if "youden" in methods else next(iter(methods))
    chosen = methods[best_m]
    return {"layer": v["layer"], "method": best_m, "threshold": chosen.get("threshold"),
            "direction": chosen.get("direction"), "train_auc": v["train_auc"],
            "threshold_cv": chosen.get("threshold_cv"),
            "threshold_ci95": chosen.get("threshold_ci95")}


def _select_named(stability: dict, key: str):
    """Operating point for ONE named signal (e.g. ``diff_means``, choan §3.4)."""
    v = stability["per_scalarizer"].get(key)
    if not v or v.get("train_auc") is None:
        return None
    return {"scalarizer": key, **_pick_threshold(v)}


def _select(stability: dict, borderline: bool):
    """Pick (scalarizer, layer, method) on TRAIN: best train-AUC signal of the
    requested family, then its most stable threshold (lowest cv)."""
    cands = [(k, v) for k, v in stability["per_scalarizer"].items()
             if bool(v["borderline"]) == borderline and v["train_auc"] is not None]
    if not cands:
        return None
    cands.sort(key=lambda kv: kv[1]["train_auc"], reverse=True)
    k, v = cands[0]
    methods = v["methods"]
    best_m, best_cv = None, np.inf
    for m, info in methods.items():
        if info.get("threshold") is None:
            continue
        cv = info.get("threshold_cv")
        cv = np.inf if cv is None else cv
        if cv < best_cv:
            best_cv, best_m = cv, m
    if best_m is None:
        best_m = "youden" if "youden" in methods else next(iter(methods))
    chosen = methods[best_m]
    return {"scalarizer": k, "layer": v["layer"], "method": best_m,
            "threshold": chosen.get("threshold"), "direction": chosen.get("direction"),
            "train_auc": v["train_auc"], "threshold_cv": chosen.get("threshold_cv"),
            "threshold_ci95": chosen.get("threshold_ci95")}


def _fit_thresholds(cfg, off, rows, mats, arr) -> dict:
    y, is_train, groups = arr["y"], arr["is_train"], arr["groups"]
    best = _best_layers(cfg, off)
    n_boot = int(min(cfg.n_bootstrap, 200))
    stability = {"pos_offset": off, "selected_on": "train", "per_scalarizer": {}}
    for k, (layer, train_auc, borderline) in best.items():
        if layer is None or k not in mats:
            continue
        col = mats[k][:, layer]
        s_tr, y_tr, g_tr = col[is_train], y[is_train], groups[is_train]
        per_method = {m: TH.threshold_stability(
            s_tr, y_tr, g_tr, m, fn_fp_cost=cfg.fn_fp_cost,
            n_folds=cfg.cv_folds, n_bootstrap=n_boot, seed=cfg.seed)
            for m in cfg.threshold_methods}
        stability["per_scalarizer"][k] = {
            "layer": layer, "train_auc": train_auc, "borderline": borderline,
            "methods": per_method}
    io.write_json(cfg.pos_dir(off) / "threshold_stability.json", stability)

    op = {"pos_offset": off, "selected_on": "train", "headline": "clean",
          "clean": _select(stability, borderline=False),
          "borderline": _select(stability, borderline=True),
          # choan §2.2 names cos_to_attack as the clean headline and diff_means as
          # the borderline token detector; record each signal's own op point so the
          # report surfaces those named numbers verbatim (not just the auto-selected
          # best-of-family, which can differ on noisy/small data).
          "cos_to_attack": _select_named(stability, "cos_to_attack"),
          "diff_means": _select_named(stability, "diff_means")}
    io.write_json(cfg.pos_dir(off) / "operating_points.json", op)
    return op


# --------------------------------------------------------------------------- #
# 2) held-out evaluation
# --------------------------------------------------------------------------- #
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
        rate["per_type"] = per_type_rates([rows[i] for i in np.where(mask)[0]], pred[mask])
        out[name] = rate
    return out


def _run_offset(cfg: ExpConfig, off: int) -> dict:
    rows, mats, meta, arr = sc.load_scalar_scores(cfg, off)
    y, is_train, is_test, groups = arr["y"], arr["is_train"], arr["is_test"], arr["groups"]

    op = _fit_thresholds(cfg, off, rows, mats, arr)

    report = {"pos_offset": off, "eval_mode": meta["eval_mode"], "selected_on": "train",
              "clean": _eval_family(op.get("clean"), rows, mats, y, is_train, is_test),
              "borderline": _eval_family(op.get("borderline"), rows, mats, y, is_train, is_test)}
    io.write_json(cfg.pos_dir(off) / "holdout_eval.json", report)

    # per-type flagged rate at the clean op point (TRAIN + TEST)
    sel = op.get("clean")
    per_type_rows = []
    if sel and sel["scalarizer"] in mats:
        col = mats[sel["scalarizer"]][:, sel["layer"]]
        pred = TH.predict(col, sel["threshold"], sel["direction"])
        for split_name, mask in (("train", is_train), ("test", is_test)):
            for letter, info in per_type_rates(
                    [rows[i] for i in np.where(mask)[0]], pred[mask]).items():
                per_type_rows.append({"split": split_name, "letter": letter, **info})
        cur = curves.roc_det_pr(col[is_test], y[is_test])
        cur["calibration"] = curves.calibration(col[is_test], y[is_test])
        cur["scalarizer"] = sel["scalarizer"]; cur["layer"] = sel["layer"]
        io.write_json(cfg.pos_dir(off) / "curves.json", cur)
        perm = stats.permutation_pvalue(col[is_test], y[is_test], groups[is_test],
                                        cfg.n_perm, cfg.seed)
        perm["scalarizer"] = sel["scalarizer"]; perm["layer"] = sel["layer"]
        io.write_json(cfg.pos_dir(off) / "permutation_test.json", perm)
    io.write_csv(cfg.pos_dir(off) / "threshold_per_type.csv", per_type_rows)

    # headline summary: the auto-selected best clean/borderline op points PLUS the
    # named choan signals (cos_to_attack clean headline, diff_means token detector)
    # evaluated on held-out, so the §2.2 claims are reported verbatim.
    def _head(evald):
        if not evald or not evald.get("test"):
            return None
        return {"scalarizer": evald.get("scalarizer"), "layer": evald.get("layer"),
                "test_auc": evald["test"].get("auc"), "test_tpr": evald["test"].get("tpr"),
                "test_benign_fpr": evald["test"].get("benign_fpr")}

    cos_named = _eval_family(op.get("cos_to_attack"), rows, mats, y, is_train, is_test)
    dm_named = _eval_family(op.get("diff_means"), rows, mats, y, is_train, is_test)
    summary = {"pos_offset": off, "eval_mode": meta["eval_mode"],
               "clean_headline": _head(report.get("clean")),
               "borderline_detector": _head(report.get("borderline")),
               "cos_to_attack": _head(cos_named),     # §2.2 named clean headline
               "diff_means": _head(dm_named)}          # §2.2 named token detector
    io.write_json(cfg.pos_dir(off) / "detect_summary.json", summary)

    ct = report["clean"]["test"] if report["clean"] else None
    bt = report["borderline"]["test"] if report["borderline"] else None
    print(f"[06] pos{off}: eval_mode={meta['eval_mode']} | clean "
          f"{op.get('clean', {}) and op['clean']['scalarizer']}: "
          f"AUC={ct['auc'] if ct else None} TPR={ct['tpr'] if ct else None} "
          f"FPR={ct['benign_fpr'] if ct else None} | borderline "
          f"{op.get('borderline', {}) and op['borderline'] and op['borderline']['scalarizer']}: "
          f"AUC={bt['auc'] if bt else None} TPR={bt['tpr'] if bt else None} "
          f"FPR={bt['benign_fpr'] if bt else None}")
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
