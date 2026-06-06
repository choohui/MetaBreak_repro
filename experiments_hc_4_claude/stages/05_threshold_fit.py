"""Stage 05 — fit thresholds on TRAIN, measure stability, pick an operating point.

For each scalarizer at its stage-04 best layer, every selected threshold method is
fit on the TRAIN rows and refit across CV folds + bootstraps to measure drift
(``threshold_cv``). The operating point is then chosen ON TRAIN ONLY — the highest
train-AUC scalarizer, then the most STABLE threshold (lowest cv) — never by
peeking at the held-out set. This is the explicit guard against the hc_2 collapse
(an unstable low-FPR threshold that did not transfer).

A separate operating point is recorded for the borderline (fitted-direction)
scalarizers so they never enter the clean headline.

Outputs (per ``pos{off}/``):
    threshold_stability.json - per scalarizer/method: threshold, cv, CI, AUC
    operating_points.json    - selected (scalarizer, layer, method) for clean + borderline
    threshold_per_type.csv   - flagged rate per letter at the selected op point
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
from experiments_hc_4_claude.core import io                       # noqa: E402
from experiments_hc_4_claude.core import thresholds as TH         # noqa: E402
from experiments_hc_4_claude.core.cascade import per_type_rates   # noqa: E402
from experiments_hc_4_claude.stages import scalar_common as sc    # noqa: E402


def _best_layers(cfg, off) -> dict:
    rep = io.read_json(cfg.pos_dir(off) / "scalarizer_auc.json")
    return {k: (v.get("best_layer"), v.get("best_train_auc"), v.get("borderline"))
            for k, v in rep.get("per_scalarizer", {}).items()}


def _run_offset(cfg: ExpConfig, off: int) -> dict:
    rows, mats, meta, arr = sc.load_scalar_scores(cfg, off)
    y, is_train, groups = arr["y"], arr["is_train"], arr["groups"]
    best = _best_layers(cfg, off)
    n_boot = int(min(cfg.n_bootstrap, 200))

    stability = {"pos_offset": off, "selected_on": "train", "per_scalarizer": {}}
    for k, (layer, train_auc, borderline) in best.items():
        if layer is None or k not in mats:
            continue
        col = mats[k][:, layer]
        s_tr, y_tr, g_tr = col[is_train], y[is_train], groups[is_train]
        per_method = {}
        for method in cfg.threshold_methods:
            per_method[method] = TH.threshold_stability(
                s_tr, y_tr, g_tr, method, fn_fp_cost=cfg.fn_fp_cost,
                n_folds=cfg.cv_folds, n_bootstrap=n_boot, seed=cfg.seed)
        stability["per_scalarizer"][k] = {
            "layer": layer, "train_auc": train_auc, "borderline": borderline,
            "methods": per_method}
    io.write_json(cfg.pos_dir(off) / "threshold_stability.json", stability)

    op = {"pos_offset": off, "selected_on": "train", "headline": "clean",
          "clean": _select(stability, borderline=False),
          "borderline": _select(stability, borderline=True)}
    io.write_json(cfg.pos_dir(off) / "operating_points.json", op)

    # per-type flagged rate at the selected clean op point (descriptive, TRAIN view)
    per_type_rows = []
    sel = op["clean"]
    if sel:
        col = mats[sel["scalarizer"]][:, sel["layer"]]
        pred = TH.predict(col, sel["threshold"], sel["direction"])
        for split_name, mask in (("train", is_train), ("test", arr["is_test"])):
            rates = per_type_rates([rows[i] for i in np.where(mask)[0]], pred[mask])
            for letter, info in rates.items():
                per_type_rows.append({"split": split_name, "letter": letter, **info})
    io.write_csv(cfg.pos_dir(off) / "threshold_per_type.csv", per_type_rows)

    print(f"[05] pos{off}: clean op-point={_fmt(op['clean'])}  borderline={_fmt(op['borderline'])}")
    return {"pos_offset": off, "operating_point": op}


def _select(stability: dict, borderline: bool):
    """Pick the (scalarizer, layer, method) on TRAIN: best train-AUC scalarizer of
    the requested family, then its most stable threshold (lowest cv)."""
    cands = [(k, v) for k, v in stability["per_scalarizer"].items()
             if bool(v["borderline"]) == borderline and v["train_auc"] is not None]
    if not cands:
        return None
    cands.sort(key=lambda kv: kv[1]["train_auc"], reverse=True)
    k, v = cands[0]
    methods = v["methods"]
    # most stable threshold with a usable value; cv None -> treat as +inf
    best_m, best_cv = None, np.inf
    for m, info in methods.items():
        if info.get("threshold") is None:
            continue
        cv = info.get("threshold_cv")
        cv = np.inf if cv is None else cv
        if cv < best_cv:
            best_cv, best_m = cv, m
    if best_m is None:                       # fall back to youden if present
        best_m = "youden" if "youden" in methods else next(iter(methods))
    chosen = methods[best_m]
    return {"scalarizer": k, "layer": v["layer"], "method": best_m,
            "threshold": chosen.get("threshold"), "direction": chosen.get("direction"),
            "train_auc": v["train_auc"], "threshold_cv": chosen.get("threshold_cv"),
            "threshold_ci95": chosen.get("threshold_ci95")}


def _fmt(sel):
    if not sel:
        return None
    return f"{sel['scalarizer']}@L{sel['layer']}/{sel['method']}(cv={sel['threshold_cv']})"


def run(cfg: ExpConfig, lm=None) -> dict:
    out = {}
    for off in cfg.pos_offsets:
        out[f"pos{off}"] = _run_offset(cfg, off)
    return out


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
