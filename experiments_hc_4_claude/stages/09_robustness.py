"""Stage 09 — robustness, ablations and an aggregated summary.

Re-runs the scalar defense (model-free, from the saved hidden cube) under several
ablation arms — per-prompt normalisation on/off, clean vs borderline scalarizer
family, sink-gate on/off — each measured on the HELD-OUT split, so the headline's
sensitivity to design choices is explicit. It also collates the permutation
p-values and bootstrap CIs from the per-offset stages and checks cross-pos_offset
consistency, and records the 7-way balance census (the A-G equalisation).

Outputs:
    pos{off}/ablations.json
    robustness_report.json + robustness_report.md
    summary.md
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_4_claude.config import ExpConfig, config_from_args, make_stage_parser  # noqa: E402
from experiments_hc_4_claude.core import io                                  # noqa: E402
from experiments_hc_4_claude.core import thresholds as TH                    # noqa: E402
from experiments_hc_4_claude.core.cascade import prompt_block_and_asr, sink_gate  # noqa: E402
from experiments_hc_4_claude.stages import scalar_common as sc              # noqa: E402
from experiments_hc_4_claude.stages.analysis_common import success_set       # noqa: E402


def _gate_keep_ids(cfg, off, pct):
    if pct >= 100:
        return None
    summary = io.read_json(cfg.out_dir / "extract_summary.json")
    keep = set(summary.get("balanced_row_ids", []))
    toks = [r for r in io.read_jsonl(cfg.out_dir / "tokens.jsonl")
            if int(r["pos_offset"]) == off and r["row_id"] in keep]
    return {r["row_id"] for r in sink_gate(toks, pct)}


def _eval_variant(cfg2: ExpConfig, off: int, family: str, gate_pct: float, success) -> dict:
    rows, hidden, _ = sc.load_pos(cfg2, off, balanced=True)
    if not rows:
        return {"skipped": "no rows"}
    is_train, is_test, groups, eval_mode = sc.split_masks(cfg2, rows)
    keys = sc.SZ.SCALARIZER_SETS.get(family, sc.SZ.CLEAN_SET)
    have_hidden = bool(getattr(hidden, "size", 0))
    keys = [k for k in keys if not sc.SZ.needs_hidden(k) or have_hidden]
    mats, _aux, y, _ = sc.compute_production(cfg2, rows, hidden, is_train, keys=keys)

    # pick the best scalarizer by honest train AUC, then fit a youden threshold
    best_k, best_layer, best_auc = None, None, -1.0
    layer_cols = {}
    for k in mats:
        auc_list, _m = sc.honest_train_layer_auc(cfg2, rows, hidden, k, mats[k],
                                                 is_train, y, groups)
        for l, a in enumerate(auc_list):
            if a is not None and a == a and a > best_auc:
                best_auc, best_k, best_layer = a, k, l
        layer_cols[k] = mats[k]
    if best_k is None:
        return {"eval_mode": eval_mode, "skipped": "no usable scalarizer"}
    col = layer_cols[best_k][:, best_layer]
    thr = TH.select_threshold(col[is_train], y[is_train], "youden", cfg2.fn_fp_cost)
    pred = TH.predict(col, thr["threshold"], thr["direction"])
    gate = _gate_keep_ids(cfg2, off, gate_pct)
    if gate is not None:
        pred = pred & np.array([r["row_id"] in gate for r in rows])
    test_idx = np.where(is_test)[0]
    proxy = prompt_block_and_asr([rows[i] for i in test_idx], pred[is_test], success)
    held_auc = sc._safe_auc(col[is_test], y[is_test])
    return {"family": family, "normalize": cfg2.normalize, "gate_pct": gate_pct,
            "eval_mode": eval_mode, "scalarizer": best_k, "layer": best_layer,
            "train_auc": round(best_auc, 5), "held_out_auc": held_auc,
            "asr_before": proxy["asr_before"], "asr_after": proxy["asr_after"],
            "block_rate_among_successful": proxy["block_rate_among_successful"]}


def _run_offset(cfg: ExpConfig, off: int) -> dict:
    success = success_set(cfg.out_dir, cfg.asr_judge)
    arms = []
    for norm in ("none", "zscore", "rank"):
        arms.append(_eval_variant(replace(cfg, normalize=norm), off, "clean", 100.0, success))
    arms.append(_eval_variant(cfg, off, "borderline", 100.0, success))
    arms.append(_eval_variant(cfg, off, "clean", 30.0, success))    # sink-gate arm
    ablations = {"pos_offset": off, "n_arms": len(arms), "arms": arms}
    io.write_json(cfg.pos_dir(off) / "ablations.json", ablations)
    return ablations


def run(cfg: ExpConfig, lm=None) -> dict:
    summary = io.read_json(cfg.out_dir / "extract_summary.json")
    census = summary.get("census", {})
    counts = sorted(census.values())
    balance_ok = bool(counts and counts[0] == counts[-1])

    per_pos = {}
    for off in cfg.pos_offsets:
        ab = _run_offset(cfg, off)
        holdout = _safe_read(cfg.pos_dir(off) / "holdout_eval.json")
        perm = _safe_read(cfg.pos_dir(off) / "permutation_test.json")
        scauc = _safe_read(cfg.pos_dir(off) / "scalarizer_auc.json")
        clean = (holdout or {}).get("clean") or {}
        clean_test = clean.get("test") or {}
        best_ci = None
        if scauc and clean.get("scalarizer") in (scauc.get("per_scalarizer") or {}):
            best_ci = scauc["per_scalarizer"][clean["scalarizer"]].get("best_train_auc_ci95")
        per_pos[f"pos{off}"] = {
            "eval_mode": (holdout or {}).get("eval_mode"),
            "clean_scalarizer": clean.get("scalarizer"),
            "clean_layer": clean.get("layer"),
            "clean_held_out_auc": clean_test.get("auc"),
            "clean_held_out_tpr": clean_test.get("tpr"),
            "clean_held_out_fpr": clean_test.get("benign_fpr"),
            "permutation": perm,
            "train_auc_ci95": best_ci,
            "n_ablation_arms": ab["n_arms"],
        }

    aucs = [v["clean_held_out_auc"] for v in per_pos.values()
            if isinstance(v["clean_held_out_auc"], (int, float))]
    report = {
        "experiment": "experiments_hc_4_claude",
        "question": "Can a non-logistic scalar+threshold per-token defense generalise "
                    "to held-out and reduce ASR (beat the hc_2 0% held-out collapse)?",
        "balance": {"census": census, "all_seven_equal": balance_ok,
                    "balance_a": cfg.balance_a, "cap_mode": summary.get("cap_mode")},
        "scalarizer_set": cfg.scalarizer_set, "normalize": cfg.normalize,
        "per_pos": per_pos,
        "cross_pos_auc_spread": (round(max(aucs) - min(aucs), 5) if len(aucs) >= 2 else None),
    }
    io.write_json(cfg.out_dir / "robustness_report.json", report)
    io.write_text(cfg.out_dir / "robustness_report.md", _md(report))
    io.write_text(cfg.out_dir / "summary.md", _summary_md(report))
    print(f"[09] balance_all_seven_equal={balance_ok}; "
          f"per-pos clean held-out AUC={[per_pos[k]['clean_held_out_auc'] for k in per_pos]}")
    return report


def _safe_read(path: Path):
    try:
        return io.read_json(path)
    except Exception:
        return None


def _md(r: dict) -> str:
    lines = ["# hc_4_claude — Robustness Report", "",
             f"**Question:** {r['question']}", "",
             f"- 7-way balance (A-G equal counts): **{r['balance']['all_seven_equal']}** "
             f"(balance_a={r['balance']['balance_a']}, cap_mode={r['balance']['cap_mode']})",
             f"- census: {r['balance']['census']}",
             f"- scalarizer_set={r['scalarizer_set']}, normalize={r['normalize']}",
             f"- cross-pos held-out AUC spread: {r['cross_pos_auc_spread']}", "",
             "## Held-out (per pos_offset)", "",
             "| pos | eval | scalarizer | layer | AUC | TPR | benign FPR | perm p |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for pos, v in r["per_pos"].items():
        p = (v.get("permutation") or {}).get("p_value")
        lines.append(f"| {pos} | {v['eval_mode']} | {v['clean_scalarizer']} | "
                     f"{v['clean_layer']} | {v['clean_held_out_auc']} | {v['clean_held_out_tpr']} | "
                     f"{v['clean_held_out_fpr']} | {p} |")
    lines += ["", "See each `pos*/ablations.json` for the normalisation / family / sink-gate arms."]
    return "\n".join(lines) + "\n"


def _summary_md(r: dict) -> str:
    per_pos = ", ".join("{}:{}".format(k, v["clean_held_out_auc"])
                        for k, v in r["per_pos"].items())
    return ("# hc_4_claude — Summary\n\n"
            f"{r['question']}\n\n"
            f"- All seven types balanced: {r['balance']['all_seven_equal']} "
            f"({r['balance']['census']})\n"
            f"- Headline family: {r['scalarizer_set']} (borderline reported separately)\n"
            f"- Per-pos held-out AUC: {{ {per_pos} }}\n"
            "- See robustness_report.md and pos*/holdout_eval.json, defense_report.json.\n")


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
