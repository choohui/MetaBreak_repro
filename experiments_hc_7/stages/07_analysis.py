"""Stage 07 (model-free) — analyze the sweep/utility/amplify/controls outputs.

Builds, per pos_offset:
  * dose-response ASR(alpha) on the headline arm (attack, all) + utility(alpha);
  * alpha* (max ASR reduction within the over-refusal budget) with bootstrap CIs
    on baseline vs steered ASR and a paired permutation p-value for the reduction;
  * Pareto frontier (ASR-reduction vs over-refusal increase);
  * controls check (random vs attack) and layer-specificity (control-layer vs
    detector layer) from stage 06;
  * amplification trend (rescue rate vs +alpha) from stage 05;
  * head-to-head vs hc_4's token-exclusion block-rate proxy (defense_report.json).

Outputs: ``pos{off}/steer_analysis.json`` + aggregate ``out_dir/steer_analysis.json``.
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

from experiments_hc_7.config import ExpConfig, config_from_args, make_stage_parser  # noqa: E402
from experiments_hc_7.core import io, stats                                          # noqa: E402
from experiments_hc_7.stages import steer_common as sc                               # noqa: E402


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _per_prompt_success(records, vt, tm, alpha):
    out = {}
    for r in records:
        if r["vector_type"] == vt and r["token_mode"] == tm and float(r["alpha"]) == alpha:
            out[int(r["sample_index"])] = bool(r["success"])
    return out


def _hc4_baseline(cfg, off) -> dict:
    p = cfg.src_pos_dir(off) / "defense_report.json"
    if not p.exists():
        return {}
    d = io.read_json(p)
    return {"hc4_asr_before": d.get("asr_before"), "hc4_asr_after": d.get("asr_after"),
            "hc4_block_rate_among_successful": d.get("block_rate_among_successful")}


def _run_offset(cfg: ExpConfig, off: int) -> dict:
    pdir = cfg.pos_dir(off)
    meta = io.read_json(pdir / "build_vectors.json")
    asr_rows = sc.read_csv(pdir / "steer_sweep_asr.csv")
    util_rows = sc.read_csv(pdir / "steer_utility.csv")
    amp_rows = sc.read_csv(pdir / "amplify.csv")
    ctrl_rows = sc.read_csv(pdir / "controls.csv")
    records = io.read_jsonl(pdir / "steer_sweep.jsonl") if (pdir / "steer_sweep.jsonl").exists() else []

    # headline dose-response (attack, all)
    head = sorted([(_f(r["alpha"]), _f(r["asr"])) for r in asr_rows
                   if r["vector_type"] == "attack" and r["token_mode"] == "all"],
                  key=lambda t: (t[0] is None, t[0]))
    or_by_a = {_f(r["alpha"]): _f(r["over_refusal_rate"]) for r in util_rows if r["vector_type"] == "attack"}
    base_asr = next((a for (al, a) in head if al == 0.0), None)
    base_or = or_by_a.get(0.0, 0.0) or 0.0

    sel = sc.select_alpha_star(cfg, off)
    a_star = sel["alpha_star"]

    # bootstrap CIs + paired permutation (baseline vs alpha*) on per-prompt success
    ci = {}
    perm = {}
    if records and a_star is not None:
        s0 = _per_prompt_success(records, "attack", "all", 0.0)
        sa = _per_prompt_success(records, "attack", "all", float(a_star))
        common = sorted(set(s0) & set(sa))
        if common:
            arr0 = np.array([s0[i] for i in common], dtype=float)
            arra = np.array([sa[i] for i in common], dtype=float)
            ci = {"baseline": stats.bootstrap_rate_ci(arr0, cfg.n_bootstrap, cfg.seed),
                  "alpha_star": stats.bootstrap_rate_ci(arra, cfg.n_bootstrap, cfg.seed + 1)}
            perm = stats.permutation_delta_pvalue(arr0, arra, cfg.n_perm, cfg.seed)

    # Pareto frontier: defense side (alpha<0): reduction vs over-refusal increase
    pareto = []
    for al, asr_a in head:
        if al is None or al > 0 or asr_a is None:
            continue
        red = (base_asr - asr_a) if base_asr is not None else None
        oi = (or_by_a.get(al) - base_or) if or_by_a.get(al) is not None else None
        pareto.append({"alpha": al, "asr": asr_a, "asr_reduction": None if red is None else round(red, 5),
                       "over_refusal_increase": None if oi is None else round(oi, 5)})

    # controls + layer specificity
    ctrl = {}
    for r in ctrl_rows:
        ctrl[r["arm"]] = {"layer": _f(r["layer"]), "asr": _f(r["asr"]),
                          "delta_vs_baseline": _f(r.get("delta_vs_baseline"))}
    controls_check = None
    if "attack" in ctrl and "random" in ctrl:
        controls_check = {
            "attack_delta": ctrl["attack"]["delta_vs_baseline"],
            "random_delta": ctrl["random"]["delta_vs_baseline"],
            "random_is_inert": (ctrl["random"]["delta_vs_baseline"] is not None
                                and ctrl["attack"]["delta_vs_baseline"] is not None
                                and ctrl["random"]["delta_vs_baseline"] > ctrl["attack"]["delta_vs_baseline"] + 0.1),
        }
    layer_specificity = {k: v for k, v in ctrl.items() if k.startswith("control_layer")}

    amplification = sorted([{"vector_type": r["vector_type"], "alpha": _f(r["alpha"]),
                             "rescue_rate": _f(r["rescue_rate"])} for r in amp_rows],
                           key=lambda d: (d["vector_type"], d["alpha"] or 0))

    out = {
        "pos_offset": off, "layer": meta["layer"], "block_idx": meta["block_idx"],
        "rho": meta["rho"], "eval_mode": "holdout",
        "baseline_asr": base_asr, "alpha_star": a_star,
        "alpha_star_selection": sel,
        "asr_at_alpha_star": sel.get("asr_at_star"),
        "over_refusal_at_alpha_star": sel.get("over_refusal_at_star"),
        "dose_response": [{"alpha": al, "asr": a, "over_refusal_rate": or_by_a.get(al)} for al, a in head],
        "pareto_frontier": pareto,
        "bootstrap_ci": ci,
        "permutation_reduction": perm,
        "controls": ctrl,
        "controls_check": controls_check,
        "layer_specificity": layer_specificity,
        "amplification": amplification,
        "head_to_head": {"steering_asr_at_alpha_star": sel.get("asr_at_star"),
                         "baseline_asr": base_asr, **_hc4_baseline(cfg, off)},
    }
    io.write_json(pdir / "steer_analysis.json", out)
    print(f"[07] pos{off}: baseline_asr={base_asr} alpha*={a_star} "
          f"asr*={sel.get('asr_at_star')} perm_p={perm.get('p_value')}")
    return out


def run(cfg: ExpConfig, lm=None) -> dict:
    per = {f"pos{off}": _run_offset(cfg, off) for off in cfg.pos_offsets}
    io.write_json(cfg.out_dir / "steer_analysis.json", per)
    return per


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
