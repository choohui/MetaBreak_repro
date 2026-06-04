"""Model-free smoke test — exercises every stage + run_all with a fake model.

Runs ``run_all`` in ``--smoke`` mode (synthetic model/tokenizer, real labeling
and analysis code), then asserts each stage's outputs exist with the right
schema/shape — including the hc_2 additions: C captured from the start, the
raw-vs-balanced census, the naive+grouped probe AUC, stage-05 operating points,
the §3 sink-filter sweep, and the §4 cascade report.

    python -m experiments_hc_2.smoke_test      # exit 0 = OK, non-zero = fail
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # .../repro_mb/experiments_hc_2
REPO_ROOT = HERE.parent                          # .../repro_mb
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_2.config import ExpConfig  # noqa: E402
from experiments_hc_2.core import io  # noqa: E402
import experiments_hc_2.run_all as run_all  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok  - {msg}")
    else:
        print(f"  FAIL- {msg}")
        FAILS.append(msg)


def main() -> int:
    out_dir = HERE / "results" / "_smoke"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cfg = ExpConfig(smoke=True, n=4, ordinary=4, out_dir=out_dir,
                    smoke_layers=4, smoke_dim=64, smoke_heads=4)

    print("\n########## running run_all (smoke) ##########")
    run_all.run(cfg)

    print("\n########## checking artifacts ##########")
    # ---- 01 prompts -------------------------------------------------------- #
    prompts = io.read_jsonl(out_dir / "prompts.jsonl")
    variants = {r["variant"] for r in prompts}
    expected_variants = {"malicious_special", "malicious_mimicry", "positioned_regular",
                         "benign_mimicry", "benign_special", "ordinary"}
    check(expected_variants <= variants, f"prompts.jsonl has all 6 variants ({variants})")
    f_rows = [r for r in prompts if r["variant"] == "positioned_regular"]
    check(all(r["slot_word"] for r in f_rows), "F prompts carry slot_word")

    # ---- 02 ASR ------------------------------------------------------------ #
    asr = io.read_jsonl(out_dir / "asr.jsonl")
    asr_variants = {r["variant"] for r in asr}
    check(asr_variants <= {"malicious_mimicry", "malicious_special", "positioned_regular"},
          f"ASR only on B/D/F ({asr_variants})")
    succ = {r["refusal_success"] for r in asr}
    check(True in succ and False in succ, "ASR has both success and refusal outcomes")
    asr_sum = io.read_json(out_dir / "asr_summary.json")
    check(asr_sum.get("asr_judge_mode") == "keyword", "asr_summary records judge mode")

    # ---- 03 extraction (C from the start + balanced/raw census) ------------ #
    tokens = io.read_jsonl(out_dir / "tokens.jsonl")
    feats = np.load(out_dir / "features.npz")["hidden"]
    summary = io.read_json(out_dir / "extract_summary.json")
    check(len(tokens) == feats.shape[0], "tokens row count == hidden cube rows")
    check(feats.ndim == 3 and feats.shape[1] == cfg.smoke_layers + 1,
          f"hidden cube is [N, L+1, dim] = {feats.shape}")
    census = summary["census"]                 # balanced view
    raw_census = summary.get("raw_census", {})  # full set
    letters = {c.split('_')[0] for c in census}
    check(set("ABCDEFG") <= letters, f"all 7 categories present in census ({sorted(letters)})")
    from experiments_hc_2.core.labels import CAT_C
    check(census.get(CAT_C, 0) > 0, f"C (benign mimicry) census > 0 (={census.get(CAT_C, 0)})")
    check("raw_census" in summary and "cap_applied" in summary and "cap_mode" in summary,
          "extract_summary has raw_census + cap_applied + cap_mode")
    check("balanced_row_ids" in summary, "extract_summary has balanced_row_ids index")
    check(summary.get("cap_mode") == "balanced", "balanced cap mode applied by default")
    # full tokens.jsonl == full hidden cube; balanced subset is an index into it
    check(len(tokens) == summary["n_rows"], "tokens.jsonl is the FULL set")
    check(summary["n_balanced_rows"] <= summary["n_rows"], "balanced subset <= full set")
    check(all(census.get(k, 0) <= raw_census.get(k, 0) for k in raw_census),
          "balanced census <= raw census per type")

    # ---- 04 probe + cosine (naive + grouped AUC) --------------------------- #
    for off in cfg.pos_offsets:
        pdir = cfg.pos_dir(off)
        rm = io.read_json(pdir / "representation_metrics.json")
        check(any("probe_auc" in r for r in rm["per_layer"]),
              f"pos{off} representation_metrics has probe_auc")
        check(any("probe_auc_grouped" in r for r in rm["per_layer"]),
              f"pos{off} representation_metrics has prompt-level probe_auc_grouped")
        cp = io.read_json(pdir / "cosine_pairs.json")
        check(len(cp["by_pair"]) == 6, f"pos{off} cosine_pairs has 6 pairs")
        check((pdir / "ref_centroids.npz").exists(), f"pos{off} ref_centroids.npz written")

    # ---- 05 threshold + operating points ----------------------------------- #
    for off in cfg.pos_offsets:
        pdir = cfg.pos_dir(off)
        td = io.read_json(pdir / "threshold_defense.json")
        sigs = set(td["per_signal"].keys())
        check({"hidden_norm", "sink", "value_norm", "output_norm", "cos_to_ref"} <= sigs,
              f"pos{off} threshold_defense covers all 5 signals ({sigs})")
        check((pdir / "threshold_per_type.csv").exists(), f"pos{off} threshold_per_type.csv")
        op = io.read_json(pdir / "operating_points.json")
        check(op.get("best_signal") is not None, f"pos{off} operating_points has a best_signal")
        bs = op["best_signal"]
        check("threshold_at_fpr" in bs and "direction" in bs,
              f"pos{off} best_signal has threshold_at_fpr + direction")

    # ---- 06 sink-filter sweep (§3) ----------------------------------------- #
    for off in cfg.pos_offsets:
        sf = io.read_json(cfg.pos_dir(off) / "sink_filter_report.json")
        check(len(sf["sweep"]) >= 1, f"pos{off} sink_filter sweep has entries")
        ok = all(s["n_reduced"] <= s["n_full"] for s in sf["sweep"])
        check(ok, f"pos{off} every gated set <= full")
        check("bd_recall" in sf["sweep"][0], f"pos{off} sweep entries report B/D recall")

    # ---- 07 cascade defense (§4) ------------------------------------------- #
    cr = io.read_json(cfg.pos_dir(cfg.cascade_pos_offset) / "cascade_report.json")
    check({"one_stage_threshold", "gate_only", "cascade"} <= set(cr["strategies"]),
          "cascade_report has all 3 strategies")
    check(cr.get("eval_mode") in ("holdout", "in_sample"), f"cascade eval_mode={cr.get('eval_mode')}")
    casc = cr["strategies"]["cascade"]
    pt = casc["per_type"]
    check(set(pt) <= set("ABCDEFG") and len(pt) > 0, "cascade per_type letters valid")
    check(("B" in pt) or ("D" in pt), "cascade test split contains attack (B/D) tokens")
    p = casc["prompt"]
    check("asr_before" in p and "asr_after" in p and "block_rate_among_successful" in p,
          "cascade reports ASR before/after + block-rate-among-successful")
    if p["asr_before"] is not None and p["asr_after"] is not None:
        check(p["asr_after"] <= p["asr_before"], "cascade ASR after <= before (held-out)")

    # ---- 00 embedding ------------------------------------------------------ #
    ea = io.read_json(out_dir / "embedding_analysis.json")
    check("separability_auc" in ea, "embedding_analysis has separability_auc")

    # ---- stage composition (no model) ------------------------------------- #
    print("\n########## re-running stages 05->06->07 standalone (no model) ##########")
    run_all.load_stage("05").run(cfg, None)
    run_all.load_stage("06").run(cfg, None)
    run_all.load_stage("07").run(cfg, None)
    check(True, "stages 05/06/07 re-ran from disk artifacts without a model")

    print("\n########## summary ##########")
    if FAILS:
        print(f"SMOKE TEST FAILED: {len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("OK - all smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
