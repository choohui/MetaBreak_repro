"""Model-free smoke test — exercises every stage + run_all with a fake model.

Runs ``run_all`` in ``--smoke`` mode (synthetic model/tokenizer, real labeling and
analysis code), then asserts each stage's outputs exist with the right schema —
including the hc_4_claude requirements: ALL SEVEN types A-G equalised, the
fit-on-train provenance, the train-only operating-point selection, honest held-out
evaluation, the 5 counterfactual pairs, the ASR proxy (asr_after <= asr_before),
finite covariance-scalarizer scores, and the ablation arms. Finally it re-runs
stages 04-09 standalone (no model) to prove model-free composition.

    python -m experiments_hc_4_claude.smoke_test    # exit 0 = OK, non-zero = fail
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_4_claude.config import ExpConfig  # noqa: E402
from experiments_hc_4_claude.core import io           # noqa: E402
import experiments_hc_4_claude.run_all as run_all     # noqa: E402

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

    # pos_offsets=[0] so every category is present at the analysed offset and the
    # 7-way balance yields EXACT equality (the user's hard requirement) — the
    # assertion below would otherwise be sensitive to offset-1 coverage.
    cfg = ExpConfig(smoke=True, n=4, ordinary=4, out_dir=out_dir,
                    pos_offsets=[0], smoke_layers=4, smoke_dim=64, smoke_heads=4,
                    n_bootstrap=50, n_perm=50, cv_folds=3, scalarizer_set="all")

    print("\n########## running run_all (smoke) ##########")
    run_all.run(cfg)

    print("\n########## checking artifacts ##########")
    off = cfg.pos_offsets[0]
    pdir = cfg.pos_dir(off)

    # ---- 01 prompts -------------------------------------------------------- #
    prompts = io.read_jsonl(out_dir / "prompts.jsonl")
    variants = {r["variant"] for r in prompts}
    expected = {"malicious_special", "malicious_mimicry", "positioned_regular",
                "benign_mimicry", "benign_special", "ordinary"}
    check(expected <= variants, f"prompts.jsonl has all 6 variants ({variants})")
    f_rows = [r for r in prompts if r["variant"] == "positioned_regular"]
    check(all(r["slot_word"] for r in f_rows), "F prompts carry slot_word")

    # ---- 02 ASR ------------------------------------------------------------ #
    asr = io.read_jsonl(out_dir / "asr.jsonl")
    check({r["variant"] for r in asr} <= {"malicious_mimicry", "malicious_special", "positioned_regular"},
          "ASR only on B/D/F")
    succ = {r["refusal_success"] for r in asr}
    check(True in succ and False in succ, "ASR has both success and refusal outcomes")
    check(io.read_json(out_dir / "asr_summary.json").get("asr_judge_mode") == "keyword",
          "asr_summary records judge mode")

    # ---- 03 extraction: 7-way balance incl. A ------------------------------ #
    tokens = io.read_jsonl(out_dir / "tokens.jsonl")
    feats = np.load(out_dir / "features.npz")["hidden"]
    summary = io.read_json(out_dir / "extract_summary.json")
    check(len(tokens) == feats.shape[0], "tokens row count == hidden cube rows")
    check(feats.ndim == 3 and feats.shape[1] == cfg.smoke_layers + 1,
          f"hidden cube is [N, L+1, dim] = {feats.shape}")
    census = summary["census"]
    letters = {c.split('_')[0] for c in census}
    check(set("ABCDEFG") <= letters, f"all 7 categories present in census ({sorted(letters)})")
    counts = sorted(census.values())
    check(counts and counts[0] == counts[-1],
          f"ALL SEVEN types A-G have EQUAL counts (census={census})")
    check(summary.get("cap_mode") == "balanced7", "7-way balance cap mode (balanced7)")
    check("balanced_row_ids" in summary, "extract_summary has balanced_row_ids index")

    # ---- 04 scalarize ------------------------------------------------------ #
    sa = io.read_json(pdir / "scalarizer_auc.json")
    check(sa.get("fit_on") == "train", "stage 04 fits scalarizers on TRAIN only")
    ps = sa.get("per_scalarizer", {})
    check(len(ps) >= 8, f"stage 04 evaluated many scalarizers ({len(ps)})")
    check(all("per_layer" in v and "best_layer" in v for v in ps.values()),
          "each scalarizer has per_layer AUC + best_layer")
    check(any(v.get("borderline") for v in ps.values()) and
          any(not v.get("borderline") for v in ps.values()),
          "both clean and borderline scalarizers tagged")
    check((pdir / "scalar_scores.npz").exists(), "scalar_scores.npz written")
    # finite scores for covariance-based scalarizers (shrinkage guard works)
    z = np.load(pdir / "scalar_scores.npz")
    for k in ("mahalanobis_benign", "energy_lse", "pca_resid"):
        if "mat__" + k in z:
            m = z["mat__" + k]
            check(np.isfinite(m).any(), f"{k} produced finite scores in smoke")

    # ---- 05 thresholds + operating point ----------------------------------- #
    ts = io.read_json(pdir / "threshold_stability.json")
    any_cv = any("threshold_cv" in mm for v in ts["per_scalarizer"].values()
                 for mm in v["methods"].values())
    check(any_cv, "threshold_stability reports threshold_cv")
    op = io.read_json(pdir / "operating_points.json")
    check(op.get("selected_on") == "train", "operating point selected_on == train")
    clean = op.get("clean")
    check(clean is not None and all(clean.get(x) is not None for x in
          ("scalarizer", "layer", "method", "direction")),
          "clean operating point has scalarizer/layer/method/direction")
    check((pdir / "threshold_per_type.csv").exists(), "threshold_per_type.csv written")

    # ---- 06 held-out evaluation -------------------------------------------- #
    he = io.read_json(pdir / "holdout_eval.json")
    check(he.get("eval_mode") in ("holdout", "in_sample"), f"eval_mode={he.get('eval_mode')}")
    ctest = (he.get("clean") or {}).get("test") or {}
    check(ctest.get("n_pos", 0) > 0, "held-out TEST split contains attack (B/D) tokens")
    check(ctest.get("auc") is None or np.isfinite(ctest.get("auc")), "held-out AUC is finite/None")
    check((pdir / "curves.json").exists() and (pdir / "permutation_test.json").exists(),
          "curves.json + permutation_test.json written")

    # ---- 07 counterfactual ------------------------------------------------- #
    cf = io.read_json(pdir / "counterfactual_validation_report.json")
    pairs = {s["pair"] for s in cf["summary"]}
    check(pairs == {"B_minus_C", "B_minus_F", "D_minus_E", "D_minus_F", "F_minus_G"},
          f"counterfactual has all 5 pairs ({pairs})")
    check(all("paired_auc" in s for s in cf["summary"]), "each pair has paired_auc")

    # ---- 08 token-exclusion defense + ASR proxy ---------------------------- #
    dr = io.read_json(pdir / "defense_report.json")
    check(all(kk in dr for kk in ("asr_before", "asr_after", "block_rate_among_successful")),
          "defense_report has ASR before/after + block-rate-among-successful")
    if dr["asr_before"] is not None and dr["asr_after"] is not None:
        check(dr["asr_after"] <= dr["asr_before"], "proxy ASR after <= before (held-out)")
    check(not (pdir / "real_asr.json").exists(), "real intervention NOT run in smoke")

    # ---- 09 robustness / ablations ----------------------------------------- #
    ab = io.read_json(pdir / "ablations.json")
    check(ab.get("n_arms", 0) >= 2, f"ablations has >= 2 arms ({ab.get('n_arms')})")
    rr = io.read_json(out_dir / "robustness_report.json")
    check(rr["balance"]["all_seven_equal"] is True, "robustness confirms 7-way balance")
    pp = rr["per_pos"].get(f"pos{off}", {})
    check("permutation" in pp and "train_auc_ci95" in pp,
          "robustness per-pos has permutation p + bootstrap CI fields")

    # ---- 00 embedding ------------------------------------------------------ #
    check("separability_auc" in io.read_json(out_dir / "embedding_analysis.json"),
          "embedding_analysis has separability_auc")

    # ---- model-free composition (re-run 04->09 from disk) ------------------ #
    print("\n########## re-running stages 04->09 standalone (no model) ##########")
    for num in ("04", "05", "06", "07", "08", "09"):
        run_all.load_stage(num).run(cfg, None)
    check(True, "stages 04-09 re-ran from disk artifacts without a model")

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
