"""Model-free smoke test — exercises every stage (00–10) + run_all with a fake model.

Runs ``run_all`` in ``--smoke`` mode (synthetic model/tokenizer, real labeling and
analysis code), then asserts each stage's outputs exist with the right schema —
the choan.md §0–§3.4 contract: a self-contained ``replacement.json``, ALL SEVEN
types A-G equalised, the §2.1 probe, the §2.2 clean+borderline signals and held-out
detector (cos_to_attack + diff_means), and the three §3 defense arms
(mask / steer / drop±1) each producing an ASR/flag report, plus the §3.4 report.
Finally it re-runs the model-free stages standalone to prove composition.

    python -m experiments_attack_attr.smoke_test    # exit 0 = OK, non-zero = fail
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

from experiments_attack_attr.config import ExpConfig  # noqa: E402
from experiments_attack_attr.core import io           # noqa: E402
import experiments_attack_attr.run_all as run_all     # noqa: E402

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
    # 7-way balance yields EXACT equality.
    cfg = ExpConfig(smoke=True, n=4, ordinary=4, out_dir=out_dir,
                    pos_offsets=[0], smoke_layers=4, smoke_dim=64, smoke_heads=4,
                    n_bootstrap=50, n_perm=50, cv_folds=3, scalarizer_set="all")

    print("\n########## running run_all (smoke) ##########")
    run_all.run(cfg)

    print("\n########## checking artifacts ##########")
    off = cfg.pos_offsets[0]
    pdir = cfg.pos_dir(off)

    # ---- 00 embedding + self-contained replacement ------------------------- #
    emb = io.read_json(out_dir / "embedding_analysis.json")
    check("separability_auc" in emb, "embedding_analysis has separability_auc (§1)")
    check((out_dir / "replacement.json").exists(),
          "stage 00 wrote a self-contained out_dir/replacement.json")

    # ---- 01 prompts -------------------------------------------------------- #
    prompts = io.read_jsonl(out_dir / "prompts.jsonl")
    variants = {r["variant"] for r in prompts}
    expected = {"malicious_special", "malicious_mimicry", "positioned_regular",
                "benign_mimicry", "benign_special", "ordinary"}
    check(expected <= variants, f"prompts.jsonl has all 6 variants ({variants})")
    check(all(r["slot_word"] for r in prompts if r["variant"] == "positioned_regular"),
          "F prompts carry slot_word")

    # ---- 02 ASR ------------------------------------------------------------ #
    asr = io.read_jsonl(out_dir / "asr.jsonl")
    check({r["variant"] for r in asr} <= {"malicious_mimicry", "malicious_special", "positioned_regular"},
          "ASR only on B/D/F")
    succ = {r["refusal_success"] for r in asr}
    check(True in succ and False in succ, "ASR has both success and refusal outcomes")

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

    # ---- 04 separability (§2.1) -------------------------------------------- #
    sep = io.read_json(pdir / "separability.json")
    check("per_layer" in sep and len(sep["per_layer"]) == cfg.smoke_layers + 1,
          "separability has a per-layer probe AUC for every layer")
    check(sep.get("best_layer") is not None, "separability picked a best probe layer")
    check((out_dir / "separability.csv").exists(), "separability.csv written")

    # ---- 05 scalars (§2.2 signals) ----------------------------------------- #
    sa = io.read_json(pdir / "scalarizer_auc.json")
    check(sa.get("fit_on") == "train", "stage 05 fits signals on TRAIN only")
    ps = sa.get("per_scalarizer", {})
    check(len(ps) >= 8, f"stage 05 evaluated many signals ({len(ps)})")
    check("cos_to_attack" in ps and "diff_means" in ps,
          "both the cos_to_attack headline and diff_means detector are present")
    check(any(v.get("borderline") for v in ps.values()) and
          any(not v.get("borderline") for v in ps.values()),
          "both clean and borderline signals tagged")
    z = np.load(pdir / "scalar_scores.npz")
    for k in ("mahalanobis_benign", "energy_lse", "pca_resid"):
        if "mat__" + k in z:
            check(np.isfinite(z["mat__" + k]).any(), f"{k} produced finite scores in smoke")
    fit = np.load(pdir / "scalarizer_fit.npz")
    check("dir__diff_means" in fit, "scalarizer_fit.npz holds the diff_means steering direction")

    # ---- 06 detector: thresholds (train) + held-out ------------------------ #
    op = io.read_json(pdir / "operating_points.json")
    check(op.get("selected_on") == "train", "operating point selected_on == train")
    check(op.get("clean") is not None, "clean (cos_to_attack family) op point exists")
    check(op.get("borderline") is not None or op.get("diff_means") is not None,
          "borderline / diff_means token-detector op point exists")
    he = io.read_json(pdir / "holdout_eval.json")
    check(he.get("eval_mode") in ("holdout", "in_sample"), f"eval_mode={he.get('eval_mode')}")
    ds = io.read_json(pdir / "detect_summary.json")
    check("clean_headline" in ds and "borderline_detector" in ds,
          "detect_summary reports clean headline + borderline detector")
    ts = io.read_json(pdir / "threshold_stability.json")
    any_cv = any("threshold_cv" in mm for v in ts["per_scalarizer"].values()
                 for mm in v["methods"].values())
    check(any_cv, "threshold_stability reports threshold_cv")
    check((pdir / "curves.json").exists() and (pdir / "permutation_test.json").exists(),
          "curves.json + permutation_test.json written")

    # ---- 07 mask (§3.1) ---------------------------------------------------- #
    dm = io.read_json(pdir / "defense_mask.json")
    check(all(k in dm for k in ("asr_before", "asr_after_proxy", "block_rate_among_successful")),
          "defense_mask has ASR before / proxy-after / block-rate-among-successful")
    check(not (pdir / "real_asr_mask.json").exists(), "mask real intervention NOT run in smoke")

    # ---- 08 steer (§3.2) --------------------------------------------------- #
    st = io.read_json(pdir / "defense_steer.json")
    check("flag_coverage" in st, "defense_steer reports flag coverage")
    check("real_note" in st, "steer records that it was not exercised under mock")

    # ---- 09 drop±1 (§3.3 HEADLINE) ----------------------------------------- #
    dd = io.read_json(pdir / "defense_drop_token_pm1.json")
    check(dd.get("action") == "drop_token_pm1", "drop defense uses drop_token_pm1 action")
    if dd["asr_before"] is not None and dd["asr_after_proxy"] is not None:
        check(dd["asr_after_proxy"] <= dd["asr_before"], "proxy ASR after <= before (held-out)")

    # ---- 10 report (§3.4) -------------------------------------------------- #
    fr = io.read_json(out_dir / "final_report.json")
    check((out_dir / "final_report.md").exists(), "final_report.md rendered (§3.4)")
    pp = fr.get("per_pos", {}).get(f"pos{off}", {})
    check(all(k in pp for k in ("s21_probe_best_auc", "s22_clean_headline",
                                "s31_mask", "s32_steer", "s33_drop_pm1")),
          "final_report aggregates §2.1/§2.2 + all three §3 defenses")

    # ---- model-free composition (re-run 04..10 from disk, no model) -------- #
    print("\n########## re-running stages 04..10 standalone (no model) ##########")
    for num in ("04", "05", "06", "07", "08", "09", "10"):
        run_all.load_stage(num).run(cfg, None)
    check(True, "stages 04-10 re-ran from disk artifacts without a model")

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
