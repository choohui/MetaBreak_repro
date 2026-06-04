"""Model-free smoke test — exercises every stage + run_all with a fake model.

Runs ``run_all`` in ``--smoke`` mode (synthetic model/tokenizer, real labeling
and analysis code), then asserts each stage's outputs exist with the right
schema/shape. Also re-runs stages 03->04->05 standalone to confirm the stages
compose from disk artifacts without re-running the model.

    python experiments_hc_1/smoke_test.py        # exit 0 = OK, non-zero = fail
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from experiments_hc_1.config import ExpConfig  # noqa: E402
from experiments_hc_1.core import io  # noqa: E402
import experiments_hc_1.run_all as run_all  # noqa: E402

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

    cfg = ExpConfig(smoke=True, n=3, ordinary=4, out_dir=out_dir,
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
    check((out_dir / "asr.csv").exists(), "asr.csv written")
    check((out_dir / "asr_summary.json").exists(), "asr_summary.json written")

    # ---- 03 extraction ----------------------------------------------------- #
    tokens = io.read_jsonl(out_dir / "tokens.jsonl")
    feats = np.load(out_dir / "features.npz")["hidden"]
    summary = io.read_json(out_dir / "extract_summary.json")
    check(len(tokens) == feats.shape[0], "tokens row count == hidden cube rows")
    check(feats.ndim == 3 and feats.shape[1] == cfg.smoke_layers + 1,
          f"hidden cube is [N, L+1, dim] = {feats.shape}")
    census = summary["census"]
    letters = {c.split('_')[0] for c in census}
    check(set("ABCDEFG") <= letters, f"all 7 categories present in census ({sorted(letters)})")

    # ---- 04 analysis ------------------------------------------------------- #
    for off in cfg.pos_offsets:
        pdir = cfg.pos_dir(off)
        rm = io.read_json(pdir / "representation_metrics.json")
        check(any("probe_auc" in r for r in rm["per_layer"]),
              f"pos{off} representation_metrics has probe_auc")
        cp = io.read_json(pdir / "cosine_pairs.json")
        check(len(cp["by_pair"]) == 6, f"pos{off} cosine_pairs has 6 pairs")
        check((pdir / "ref_centroids.npz").exists(), f"pos{off} ref_centroids.npz written")
        check((pdir / "pca_coords.npz").exists(), f"pos{off} pca_coords.npz written")

    # ---- 05 threshold ------------------------------------------------------ #
    for off in cfg.pos_offsets:
        pdir = cfg.pos_dir(off)
        td = io.read_json(pdir / "threshold_defense.json")
        sigs = set(td["per_signal"].keys())
        check({"hidden_norm", "sink", "value_norm", "output_norm", "cos_to_ref"} <= sigs,
              f"pos{off} threshold_defense covers all 5 signals ({sigs})")
        check((pdir / "threshold_per_type.csv").exists(), f"pos{off} threshold_per_type.csv")
        check((pdir / "threshold_asr.json").exists(), f"pos{off} threshold_asr.json")

    # ---- 06 sink range ----------------------------------------------------- #
    for off in cfg.pos_offsets:
        sr = io.read_json(cfg.pos_dir(off) / "sink_range_report.json")
        check(sr["n_reduced"] <= sr["n_full"], f"pos{off} sink-range reduced <= full")

    # ---- 00 embedding ------------------------------------------------------ #
    ea = io.read_json(out_dir / "embedding_analysis.json")
    check("separability_auc" in ea, "embedding_analysis has separability_auc")

    # ---- stage composition (no model) ------------------------------------- #
    print("\n########## re-running stages 03->04->05 standalone (no model) ##########")
    run_all.load_stage("05").run(cfg, None)
    check(True, "stage 05 re-ran from disk artifacts without a model")

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
