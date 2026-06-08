from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments_hc_6.config import ExpConfig  # noqa: E402
from experiments_hc_6.core import io  # noqa: E402
import experiments_hc_6.run_all as run_all  # noqa: E402


FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok  - {msg}")
    else:
        print(f"  FAIL- {msg}")
        FAILS.append(msg)


def _check_dependency_rule() -> None:
    cmd = ["rg", "-n", r"from experiments_|import experiments_", str(HERE)]
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    except FileNotFoundError:
        check(False, "rg is available for dependency rule check")
        return
    bad = []
    for line in proc.stdout.splitlines():
        if "experiments_hc_6" not in line and "smoke_test.py" not in line:
            bad.append(line)
    check(not bad, "no imports from older experiments_* packages")
    if bad:
        for line in bad[:20]:
            print("    " + line)


def main() -> int:
    out_dir = HERE / "results" / "_smoke"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cfg = ExpConfig(
        smoke=True,
        n=3,
        out_dir=out_dir,
        run_name="_smoke",
        pos_offsets=[0],
        max_new_tokens=32,
        mask_candidate_k=8,
        mask_top_n=4,
        steer_alphas=[0.0, 0.5],
        steer_modes=["add", "project_out", "pull_to_benign"],
        semantic_eval_n=8,
    )
    print("\n########## running experiments_hc_6 smoke ##########")
    run_all.run(cfg)

    print("\n########## checking artifacts ##########")
    for name in [
        "prepare_data_summary.json",
        "prompts.jsonl",
        "tokens.jsonl",
        "balanced_tokens.jsonl",
        "scalar_values.npz",
        "threshold_rules.json",
        "counterfactual_report.json",
        "mask_candidates.json",
        "mask_candidate_eval.json",
        "mask_eval.json",
        "steering_vectors.npz",
        "steering_fit.json",
        "steering_eval.json",
        "stress_report.json",
        "report.md",
        "metrics.json",
    ]:
        check((out_dir / name).exists(), f"{name} exists")

    tokens = io.read_jsonl(out_dir / "tokens.jsonl")
    hidden = np.load(out_dir / "features.npz")["hidden"]
    check(len(tokens) == hidden.shape[0], "features hidden rows align with tokens")
    check(hidden.ndim == 3 and hidden.shape[1] > 1 and hidden.shape[2] > 0, f"hidden cube shape {hidden.shape}")
    letters = {r["letter"] for r in io.read_jsonl(out_dir / "balanced_tokens.jsonl")}
    check(set("ABCDEFG") <= letters, f"balanced rows contain A-G letters ({sorted(letters)})")

    mask_eval = io.read_json(out_dir / "mask_eval.json")
    check("unk_or_eos_mask" in mask_eval.get("actions", {}), "mask eval includes unk/eos baseline")
    check(any(k.startswith("mask__") for k in mask_eval.get("actions", {})), "mask eval includes non-unk candidates")
    steering = io.read_json(out_dir / "steering_eval.json")
    check(bool(steering.get("grid")), "steering grid is non-empty")
    check((out_dir / "intervention_eval.csv").exists(), "intervention_eval.csv written")
    _check_dependency_rule()

    print("\n########## summary ##########")
    if FAILS:
        print(f"SMOKE TEST FAILED: {len(FAILS)} check(s) failed")
        return 1
    print("OK - all smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
