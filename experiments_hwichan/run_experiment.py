"""End-to-end orchestrator: build -> extract -> analyze -> defense.

Loads the victim model **once** and reuses it across both experiments.

Experiment 1 keeps the TM-1 attack tokens at the prompt tail (as in
``Q_TM-1_Llama.txt``); Experiment 2 scatters the injected special / mimicked
tokens across start / middle / end / multiple positions so that findings are
not an artefact of the last-few-token positions.

Example (GPU server):

    cd repro_mb
    python experiments_hwichan/run_experiment.py \
        --model <local-llama31-8b-snapshot> \
        --replacement experiments_yeonseok/results/l2_guard_llama31_8b_n450/common/replacement.json \
        --exp both --n 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_hwichan import build_category_prompts as build  # noqa: E402
from experiments_hwichan import analyze_representations as analyze  # noqa: E402
from experiments_hwichan import defense_thresholds as defense  # noqa: E402
from experiments_hwichan.common import load_model, write_json, write_jsonl  # noqa: E402
from experiments_hwichan.extract_representations import extract  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = HERE / "results"


def run_one(lm, exp: str, prompt_rows: list[dict], out_dir: Path, pos_offsets) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / f"exp{exp}_prompts.jsonl", prompt_rows)

    token_rows, hidden_arr = extract(lm, prompt_rows, desc=f"exp{exp} forward")
    write_jsonl(out_dir / "tokens.jsonl", token_rows)
    np.savez_compressed(out_dir / "features.npz", hidden=hidden_arr)

    census: dict[str, int] = {}
    for r in token_rows:
        census[r["category"]] = census.get(r["category"], 0) + 1
    write_json(
        out_dir / "extract_summary.json",
        {
            "exp": exp,
            "n_prompt_rows": len(prompt_rows),
            "n_token_rows": len(token_rows),
            "hidden_shape": list(hidden_arr.shape),
            "category_census": census,
        },
    )
    print(f"[exp{exp}] tokens={len(token_rows)} hidden={list(hidden_arr.shape)} census={census}")

    out = {"census": census, "analyze": {}, "defense": {}}
    for off in pos_offsets:
        sub = out_dir / (f"pos{off}")
        sub.mkdir(parents=True, exist_ok=True)
        # analyze/defense read tokens.jsonl & features.npz from the parent out_dir,
        # so symlink-free: point them at out_dir but write their own files there.
        # We run them on out_dir and tag pos_offset; outputs differ by suffix.
        try:
            out["analyze"][off] = analyze.analyze(out_dir, pos_offset=off)
            (out_dir / f"representation_metrics.json").replace(
                sub / "representation_metrics.json"
            )
            if (out_dir / "representation_metrics.csv").exists():
                (out_dir / "representation_metrics.csv").replace(
                    sub / "representation_metrics.csv"
                )
            if (out_dir / "pca_coords.npz").exists():
                (out_dir / "pca_coords.npz").replace(sub / "pca_coords.npz")
        except SystemExit as e:
            print(f"[exp{exp}] analyze pos{off} skipped: {e}")
        try:
            out["defense"][off] = defense.evaluate(out_dir, pos_offset=off)
            (out_dir / "defense_report.json").replace(sub / "defense_report.json")
            (out_dir / "defense_report.md").replace(sub / "defense_report.md")
        except SystemExit as e:
            print(f"[exp{exp}] defense pos{off} skipped: {e}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--model_type", default="llama")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--exp", choices=["1", "2", "both"], default="both")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--replacement", default=str(build.DEFAULT_REPLACEMENT))
    p.add_argument("--tm1", default=str(build.DEFAULT_TM1))
    p.add_argument("--q", default=str(build.DEFAULT_Q))
    p.add_argument("--benign_special", default=str(build.DEFAULT_BENIGN_SPECIAL))
    p.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--pos_offsets", default="0,1", help="comma list among 0,1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    pos_offsets = [int(x) for x in args.pos_offsets.split(",") if x.strip() != ""]

    lm = load_model(args.model, args.model_type, args.dtype, args.device)

    if args.exp in ("1", "both"):
        rows = build.build_exp1(
            Path(args.tm1), Path(args.benign_special), Path(args.replacement), args.n
        )
        run_one(lm, "1", rows, out_root / "exp1_llama31_8b", pos_offsets)
    if args.exp in ("2", "both"):
        rows = build.build_exp2(Path(args.q), Path(args.replacement), args.n)
        run_one(lm, "2", rows, out_root / "exp2_llama31_8b", pos_offsets)


if __name__ == "__main__":
    main()
