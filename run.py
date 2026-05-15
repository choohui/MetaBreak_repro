"""MetaBreak TM-1 reproduction — single-shot orchestrator.

Runs the four stages back-to-back so an end-to-end attempt can be launched
with a single command:

    python run.py --model /path/to/Llama-3.1-8B-Instruct [--guard_model ...]

Stages
------
1. embedding.py  : search best (i, j, k) replacement triple, save replacement.json
2. mimicry.py    : rewrite Q_TM-1_Llama.txt, save prompt_mimicked.jsonl
3. attack.py     : run LLM on mimicked (and optionally baseline) prompts,
                   save responses.jsonl
4. evaluate.py   : refusal-keyword + (optional) Llama Guard judge,
                   save eval_report.json + eval_per_item.jsonl

Each stage is implemented in its own module and can also be invoked
standalone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import embedding as emb_mod
import mimicry as mim_mod
import attack as atk_mod
import evaluate as ev_mod


HERE = Path(__file__).resolve().parent
DEFAULT_PROMPTS = HERE / "Q_TM-1_Llama.txt"
DEFAULT_OUTDIR = HERE / "repro_mb_results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="Local HF path to Llama-3.1-8B-Instruct.")
    p.add_argument("--guard_model", default=None,
                   help="Local HF path to Llama-Guard-3-8B (optional).")
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS),
                   help="Path to Q_TM-1_Llama.txt.")
    p.add_argument("--out_dir", default=str(DEFAULT_OUTDIR),
                   help="Where every intermediate + final file goes.")
    p.add_argument("--n", type=int, default=10,
                   help="Number of prompts to attack (default 10).")
    p.add_argument("--topk", type=int, default=200,
                   help="Top-k candidate pool size per target token.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--skip_embedding", action="store_true",
                   help="Reuse existing replacement.json if present.")
    p.add_argument("--also_baseline", action="store_true",
                   help="Also generate responses for un-mimicked prompts.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repl_json = out_dir / "replacement.json"
    mim_jsonl = out_dir / "prompt_mimicked.jsonl"
    resp_jsonl = out_dir / "responses.jsonl"
    report_json = out_dir / "eval_report.json"
    per_item = out_dir / "eval_per_item.jsonl"

    # ----- Stage 1: embedding search ------------------------------------- #
    if args.skip_embedding and repl_json.exists():
        print(f"[run] re-using {repl_json}")
    else:
        print("=" * 70)
        print("[run] Stage 1/4: embedding search")
        print("=" * 70)
        sys.argv = [
            "embedding.py",
            "--model", args.model,
            "--output", str(repl_json),
            "--topk", str(args.topk),
            "--dtype", args.dtype,
        ]
        if args.device:
            sys.argv += ["--device", args.device]
        emb_mod.main()

    # ----- Stage 2: prompt mimicry --------------------------------------- #
    print("=" * 70)
    print("[run] Stage 2/4: prompt mimicry")
    print("=" * 70)
    sys.argv = [
        "mimicry.py",
        "--model", args.model,
        "--prompts", args.prompts,
        "--replacement", str(repl_json),
        "--output", str(mim_jsonl),
        "--n", str(args.n),
    ]
    mim_mod.main()

    # ----- Stage 3: attack ----------------------------------------------- #
    print("=" * 70)
    print("[run] Stage 3/4: LLM attack")
    print("=" * 70)
    sys.argv = [
        "attack.py",
        "--model", args.model,
        "--prompts", str(mim_jsonl),
        "--output", str(resp_jsonl),
        "--max_new_tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--dtype", args.dtype,
    ]
    if args.device:
        sys.argv += ["--device", args.device]
    if args.also_baseline:
        sys.argv += ["--also_baseline"]
    atk_mod.main()

    # ----- Stage 4: evaluate --------------------------------------------- #
    print("=" * 70)
    print("[run] Stage 4/4: evaluate")
    print("=" * 70)
    sys.argv = [
        "evaluate.py",
        "--responses", str(resp_jsonl),
        "--output", str(report_json),
        "--per_item", str(per_item),
        "--dtype", args.dtype,
    ]
    if args.guard_model:
        sys.argv += ["--guard_model", args.guard_model]
    if args.device:
        sys.argv += ["--device", args.device]
    ev_mod.main()

    print()
    print("[run] all stages done.")
    print(f"[run] final report : {report_json}")
    print(f"[run] per-item     : {per_item}")


if __name__ == "__main__":
    main()
