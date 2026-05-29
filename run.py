"""MetaBreak TM-1 reproduction — single-shot orchestrator.

Runs the four stages back-to-back so an end-to-end experiment can be launched
with a single command:

    python run.py --model_type llama --model /path/to/Llama-3.1-8B-Instruct

Stages
------
1. src.embedding  — search best regular-token replacement tuple, save replacement.json
2. src.mimicry    — rewrite prompts with the replacement, save prompt_mimicked.jsonl
3. src.attack     — run the victim LLM on mimicked (+ optional baseline) prompts,
                    save responses.jsonl
4. src.evaluate   — refusal-keyword + (optional) Llama Guard judge,
                    save eval_report.json + eval_per_item.jsonl

Each stage is imported from ``src/`` and invoked via its ``run(args)`` function —
no ``sys.argv`` mutation, no subprocess spawning. Stages can also be run
standalone:

    python -m src.embedding --model_type llama --model /path/...
    python -m src.mimicry   --model_type llama --model /path/...
    python -m src.attack    --model_type llama --model /path/...
    python -m src.evaluate  --model_type llama

Or via the Llama-only legacy entry points (backward compatible):

    python embedding.py --model /path/...   # auto-injects --model_type llama
    python mimicry.py   --model /path/...
    python attack.py    --model /path/...
    python evaluate.py
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from src.embedding import run as _emb_run
from src.mimicry import run as _mim_run
from src.attack import run as _atk_run
from src.evaluate import run as _ev_run


HERE = Path(__file__).resolve().parent


def _prompts_path(model_type: str) -> Path:
    """Default Q_TM-1_<Model>.txt path given a model_type slug."""
    suffix = model_type[:1].upper() + model_type[1:]
    return HERE / "prompts" / f"Q_TM-1_{suffix}.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model_type", default="llama",
        help=(
            "Model family slug (default: llama). "
            "Known: llama, qwen, gemma, phi. "
            "Unknown values trigger tokenizer-based auto-detection."
        ),
    )
    p.add_argument(
        "--model", required=True,
        help="Local HF path to the victim model (e.g. Llama-3.1-8B-Instruct).",
    )
    p.add_argument(
        "--guard_model", default=None,
        help="Local HF path to Llama-Guard-3-8B (optional; enables guard-based ASR).",
    )
    p.add_argument(
        "--prompts", default=None,
        help=(
            "Path to Q_TM-1_<Model>.txt. "
            "Default: prompts/Q_TM-1_<ModelType>.txt relative to repro_mb/."
        ),
    )
    p.add_argument(
        "--out_dir", default=None,
        help="Output directory. Default: results/<model_type>/ relative to repro_mb/.",
    )
    p.add_argument("--n", type=int, default=10,
                   help="Number of prompts to attack (default 10).")
    p.add_argument("--topk", type=int, default=200,
                   help="Top-k candidate pool size per special token (default 200).")
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Max tokens to generate per response (default 256).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature; 0.0 = greedy decoding (default).")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Model dtype (default bfloat16).")
    p.add_argument("--device", default=None,
                   help="cuda / cpu (auto-detect if omitted).")
    p.add_argument("--skip_embedding", action="store_true",
                   help="Reuse existing replacement.json if present (faster re-runs).")
    p.add_argument("--also_baseline", action="store_true",
                   help="Also generate responses for the un-mimicked originals.")
    return p.parse_args()


def _banner(stage: str, label: str) -> None:
    print("=" * 70)
    print(f"[run] {stage}: {label}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    model_type = args.model_type

    # ── Resolve output directory ─────────────────────────────────────────── #
    out_dir = Path(args.out_dir) if args.out_dir else HERE / "results" / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Intermediate file paths ──────────────────────────────────────────── #
    repl_json   = out_dir / "replacement.json"
    mim_jsonl   = out_dir / "prompt_mimicked.jsonl"
    resp_jsonl  = out_dir / "responses.jsonl"
    report_json = out_dir / "eval_report.json"
    per_item    = out_dir / "eval_per_item.jsonl"

    # ── Prompts file ─────────────────────────────────────────────────────── #
    prompts_path = Path(args.prompts) if args.prompts else _prompts_path(model_type)

    # ── Stage 1: embedding search ────────────────────────────────────────── #
    if args.skip_embedding and repl_json.exists():
        print(f"[run] --skip_embedding: re-using {repl_json}")
    else:
        _banner("Stage 1/4", "embedding search")
        _emb_run(Namespace(
            model_type=model_type,
            model=args.model,
            output=str(repl_json),
            topk=args.topk,
            dtype=args.dtype,
            device=args.device,
        ))

    # ── Stage 2: prompt mimicry ──────────────────────────────────────────── #
    _banner("Stage 2/4", "prompt mimicry")
    _mim_run(Namespace(
        model_type=model_type,
        model=args.model,
        prompts=str(prompts_path),
        replacement=str(repl_json),
        output=str(mim_jsonl),
        n=args.n,
    ))

    # ── Stage 3: LLM attack ──────────────────────────────────────────────── #
    _banner("Stage 3/4", "LLM attack")
    _atk_run(Namespace(
        model_type=model_type,
        model=args.model,
        prompts=str(mim_jsonl),
        output=str(resp_jsonl),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        dtype=args.dtype,
        also_baseline=args.also_baseline,
    ))

    # ── Stage 4: evaluate ───────────────────────────────────────────────── #
    _banner("Stage 4/4", "evaluate")
    _ev_run(Namespace(
        model_type=model_type,
        responses=str(resp_jsonl),
        output=str(report_json),
        per_item=str(per_item),
        guard_model=args.guard_model,
        device=args.device,
        dtype=args.dtype,
    ))

    print()
    print("[run] all stages done.")
    print(f"[run] results dir  : {out_dir}")
    print(f"[run] final report : {report_json}")
    print(f"[run] per-item     : {per_item}")


if __name__ == "__main__":
    main()
