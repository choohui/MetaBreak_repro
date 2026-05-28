"""Run no-defense vs L2-guarded MetaBreak experiments and summarize results."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_MODEL_TYPE = "llama"
DEFAULT_PROMPTS = REPO_ROOT / "prompts" / "Q_TM-1_Llama.txt"


def run_step(name: str, cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(cmd)}\n\n")
        log.flush()
        print(f"[runner] {name}: {shlex.join(cmd)}")
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        elapsed = time.time() - started
        log.write(f"\n[runner] exit_code={proc.returncode} elapsed_seconds={elapsed:.2f}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed: {name}. See {log_path}")


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return round(100.0 * num / den, 2)


def summarize(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    replacement_path: Path,
    no_defense_report: Path,
    defended_report: Path,
    defended_responses: Path,
    selectivity_report: Path,
) -> dict[str, Any]:
    replacement = load_json(replacement_path)
    no_defense = load_json(no_defense_report)
    defended = load_json(defended_report)
    selectivity = load_json(selectivity_report)
    defended_rows = load_jsonl(defended_responses)

    n = len(defended_rows)
    blocked_mimicked = sum(1 for r in defended_rows if r["defense_mimicked"]["blocked"])
    blocked_baseline = sum(
        1
        for r in defended_rows
        if r.get("defense_baseline") and r["defense_baseline"]["blocked"]
    )
    reason_counts: dict[str, int] = {}
    for row in defended_rows:
        reason = row["defense_mimicked"]["reason"] or "<none>"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    summary = {
        "model": args.model,
        "model_type": args.model_type,
        "guard_model": args.guard_model,
        "prompts": args.prompts,
        "n": args.n,
        "topk": args.topk,
        "neighbor_rank": args.neighbor_rank,
        "threshold_margin": args.threshold_margin,
        "structural_min_spans": args.structural_min_spans,
        "replacement": {
            "best_triple_ids": replacement["best_triple_ids"],
            "best_triple_decoded": replacement["best_triple_decoded"],
            "best_similarity_l2_sum": replacement["best_similarity_l2_sum"],
            "n_evaluated": replacement["n_evaluated"],
            "n_kept_5tok": replacement["n_kept_5tok"],
        },
        "no_defense_eval": no_defense,
        "defended_eval": defended,
        "selectivity_eval": selectivity,
        "defense_detection": {
            "n_items": n,
            "blocked_mimicked": blocked_mimicked,
            "blocked_mimicked_pct": pct(blocked_mimicked, n),
            "blocked_baseline": blocked_baseline,
            "blocked_baseline_pct": pct(blocked_baseline, n if args.also_baseline else 0),
            "mimicked_reason_counts": reason_counts,
        },
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_results_md(out_dir / "RESULTS.md", summary)
    return summary


def write_results_md(path: Path, summary: dict[str, Any]) -> None:
    no_def = summary["no_defense_eval"]
    defended = summary["defended_eval"]
    detection = summary["defense_detection"]
    selectivity = summary["selectivity_eval"]
    repl = summary["replacement"]
    lines = [
        "# L2 Mimicry Guard Results",
        "",
        "## Setup",
        "",
        f"- Model: `{summary['model']}`",
        f"- Model type: `{summary['model_type']}`",
        f"- Guard model: `{summary['guard_model']}`",
        f"- Prompts: first {summary['n']} from `{summary['prompts']}`",
        f"- Embedding search topk: {summary['topk']}",
        f"- Guard neighbor rank: {summary['neighbor_rank']}",
        f"- Guard threshold margin: {summary['threshold_margin']}",
        f"- Guard structural min spans: {summary['structural_min_spans']}",
        "",
        "## Replacement",
        "",
        f"- Best triple IDs: `{repl['best_triple_ids']}`",
        f"- Best triple decoded: `{repl['best_triple_decoded']}`",
        f"- L2 sum: `{repl['best_similarity_l2_sum']}`",
        f"- Evaluated triples: `{repl['n_evaluated']}`",
        f"- Retokenized 5-token triples kept: `{repl['n_kept_5tok']}`",
        "",
        "## Metrics",
        "",
        "| condition | refusal-keyword ASR mimicked | refusal-keyword ASR baseline | n |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| no defense | {no_def['asr_refusal_keyword_mimicked']} | "
            f"{no_def['asr_refusal_keyword_baseline']} | {no_def['n_total']} |"
        ),
        (
            f"| L2 guard | {defended['asr_refusal_keyword_mimicked']} | "
            f"{defended['asr_refusal_keyword_baseline']} | {defended['n_total']} |"
        ),
        "",
        "| condition | Llama Guard ASR mimicked | Llama Guard ASR baseline | guard used |",
        "| --- | ---: | ---: | --- |",
        (
            f"| no defense | {no_def['asr_llama_guard_mimicked']} | "
            f"{no_def['asr_llama_guard_baseline']} | {no_def['guard_model_used']} |"
        ),
        (
            f"| L2 guard | {defended['asr_llama_guard_mimicked']} | "
            f"{defended['asr_llama_guard_baseline']} | {defended['guard_model_used']} |"
        ),
        "",
        "## Detection",
        "",
        f"- Mimicked prompts blocked: {detection['blocked_mimicked']}/{detection['n_items']} "
        f"({detection['blocked_mimicked_pct']}%)",
        f"- Baseline literal-special prompts blocked: {detection['blocked_baseline']}/{detection['n_items']} "
        f"({detection['blocked_baseline_pct']}%)",
        f"- Mimicked reason counts: `{detection['mimicked_reason_counts']}`",
        "- Repeated regular assistant-header structure is recorded as an "
        "observation but does not block by itself.",
        "",
        "## Selectivity",
        "",
        "| split | total | blocked | passed | rate |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| attack mimicked | {selectivity['attack_mimicked']['total']} | "
            f"{selectivity['attack_mimicked']['blocked']} | "
            f"{selectivity['attack_mimicked']['total'] - selectivity['attack_mimicked']['blocked']} | "
            f"{selectivity['attack_mimicked']['block_rate']}% block |"
        ),
        (
            f"| benign safe text | {selectivity['benign']['total']} | "
            f"{selectivity['benign']['blocked']} | "
            f"{selectivity['benign']['passed']} | "
            f"{selectivity['benign']['pass_rate']}% pass |"
        ),
        "",
        "Benign set categories include ordinary safe requests, normal uses of the "
        "word `assistant`, a single harmless assistant-header-like line, and "
        "near-L2 replacement-token text without a chat-control skeleton.",
        "",
        "## Interpretation",
        "",
        "The guard is considered effective for this reproduction if it detects the "
        "regular-token assistant-header mimicry spans and lowers ASR relative to "
        "the no-defense condition under the same prompt set and replacement.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_type", default=DEFAULT_MODEL_TYPE)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    p.add_argument("--out_dir", default=str(HERE / "results" / "l2_guard"))
    p.add_argument("--n", type=int, default=450)
    p.add_argument("--topk", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--guard_model", default=None)
    p.add_argument("--also_baseline", action="store_true")
    p.add_argument("--skip_embedding", action="store_true")
    p.add_argument("--reuse_no_defense", action="store_true")
    p.add_argument("--neighbor_rank", type=int, default=256)
    p.add_argument("--threshold_margin", type=float, default=0.0)
    p.add_argument("--structural_min_spans", type=int, default=2)
    p.add_argument("--benign_prompts", default=str(HERE / "benign_prompts.jsonl"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    common_dir = out_dir / "common"
    no_def_dir = out_dir / "no_defense"
    defended_dir = out_dir / "defended"
    log_dir = out_dir / "logs"
    for path in (common_dir, no_def_dir, defended_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    replacement = common_dir / "replacement.json"
    prompts = common_dir / "prompt_mimicked.jsonl"
    no_def_responses = no_def_dir / "responses.jsonl"
    no_def_report = no_def_dir / "eval_report.json"
    no_def_per_item = no_def_dir / "eval_per_item.jsonl"
    defended_responses = defended_dir / "responses.jsonl"
    defended_report = defended_dir / "eval_report.json"
    defended_per_item = defended_dir / "eval_per_item.jsonl"
    selectivity_report = out_dir / "selectivity_report.json"
    selectivity_per_item = out_dir / "selectivity_per_item.jsonl"

    py = sys.executable
    if not args.skip_embedding or not replacement.exists():
        run_step(
            "embedding",
            [
                py,
                "src/embedding.py",
                "--model_type",
                args.model_type,
                "--model",
                args.model,
                "--output",
                str(replacement),
                "--topk",
                str(args.topk),
                "--dtype",
                args.dtype,
            ],
            log_dir / "01_embedding.log",
        )
    else:
        print(f"[runner] embedding: reusing {replacement}")

    run_step(
        "mimicry",
        [
            py,
            "src/mimicry.py",
            "--model_type",
            args.model_type,
            "--model",
            args.model,
            "--prompts",
            args.prompts,
            "--replacement",
            str(replacement),
            "--output",
            str(prompts),
            "--n",
            str(args.n),
        ],
        log_dir / "02_mimicry.log",
    )

    if args.reuse_no_defense and no_def_responses.exists() and no_def_report.exists():
        print(f"[runner] no_defense: reusing {no_def_responses} and {no_def_report}")
    else:
        attack_cmd = [
            py,
            "src/attack.py",
            "--model_type",
            args.model_type,
            "--model",
            args.model,
            "--prompts",
            str(prompts),
            "--output",
            str(no_def_responses),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--dtype",
            args.dtype,
        ]
        if args.device:
            attack_cmd += ["--device", args.device]
        if args.also_baseline:
            attack_cmd += ["--also_baseline"]
        run_step("attack_no_defense", attack_cmd, log_dir / "03_attack_no_defense.log")

        run_step(
            "evaluate_no_defense",
            [
                py,
                "src/evaluate.py",
                "--model_type",
                args.model_type,
                "--responses",
                str(no_def_responses),
                "--output",
                str(no_def_report),
                "--per_item",
                str(no_def_per_item),
                "--dtype",
                args.dtype,
            ]
            + (["--guard_model", args.guard_model] if args.guard_model else [])
            + (["--device", args.device] if args.device else []),
            log_dir / "04_evaluate_no_defense.log",
        )

    defended_cmd = [
        py,
        "experiments_yeonseok/defended_attack.py",
        "--model",
        args.model,
        "--prompts",
        str(prompts),
        "--output",
        str(defended_responses),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--dtype",
        args.dtype,
        "--neighbor_rank",
        str(args.neighbor_rank),
        "--threshold_margin",
        str(args.threshold_margin),
        "--structural_min_spans",
        str(args.structural_min_spans),
        "--replacement",
        str(replacement),
    ]
    if args.device:
        defended_cmd += ["--device", args.device]
    if args.also_baseline:
        defended_cmd += ["--also_baseline"]
    run_step("attack_defended", defended_cmd, log_dir / "05_attack_defended.log")

    run_step(
        "evaluate_defended",
        [
            py,
            "src/evaluate.py",
            "--model_type",
            args.model_type,
            "--responses",
            str(defended_responses),
            "--output",
            str(defended_report),
            "--per_item",
            str(defended_per_item),
            "--dtype",
            args.dtype,
        ]
        + (["--guard_model", args.guard_model] if args.guard_model else [])
        + (["--device", args.device] if args.device else []),
        log_dir / "06_evaluate_defended.log",
    )

    run_step(
        "evaluate_guard_selectivity",
        [
            py,
            "experiments_yeonseok/evaluate_guard_selectivity.py",
            "--model",
            args.model,
            "--attack_prompts",
            str(prompts),
            "--benign_prompts",
            args.benign_prompts,
            "--output",
            str(selectivity_report),
            "--per_item",
            str(selectivity_per_item),
            "--dtype",
            args.dtype,
            "--neighbor_rank",
            str(args.neighbor_rank),
            "--threshold_margin",
            str(args.threshold_margin),
            "--structural_min_spans",
            str(args.structural_min_spans),
            "--replacement",
            str(replacement),
        ],
        log_dir / "07_evaluate_guard_selectivity.log",
    )

    summary = summarize(
        args=args,
        out_dir=out_dir,
        replacement_path=replacement,
        no_defense_report=no_def_report,
        defended_report=defended_report,
        defended_responses=defended_responses,
        selectivity_report=selectivity_report,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[runner] wrote {out_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
