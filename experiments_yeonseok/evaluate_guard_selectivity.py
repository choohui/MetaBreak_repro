"""Evaluate attack blocking and benign pass-through for the MetaBreak guard."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metabreak_l2_guard import L2MimicryGuard  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(num: int, den: int) -> float | None:
    return None if den == 0 else round(100.0 * num / den, 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--attack_prompts", required=True)
    p.add_argument("--benign_prompts", default=str(HERE / "benign_prompts.jsonl"))
    p.add_argument("--output", required=True)
    p.add_argument("--per_item", required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--neighbor_rank", type=int, default=256)
    p.add_argument("--threshold_margin", type=float, default=0.0)
    p.add_argument("--structural_min_spans", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    out_path = Path(args.output)
    per_item_path = Path(args.per_item)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_item_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
    )
    guard = L2MimicryGuard.from_model(
        tokenizer,
        model,
        neighbor_rank=args.neighbor_rank,
        threshold_margin=args.threshold_margin,
        structural_min_spans=args.structural_min_spans,
    )
    del model

    rows = []
    attack_rows = load_jsonl(Path(args.attack_prompts))
    benign_rows = load_jsonl(Path(args.benign_prompts))

    for row in attack_rows:
        decision = guard.inspect_text(row["mimicked"])
        rows.append(
            {
                "split": "attack_mimicked",
                "id": row.get("idx"),
                "category": "metabreak_tm1",
                "blocked": decision["blocked"],
                "reason": decision["reason"],
                "decision": decision,
            }
        )

    for row in benign_rows:
        decision = guard.inspect_text(row["text"])
        rows.append(
            {
                "split": "benign",
                "id": row["id"],
                "category": row["category"],
                "blocked": decision["blocked"],
                "reason": decision["reason"],
                "decision": decision,
            }
        )

    with open(per_item_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    attack_total = sum(1 for row in rows if row["split"] == "attack_mimicked")
    attack_blocked = sum(1 for row in rows if row["split"] == "attack_mimicked" and row["blocked"])
    benign_total = sum(1 for row in rows if row["split"] == "benign")
    benign_blocked = sum(1 for row in rows if row["split"] == "benign" and row["blocked"])

    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "blocked": 0})
    for row in rows:
        if row["split"] != "benign":
            continue
        by_category[row["category"]]["total"] += 1
        by_category[row["category"]]["blocked"] += int(row["blocked"])

    report = {
        "guard": guard.metadata(),
        "attack_mimicked": {
            "total": attack_total,
            "blocked": attack_blocked,
            "block_rate": pct(attack_blocked, attack_total),
        },
        "benign": {
            "total": benign_total,
            "blocked": benign_blocked,
            "passed": benign_total - benign_blocked,
            "pass_rate": pct(benign_total - benign_blocked, benign_total),
            "false_positive_rate": pct(benign_blocked, benign_total),
        },
        "benign_by_category": {
            category: {
                "total": counts["total"],
                "blocked": counts["blocked"],
                "passed": counts["total"] - counts["blocked"],
                "pass_rate": pct(counts["total"] - counts["blocked"], counts["total"]),
            }
            for category, counts in sorted(by_category.items())
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
