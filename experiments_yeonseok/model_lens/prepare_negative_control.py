"""Build non-L2 MetaBreak-shaped control prompts for model-lens checks.

The control keeps the fixed assistant-header skeleton positions, such as the
literal regular token `assistant` and the final newline token for Llama, but
uses ordinary regular-token strings in the special-token replacement positions.
This tests whether lens signals come from L2 semantic mimicry or merely from
the surface pattern `regular regular assistant regular newline-newline`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.mimicry import apply_mimicry  # noqa: E402
from src.model_configs import resolve_config  # noqa: E402


DEFAULT_CONTROL_STRINGS = ["The", " red", " blue"]


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_control_strings(raw: str) -> list[str]:
    parts = raw.split("|")
    if any(part == "" for part in parts):
        raise ValueError("control strings must be pipe-separated non-empty strings")
    return parts


def validate_control(tokenizer: Any, model_type: str, control_strs: list[str]) -> dict[str, Any]:
    cfg = resolve_config(model_type, tokenizer)
    if len(control_strs) != len(cfg.replace_positions):
        raise ValueError(
            f"Expected {len(cfg.replace_positions)} control strings for "
            f"{model_type}, got {len(control_strs)}"
        )

    parts: list[str | None] = [None] * cfg.expected_n_tokens
    for pos, fixed in zip(cfg.fixed_positions, cfg.fixed_strs):
        parts[pos] = fixed
    for pos, control in zip(cfg.replace_positions, control_strs):
        parts[pos] = control
    control_header = "".join(part or "" for part in parts)
    control_ids = tokenizer(control_header, add_special_tokens=False)["input_ids"]
    if len(control_ids) != cfg.expected_n_tokens:
        raise ValueError(
            "Control header does not retokenize to the assistant-header shape: "
            f"{control_ids}"
        )

    literal_ids = tokenizer(cfg.assistant_header, add_special_tokens=False)["input_ids"]
    for pos in cfg.fixed_positions:
        if control_ids[pos] != literal_ids[pos]:
            raise ValueError(
                f"Fixed position {pos} changed: got {control_ids[pos]}, "
                f"expected {literal_ids[pos]}"
            )
    for pos in cfg.replace_positions:
        if int(control_ids[pos]) in cfg.special_token_ids:
            raise ValueError(
                f"Control position {pos} produced special token id {control_ids[pos]}"
            )

    return {
        "model_type": cfg.model_type,
        "assistant_header": cfg.assistant_header,
        "target_token_strs": cfg.target_token_strs,
        "control_replacement_decoded": control_strs,
        "control_header": control_header,
        "control_header_token_ids": [int(x) for x in control_ids],
        "literal_header_token_ids": [int(x) for x in literal_ids],
        "replace_positions": cfg.replace_positions,
        "fixed_positions": cfg.fixed_positions,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--model_type", default="llama")
    p.add_argument("--source_prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=None)
    p.add_argument(
        "--control_strings",
        default="|".join(DEFAULT_CONTROL_STRINGS),
        help="Pipe-separated replacement strings, one per special position.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    control_strs = parse_control_strings(args.control_strings)
    metadata = validate_control(tokenizer, args.model_type, control_strs)

    rows = read_jsonl(Path(args.source_prompts), limit=args.n)
    out_rows = []
    for row in rows:
        original = row["original"]
        controlled = apply_mimicry(
            original,
            metadata["target_token_strs"],
            control_strs,
        )
        out_rows.append(
            {
                "idx": int(row["idx"]),
                "original": original,
                "mimicked": controlled,
                "control_type": "non_l2_regular_header",
                "control_metadata": metadata,
            }
        )

    write_jsonl(Path(args.output), out_rows)
    print(
        json.dumps(
            {
                "output": args.output,
                "n": len(out_rows),
                "metadata": metadata,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
