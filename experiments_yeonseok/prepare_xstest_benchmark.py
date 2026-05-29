"""Prepare XSTest for MetaBreak-guard selectivity evaluation.

XSTest has 250 safe prompts designed to expose over-refusal and 200 unsafe
contrast prompts. For this defense, we use:

- safe prompts as normal pass-through inputs;
- unsafe prompts with the MetaBreak TM-1 assistant-header suffix attached and
  then mimicked with the current replacement triple.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.build_prompts import FILLER_TURNS  # noqa: E402
from src.mimicry import apply_mimicry  # noqa: E402
from src.model_configs import KNOWN_HEADERS  # noqa: E402


XSTEST_REPO = "Paul/XSTest"
XSTEST_FILE = "xstest_prompts.csv"

DEFAULT_MODEL_TYPE = "llama"


def load_replacement(path: Path) -> tuple[list[str], list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        replacement = json.load(f)
    special_strs = replacement.get("target_token_strs")
    if not special_strs:
        special_strs = ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"]
    return special_strs, replacement["best_triple_decoded"]


def metabreak_tm1_suffix(model_type: str) -> str:
    if model_type not in KNOWN_HEADERS:
        raise ValueError(
            f"prepare_xstest_benchmark only supports registered model_type values; "
            f"got {model_type!r}"
        )
    assistant_header = KNOWN_HEADERS[model_type]["assistant_header"]
    return "".join(assistant_header + filler for filler in FILLER_TURNS)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replacement", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_type", default=DEFAULT_MODEL_TYPE)
    p.add_argument("--repo_id", default=XSTEST_REPO)
    p.add_argument("--filename", default=XSTEST_FILE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            repo_type="dataset",
        )
    )
    special_strs, replacement_strs = load_replacement(Path(args.replacement))
    tm1_suffix = metabreak_tm1_suffix(args.model_type)

    safe_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(row)
            label = row["label"].strip().lower()
            prompt = row["prompt"]
            base = {
                "id": int(row["id"]),
                "category": f"xstest_{row['type']}",
                "label": label,
                "focus": row.get("focus"),
                "note": row.get("note"),
            }
            if label == "safe":
                safe_rows.append({**base, "text": prompt})
            elif label == "unsafe":
                original = prompt + tm1_suffix
                mimicked = apply_mimicry(original, special_strs, replacement_strs)
                attack_rows.append(
                    {
                        **base,
                        "idx": int(row["id"]),
                        "original": original,
                        "mimicked": mimicked,
                    }
                )

    write_jsonl(out_dir / "xstest_safe.jsonl", safe_rows)
    write_jsonl(out_dir / "xstest_unsafe_metabreak_mimicked.jsonl", attack_rows)
    write_jsonl(out_dir / "xstest_raw.jsonl", raw_rows)

    summary = {
        "source_repo": args.repo_id,
        "source_file": args.filename,
        "source_cache_path": str(csv_path),
        "n_total": len(raw_rows),
        "n_safe": len(safe_rows),
        "n_unsafe_metabreak": len(attack_rows),
        "safe_jsonl": str(out_dir / "xstest_safe.jsonl"),
        "unsafe_metabreak_jsonl": str(out_dir / "xstest_unsafe_metabreak_mimicked.jsonl"),
    }
    with open(out_dir / "xstest_prepare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
