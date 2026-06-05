"""Load hc_2-compatible artifacts for hc_3 analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import io


def _is_success(row: dict, judge: str) -> bool:
    ref = bool(row.get("refusal_success"))
    guard = row.get("guard_success")
    if judge == "keyword":
        return ref
    if judge == "guard":
        return bool(guard) if guard is not None else ref
    return ref or bool(guard)


def success_set(out_dir: Path, judge: str) -> set[int]:
    path = Path(out_dir) / "asr.jsonl"
    if not path.exists():
        return set()
    return {int(r["sample_index"]) for r in io.read_jsonl(path)
            if _is_success(r, judge)}


def load_rows(out_dir: Path, balanced: bool) -> list[dict]:
    rows = io.read_jsonl(Path(out_dir) / "tokens.jsonl")
    if not balanced:
        return rows
    summary_path = Path(out_dir) / "extract_summary.json"
    if not summary_path.exists():
        return rows
    summary = io.read_json(summary_path)
    keep = set(summary.get("balanced_row_ids", [r["row_id"] for r in rows]))
    return [r for r in rows if int(r["row_id"]) in keep]


def load_hidden(out_dir: Path) -> np.ndarray:
    path = Path(out_dir) / "features.npz"
    if not path.exists():
        return np.zeros((0, 0, 0), dtype=np.float16)
    return np.load(path)["hidden"]


def load_prompts(out_dir: Path) -> list[dict]:
    path = Path(out_dir) / "prompts.jsonl"
    return io.read_jsonl(path) if path.exists() else []

