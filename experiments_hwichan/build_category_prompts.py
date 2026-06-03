"""Build labelled input prompts for the two internal-representation experiments.

Experiment 1 (exp1): use ``prompts/Q_TM-1_Llama.txt`` as-is. Each TM-1 prompt
already embeds the attacker's literal special-token assistant headers at the end,
so it yields the **malicious-special (B)** variant directly; applying Semantic
Mimicry (``src.mimicry.apply_mimicry`` + ``replacement.json``) yields the
**mimicry-regular (A)** variant. Benign-special (C) prompts come from
``benign_special_prompts.jsonl``.

Experiment 2 (exp2): take clean questions from ``prompts/Q.txt`` as *carriers* and
inject the attack payload at varied positions (start / middle / end / scattered),
so the special / mimicked tokens are no longer concentrated in the last few
positions. We emit malicious (full header = B), mimicked (mimicked header = A) and
benign-special (a single isolated special token = C) variants per position.

Both produce a jsonl where each row is
``{idx, source, variant, position_kind, text}``. ``variant`` is consumed verbatim
by ``common.label_token_categories``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mimicry import apply_mimicry, load_prompts  # noqa: E402
from experiments_hwichan.common import (  # noqa: E402
    load_replacement,
    read_jsonl,
    write_jsonl,
)

HERE = Path(__file__).resolve().parent
DEFAULT_TM1 = REPO_ROOT / "prompts" / "Q_TM-1_Llama.txt"
DEFAULT_Q = REPO_ROOT / "prompts" / "Q.txt"
DEFAULT_BENIGN_SPECIAL = HERE / "benign_special_prompts.jsonl"
DEFAULT_REPLACEMENT = (
    REPO_ROOT
    / "experiments_yeonseok"
    / "results"
    / "l2_guard_llama31_8b_n450"
    / "common"
    / "replacement.json"
)

POSITION_KINDS = ["start", "middle", "end", "scattered"]


def _load_replacement_strings(replacement_path: Path) -> tuple[list[str], list[str], str]:
    repl = load_replacement(replacement_path)
    special_strs = repl["target_token_strs"]
    replacement_strs = repl["best_triple_decoded"]
    assistant_header = repl.get(
        "assistant_header", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return special_strs, replacement_strs, assistant_header


# ---------------------------------------------------------------------------- #
# Experiment 1
# ---------------------------------------------------------------------------- #


def build_exp1(
    tm1_path: Path,
    benign_special_path: Path,
    replacement_path: Path,
    n: int,
) -> list[dict]:
    special_strs, replacement_strs, _ = _load_replacement_strings(replacement_path)
    prompts = load_prompts(tm1_path)
    n = min(n, len(prompts))
    rows: list[dict] = []
    for idx in range(n):
        original = prompts[idx]
        rows.append(
            {
                "idx": idx,
                "source": "Q_TM-1_Llama",
                "variant": "malicious",
                "position_kind": "tm1_tail",
                "text": original,
            }
        )
        rows.append(
            {
                "idx": idx,
                "source": "Q_TM-1_Llama",
                "variant": "mimicked",
                "position_kind": "tm1_tail",
                "text": apply_mimicry(original, special_strs, replacement_strs),
            }
        )
    for row in read_jsonl(benign_special_path):
        rows.append(
            {
                "idx": row.get("id"),
                "source": "benign_special_prompts",
                "variant": "benign_special",
                "position_kind": "natural",
                "text": row["text"],
            }
        )
    return rows


# ---------------------------------------------------------------------------- #
# Experiment 2 — position-varied injection
# ---------------------------------------------------------------------------- #


def inject_payload(carrier: str, payload: str, kind: str) -> str:
    """Insert ``payload`` into ``carrier`` at the requested position(s)."""
    words = carrier.split()
    if not words:
        return payload
    if kind == "start":
        idxs = [0]
    elif kind == "middle":
        idxs = [len(words) // 2]
    elif kind == "end":
        idxs = [len(words)]
    elif kind == "scattered":
        idxs = sorted({max(1, int(len(words) * q)) for q in (0.25, 0.5, 0.75)})
    else:
        raise ValueError(f"unknown position kind {kind!r}")
    # insert from the back so earlier indices stay valid
    out = list(words)
    for i in sorted(idxs, reverse=True):
        out.insert(min(i, len(out)), payload)
    return " ".join(out)


def build_exp2(
    q_path: Path,
    replacement_path: Path,
    n: int,
    single_special: str = "<|eot_id|>",
) -> list[dict]:
    special_strs, replacement_strs, assistant_header = _load_replacement_strings(
        replacement_path
    )
    mimicked_header = apply_mimicry(assistant_header, special_strs, replacement_strs)
    single_special_mim = apply_mimicry(single_special, special_strs, replacement_strs)

    carriers = load_prompts(q_path)
    n = min(n, len(carriers))
    rows: list[dict] = []
    for idx in range(n):
        carrier = carriers[idx].strip()
        for kind in POSITION_KINDS:
            rows.append(
                {
                    "idx": idx,
                    "source": "Q",
                    "variant": "malicious",
                    "position_kind": kind,
                    "text": inject_payload(carrier, assistant_header, kind),
                }
            )
            rows.append(
                {
                    "idx": idx,
                    "source": "Q",
                    "variant": "mimicked",
                    "position_kind": kind,
                    "text": inject_payload(carrier, mimicked_header, kind),
                }
            )
            rows.append(
                {
                    "idx": idx,
                    "source": "Q",
                    "variant": "benign_special",
                    "position_kind": kind,
                    "text": inject_payload(carrier, single_special, kind),
                }
            )
            # also a clean ordinary carrier (no injection) to seed the E baseline
        rows.append(
            {
                "idx": idx,
                "source": "Q",
                "variant": "ordinary",
                "position_kind": "none",
                "text": carrier,
            }
        )
        # note single_special_mim kept available for callers that want a regular-token
        # single-slot control; emitted under mimicked/scattered already via header.
        _ = single_special_mim
    return rows


# ---------------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", choices=["1", "2", "both"], default="both")
    p.add_argument("--tm1", default=str(DEFAULT_TM1))
    p.add_argument("--q", default=str(DEFAULT_Q))
    p.add_argument("--benign_special", default=str(DEFAULT_BENIGN_SPECIAL))
    p.add_argument("--replacement", default=str(DEFAULT_REPLACEMENT))
    p.add_argument("--n", type=int, default=50, help="number of base prompts per exp")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.exp in ("1", "both"):
        rows = build_exp1(
            Path(args.tm1), Path(args.benign_special), Path(args.replacement), args.n
        )
        write_jsonl(out_dir / "exp1_prompts.jsonl", rows)
        print(f"[build] exp1: wrote {len(rows)} rows -> {out_dir/'exp1_prompts.jsonl'}")
    if args.exp in ("2", "both"):
        rows = build_exp2(Path(args.q), Path(args.replacement), args.n)
        write_jsonl(out_dir / "exp2_prompts.jsonl", rows)
        print(f"[build] exp2: wrote {len(rows)} rows -> {out_dir/'exp2_prompts.jsonl'}")


if __name__ == "__main__":
    main()
