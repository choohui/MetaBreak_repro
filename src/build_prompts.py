"""MetaBreak TM-1 reproduction — Stage 0: build Q_TM-1_{model_type}.txt.

Reads `prompts/Q.txt` (450 prompts separated by `\\ntest split\\n`, copied
from `MetaBreak/prompts/Q.txt`) and wraps each one in the family-specific
fake-assistant pattern used by MetaBreak:

    {Q}{H}Sure,{H}here{H}is

where `H` is the family's `assistant_header` (see `model_configs.py`).
The pattern is family-agnostic — only `H` changes — and reproduces the
file layout of `MetaBreak/prompts/Q_TM-1_Llama.txt`.

For known families the header is hard-coded; for unknown ones it is
auto-detected from the tokenizer's chat template, so the script needs an
HF model path even when only generating prompts.

CLI
---
    python -m src.build_prompts --model_type llama --model /path/to/Llama
    python -m src.build_prompts --model_type qwen  --model /path/to/Qwen \\
        --source prompts/Q.txt --output prompts/Q_TM-1_Qwen.txt --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .model_configs import (
        KNOWN_HEADERS,
        auto_detect_config,
        resolve_config,
    )
except ImportError:  # standalone-script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_configs import (  # type: ignore[no-redef]
        KNOWN_HEADERS,
        auto_detect_config,
        resolve_config,
    )


HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parent
DEFAULT_SOURCE = REPRO_ROOT / "prompts" / "Q.txt"
PROMPTS_DIR = REPRO_ROOT / "prompts"

# MetaBreak's fake-turn filler tokens. The three short strings each look
# like the start of a compliant assistant reply, so chaining them with the
# family boundary tokens primes the model to continue in a compliant tone.
FILLER_TURNS = ["Sure,", "here", "is"]


def load_questions(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    parts = raw.split("\ntest split\n")
    return [p for p in parts if p.strip()]


def wrap_one(question: str, assistant_header: str) -> str:
    """Build one `Q_TM-1_X.txt` entry: question + 3 fake assistant boundaries."""
    out = question.rstrip("\n")
    for filler in FILLER_TURNS:
        out += assistant_header + filler
    return out


def build_corpus(questions: list[str], assistant_header: str) -> str:
    """Join wrapped questions with the same `\\ntest split\\n` separator as the
    upstream `MetaBreak/prompts/Q_TM-1_Llama.txt` format.
    """
    wrapped = [wrap_one(q, assistant_header) for q in questions]
    return "\ntest split\n".join(wrapped)


def derive_default_output(model_type: str) -> Path:
    """`Q_TM-1_Llama.txt` style filename inside repro_mb/prompts/."""
    suffix = model_type[:1].upper() + model_type[1:]  # 'llama' -> 'Llama'
    return PROMPTS_DIR / f"Q_TM-1_{suffix}.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_type", required=True,
                   help="Family slug. Known: " + ", ".join(KNOWN_HEADERS))
    p.add_argument("--model", default=None,
                   help="HF model path. Required if --model_type is unknown "
                        "(needed for tokenizer-based auto-detection) and also "
                        "used to sanity-check the assistant_header for known "
                        "families.")
    p.add_argument("--source", default=str(DEFAULT_SOURCE),
                   help="Raw Q.txt (450 prompts, split by '\\ntest split\\n').")
    p.add_argument("--output", default=None,
                   help="Output path. Default: prompts/Q_TM-1_<Model>.txt.")
    p.add_argument("--force", action="store_true",
                   help="Regenerate even if the output already exists.")
    p.add_argument("--no_tokenizer_check", action="store_true",
                   help="For known model_types, skip loading the HF "
                        "tokenizer. Useful if --model is not available; "
                        "the assistant_header is then taken verbatim from "
                        "KNOWN_HEADERS without verifying it tokenizes well.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    out_path = Path(args.output) if args.output else derive_default_output(
        args.model_type
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force:
        print(f"[build_prompts] {out_path} already exists, skipping. "
              f"Pass --force to regenerate.")
        return

    if not source.exists():
        raise FileNotFoundError(
            f"[build_prompts] source prompts file not found: {source}"
        )

    # ----- pick the assistant_header for this model_type ------------------ #
    if args.model_type in KNOWN_HEADERS and args.no_tokenizer_check:
        assistant_header = KNOWN_HEADERS[args.model_type]["assistant_header"]
        print(f"[build_prompts] using KNOWN_HEADERS[{args.model_type!r}] "
              f"(no_tokenizer_check)")
    else:
        if not args.model:
            raise ValueError(
                "[build_prompts] --model is required unless "
                "--no_tokenizer_check is set with a known --model_type."
            )
        # Import only when needed so the helpers above (wrap_one, build_corpus)
        # remain usable without `transformers` installed.
        from transformers import AutoTokenizer
        print(f"[build_prompts] loading tokenizer from {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if args.model_type in KNOWN_HEADERS:
            cfg = resolve_config(args.model_type, tokenizer)
        else:
            cfg = auto_detect_config(args.model_type, tokenizer)
            print(f"[build_prompts] auto-detected assistant_header for "
                  f"unknown model_type={args.model_type!r}: "
                  f"{cfg.assistant_header!r}")
        assistant_header = cfg.assistant_header

    # ----- wrap and write ------------------------------------------------- #
    questions = load_questions(source)
    print(f"[build_prompts] loaded {len(questions)} questions from {source}")
    print(f"[build_prompts] assistant_header = {assistant_header!r}")

    corpus = build_corpus(questions, assistant_header)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"[build_prompts] wrote {len(questions)} wrapped prompts to "
          f"{out_path}")


if __name__ == "__main__":
    main()
