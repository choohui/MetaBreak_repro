"""Llama-3.1 LLM attack — legacy Llama-only entry point.

Automatically injects ``--model_type llama`` and delegates to ``src.attack``.
The full interface (including --model_type, multi-family support) is available via:

    python -m src.attack --model_type <family> --model /path/... [options]

This wrapper exists so existing Llama-specific invocations continue to work
without modification:

    python attack.py --model /path/to/Llama-3.1-8B-Instruct [--also_baseline]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if "--model_type" not in sys.argv:
    sys.argv[1:1] = ["--model_type", "llama"]

from src.attack import main  # noqa: E402

if __name__ == "__main__":
    main()
