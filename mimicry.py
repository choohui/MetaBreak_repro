"""Llama-3.1 prompt mimicry — legacy Llama-only entry point.

Automatically injects ``--model_type llama`` and delegates to ``src.mimicry``.
The full interface (including --model_type, multi-family support) is available via:

    python -m src.mimicry --model_type <family> --model /path/... [options]

This wrapper exists so existing Llama-specific invocations continue to work
without modification:

    python mimicry.py --model /path/to/Llama-3.1-8B-Instruct [--n 10]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if "--model_type" not in sys.argv:
    sys.argv[1:1] = ["--model_type", "llama"]

from src.mimicry import main  # noqa: E402

if __name__ == "__main__":
    main()
