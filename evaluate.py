"""Llama-3.1 evaluation — legacy Llama-only entry point.

Automatically injects ``--model_type llama`` and delegates to ``src.evaluate``.
The full interface (including --model_type, multi-family support) is available via:

    python -m src.evaluate --model_type <family> [options]

This wrapper exists so existing Llama-specific invocations continue to work
without modification:

    python evaluate.py [--guard_model /path/to/Llama-Guard-3-8B]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if "--model_type" not in sys.argv:
    sys.argv[1:1] = ["--model_type", "llama"]

from src.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
