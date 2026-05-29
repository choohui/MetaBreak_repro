"""Llama-3.1 embedding search — legacy Llama-only entry point.

Automatically injects ``--model_type llama`` and delegates to ``src.embedding``.
The full interface (including --model_type, multi-family support) is available via:

    python -m src.embedding --model_type <family> --model /path/...

This wrapper exists so existing Llama-specific invocations continue to work
without modification:

    python embedding.py --model /path/to/Llama-3.1-8B-Instruct [--topk 200]
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package root is importable when running as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Inject --model_type llama for backward compatibility if not already set.
if "--model_type" not in sys.argv:
    sys.argv[1:1] = ["--model_type", "llama"]

from src.embedding import main  # noqa: E402

if __name__ == "__main__":
    main()
