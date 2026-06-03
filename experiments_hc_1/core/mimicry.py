"""Dependency-free prompt helpers (mirror ``src.mimicry``).

Re-implemented locally so the prompt-building / extraction stages stay importable
without the heavy ``transformers`` dependency that ``src.mimicry`` pulls in at
module import time. The logic is identical to ``src.mimicry.apply_mimicry`` /
``load_prompts``.
"""

from __future__ import annotations

from pathlib import Path


def load_prompts(path: str | Path) -> list[str]:
    """Split a MetaBreak prompt file on the ``\\ntest split\\n`` separator."""
    raw = Path(path).read_text(encoding="utf-8")
    parts = raw.split("\ntest split\n")
    return [p for p in parts if p.strip()]


def apply_mimicry(prompt: str, special_strs: list[str], replacement_strs: list[str]) -> str:
    """Replace each special-token string with its replacement, longest-first
    (so a shorter special string can't eat a longer one)."""
    assert len(special_strs) == len(replacement_strs)
    pairs = sorted(zip(special_strs, replacement_strs),
                   key=lambda x: len(x[0]), reverse=True)
    out = prompt
    for s, r in pairs:
        out = out.replace(s, r)
    return out
