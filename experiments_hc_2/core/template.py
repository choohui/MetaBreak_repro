"""Chat-template metadata + assistant-header span detection.

Reuses ``src.model_configs.resolve_config`` for the chat boundary; everything
else is local (no dependency on other ``experiments_*`` folders).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow ``import src.*`` when run as a script (repo root = .../repro_mb).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_configs import ModelCfg, resolve_config  # noqa: E402

_SENTINEL = "@@HC1_CONTENT_SENTINEL@@"


@dataclass
class TemplateInfo:
    model_type: str
    assistant_header: str
    header_ids: list[int]          # tokenized assistant_header
    replace_positions: list[int]   # offsets inside header_ids that are special tokens
    fixed_positions: list[int]     # offsets that are literal regular tokens
    fixed_ids_by_pos: dict[int, int]
    target_token_ids: list[int]    # special-token ids, aligned with replace_positions
    target_token_strs: list[str]
    special_token_ids: set[int]


def build_template_info(tokenizer: Any, model_type: str) -> TemplateInfo:
    cfg: ModelCfg = resolve_config(model_type, tokenizer)
    header_ids = tokenizer(cfg.assistant_header, add_special_tokens=False)["input_ids"]
    if len(header_ids) != cfg.expected_n_tokens:
        raise RuntimeError(
            "assistant_header tokenization length disagrees with model_configs "
            f"({len(header_ids)} vs {cfg.expected_n_tokens})."
        )
    fixed_ids_by_pos = {int(p): int(header_ids[p]) for p in cfg.fixed_positions}
    return TemplateInfo(
        model_type=cfg.model_type,
        assistant_header=cfg.assistant_header,
        header_ids=[int(x) for x in header_ids],
        replace_positions=[int(x) for x in cfg.replace_positions],
        fixed_positions=[int(x) for x in cfg.fixed_positions],
        fixed_ids_by_pos=fixed_ids_by_pos,
        target_token_ids=[int(x) for x in cfg.target_token_ids],
        target_token_strs=list(cfg.target_token_strs),
        special_token_ids={int(x) for x in cfg.special_token_ids},
    )


def template_prefix_suffix_lengths(tokenizer: Any) -> tuple[int, int]:
    """Token lengths of the fixed template prefix (BOS + user header) and suffix
    (final generation-prompt assistant header) that wrap the user content."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": _SENTINEL}],
        add_generation_prompt=True,
        tokenize=False,
    )
    if _SENTINEL not in rendered:
        raise RuntimeError("Sentinel lost while rendering chat template.")
    prefix_str, suffix_str = rendered.split(_SENTINEL, 1)
    prefix_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix_str, add_special_tokens=False)["input_ids"]
    return len(prefix_ids), len(suffix_ids)


def template_prefix_suffix_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
    """The actual token ids of the fixed template prefix/suffix that wrap the user
    content (BOS + user header, and the trailing generation-prompt assistant
    header). Used to build category-C ``input_ids`` whose template wrapper matches
    every other prompt exactly. Consistent with :func:`template_prefix_suffix_lengths`."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": _SENTINEL}],
        add_generation_prompt=True,
        tokenize=False,
    )
    if _SENTINEL not in rendered:
        raise RuntimeError("Sentinel lost while rendering chat template.")
    prefix_str, suffix_str = rendered.split(_SENTINEL, 1)
    prefix_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix_str, add_special_tokens=False)["input_ids"]
    return [int(x) for x in prefix_ids], [int(x) for x in suffix_ids]


# --------------------------------------------------------------------------- #
# Span detection
# --------------------------------------------------------------------------- #


@dataclass
class Span:
    start: int
    ids: list[int]
    kind: str


def find_literal_assistant_spans(ids: list[int], tpl: TemplateInfo) -> list[Span]:
    """Exact occurrences of the literal assistant header (the D/malicious case)."""
    w = len(tpl.header_ids)
    spans = []
    for pos in range(0, max(0, len(ids) - w + 1)):
        if ids[pos : pos + w] == tpl.header_ids:
            spans.append(Span(pos, ids[pos : pos + w], "literal_assistant_header"))
    return spans


def find_regular_assistant_spans(ids: list[int], tpl: TemplateInfo) -> list[Span]:
    """Occurrences of a *mimicked* assistant header: fixed positions match the
    literal ``assistant``/``\\n\\n`` ids, but the replace positions are regular
    (non-special) tokens. Used for both B (malicious mimicry) and any other
    variant whose header preserves the 5-token shape with regular slots.
    """
    w = len(tpl.header_ids)
    spans = []
    for pos in range(0, max(0, len(ids) - w + 1)):
        window = ids[pos : pos + w]
        if any(window[fp] != fid for fp, fid in tpl.fixed_ids_by_pos.items()):
            continue
        if any(int(window[rp]) in tpl.special_token_ids for rp in tpl.replace_positions):
            continue
        spans.append(Span(pos, window, "regular_assistant_header"))
    return spans


def build_mimicked_header(assistant_header: str, special_strs: list[str],
                          repl_strs: list[str]) -> str:
    """Replace special-token strings in the assistant header with replacements
    (longest-first, mirroring ``src.mimicry.apply_mimicry``)."""
    pairs = sorted(zip(special_strs, repl_strs), key=lambda x: len(x[0]), reverse=True)
    out = assistant_header
    for s, r in pairs:
        out = out.replace(s, r)
    return out
