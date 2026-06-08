from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_configs import resolve_config  # noqa: E402

SENTINEL = "@@HC4_CONTENT_SENTINEL@@"


@dataclass
class Span:
    start: int
    ids: list[int]
    kind: str


@dataclass
class TemplateInfo:
    model_type: str
    assistant_header: str
    header_ids: list[int]
    replace_positions: list[int]
    fixed_positions: list[int]
    fixed_ids_by_pos: dict[int, int]
    target_token_ids: list[int]
    target_token_strs: list[str]
    special_token_ids: set[int]


def build_template_info(tokenizer: Any, model_type: str) -> TemplateInfo:
    cfg = resolve_config(model_type, tokenizer)
    header_ids = [int(x) for x in tokenizer(cfg.assistant_header, add_special_tokens=False)["input_ids"]]
    fixed_ids_by_pos = {int(p): int(header_ids[p]) for p in cfg.fixed_positions}
    return TemplateInfo(
        model_type=cfg.model_type,
        assistant_header=cfg.assistant_header,
        header_ids=header_ids,
        replace_positions=[int(x) for x in cfg.replace_positions],
        fixed_positions=[int(x) for x in cfg.fixed_positions],
        fixed_ids_by_pos=fixed_ids_by_pos,
        target_token_ids=[int(x) for x in cfg.target_token_ids],
        target_token_strs=list(cfg.target_token_strs),
        special_token_ids={int(x) for x in cfg.special_token_ids},
    )


def template_prefix_suffix_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": SENTINEL}],
        add_generation_prompt=True,
        tokenize=False,
    )
    if SENTINEL not in rendered:
        raise RuntimeError("chat template sentinel was not preserved")
    prefix, suffix = rendered.split(SENTINEL, 1)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    return [int(x) for x in prefix_ids], [int(x) for x in suffix_ids]


def template_prefix_suffix_lengths(tokenizer: Any) -> tuple[int, int]:
    p, s = template_prefix_suffix_ids(tokenizer)
    return len(p), len(s)


def find_literal_assistant_spans(ids: list[int], tpl: TemplateInfo) -> list[Span]:
    w = len(tpl.header_ids)
    spans: list[Span] = []
    for pos in range(0, max(0, len(ids) - w + 1)):
        if ids[pos: pos + w] == tpl.header_ids:
            spans.append(Span(pos, ids[pos: pos + w], "literal_assistant_header"))
    return spans


def find_regular_assistant_spans(ids: list[int], tpl: TemplateInfo) -> list[Span]:
    w = len(tpl.header_ids)
    spans: list[Span] = []
    for pos in range(0, max(0, len(ids) - w + 1)):
        window = ids[pos: pos + w]
        if any(window[p] != tid for p, tid in tpl.fixed_ids_by_pos.items()):
            continue
        if any(int(window[p]) in tpl.special_token_ids for p in tpl.replace_positions):
            continue
        spans.append(Span(pos, window, "regular_assistant_header"))
    return spans


def content_bounds(tokenizer: Any, ids: list[int]) -> tuple[int, int]:
    prefix, suffix = template_prefix_suffix_ids(tokenizer)
    return len(prefix), len(ids) - len(suffix)

