"""Assign analyzed token positions to A-G categories."""

from __future__ import annotations

from typing import Any

from .labels import CAT_A, CAT_B, CAT_C, CAT_D, CAT_E, CAT_F
from .template import (
    TemplateInfo,
    find_literal_assistant_spans,
    find_regular_assistant_spans,
)


def label_positions_for_variant(
    ids: list[int],
    tpl: TemplateInfo,
    prefix_len: int,
    suffix_len: int,
    variant: str,
    extra: dict[str, Any] | None = None,
) -> dict[int, str]:
    extra = extra or {}
    content_lo = prefix_len
    content_hi = len(ids) - suffix_len
    labels: dict[int, str] = {}

    for p, tid in enumerate(ids):
        in_template = p < content_lo or p >= content_hi
        if in_template and int(tid) in tpl.special_token_ids:
            labels[p] = CAT_A

    if variant == "malicious_special":
        for span in find_literal_assistant_spans(ids, tpl):
            if content_lo <= span.start and span.start + len(span.ids) <= content_hi:
                for off in tpl.replace_positions:
                    labels[span.start + off] = CAT_D

    elif variant == "malicious_mimicry":
        for span in find_regular_assistant_spans(ids, tpl):
            if content_lo <= span.start and span.start + len(span.ids) <= content_hi:
                for off in tpl.replace_positions:
                    labels[span.start + off] = CAT_B

    elif variant == "positioned_regular":
        slot_ids = {int(x) for x in extra.get("slot_word_ids", [])}
        for p in range(content_lo, content_hi):
            if int(ids[p]) in slot_ids:
                labels[p] = CAT_F

    elif variant == "benign_mimicry":
        mimicry_ids = {int(x) for x in extra.get("mimicry_ids", [])}
        header_positions: set[int] = set()
        for span in find_regular_assistant_spans(ids, tpl):
            for off in tpl.replace_positions:
                header_positions.add(span.start + off)
        for p in range(content_lo, content_hi):
            if int(ids[p]) in mimicry_ids and p not in header_positions:
                labels[p] = CAT_C

    elif variant == "benign_special":
        attack_positions: set[int] = set()
        for span in find_literal_assistant_spans(ids, tpl):
            if content_lo <= span.start and span.start + len(span.ids) <= content_hi:
                for off in tpl.replace_positions:
                    attack_positions.add(span.start + off)
        for p in range(content_lo, content_hi):
            if int(ids[p]) in tpl.special_token_ids and p not in attack_positions:
                labels[p] = CAT_E

    return labels


def sample_ordinary_positions(
    ids: list[int],
    tpl: TemplateInfo,
    prefix_len: int,
    suffix_len: int,
    max_positions: int | None = 6,
) -> list[int]:
    content_lo, content_hi = prefix_len, len(ids) - suffix_len
    cand = [
        p for p in range(content_lo, content_hi)
        if int(ids[p]) not in tpl.special_token_ids
    ]
    if not cand:
        return []
    if max_positions is None or max_positions < 0 or len(cand) <= max_positions:
        return cand
    step = len(cand) / max_positions
    return [cand[int(i * step)] for i in range(max_positions)]

