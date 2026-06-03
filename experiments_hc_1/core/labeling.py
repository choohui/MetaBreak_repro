"""Assign each token position one of the 7 categories A..G (Main.md §2.1).

Region rule (content-agnostic): the chat-template prefix/suffix wrap the user
content. Specials inside the template region are **A (system special,
reference)**. Inside the content region, the assignment depends on the prompt's
``variant`` (written by stage 01):

    malicious_special  -> D at literal-assistant-header replace slots
    malicious_mimicry  -> B at mimicked-assistant-header replace slots
    positioned_regular -> F at the injected benign ``slot_word`` positions
    benign_mimicry     -> C at mimicry-replacement token ids (not in a header)
    benign_special     -> E at special-token ids (not part of a literal header)
    ordinary           -> (only A here; G positions are sampled by the caller)

``extra`` carries variant-specific lookups:
    mimicry_ids   : set[int]  - replacement token ids (e.g. {115614, 182})  [C]
    slot_word_ids : set[int]  - token ids of the F slot word                [F]
"""

from __future__ import annotations

from typing import Any

from .labels import CAT_A, CAT_B, CAT_C, CAT_D, CAT_E, CAT_F, CAT_G
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
    n = len(ids)
    content_lo = prefix_len
    content_hi = n - suffix_len
    labels: dict[int, str] = {}

    # A: every special-token id that lives in the fixed template prefix/suffix.
    for p, tid in enumerate(ids):
        in_template = p < content_lo or p >= content_hi
        if in_template and int(tid) in tpl.special_token_ids:
            labels[p] = CAT_A

    def in_content(p: int) -> bool:
        return content_lo <= p < content_hi

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
        mim_ids = {int(x) for x in extra.get("mimicry_ids", [])}
        # exclude any position that is part of a (mimicked) assistant-header span
        header_positions: set[int] = set()
        for span in find_regular_assistant_spans(ids, tpl):
            for off in tpl.replace_positions:
                header_positions.add(span.start + off)
        for p in range(content_lo, content_hi):
            if int(ids[p]) in mim_ids and p not in header_positions:
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

    elif variant == "ordinary":
        pass  # G positions added by the caller via sample_ordinary_positions

    return labels


def sample_ordinary_positions(
    ids: list[int],
    tpl: TemplateInfo,
    prefix_len: int,
    suffix_len: int,
    max_positions: int | None = 6,
) -> list[int]:
    """Non-special content positions to seed the G (ordinary) baseline.

    ``max_positions`` None or < 0 -> every ordinary content position.
    """
    content_lo, content_hi = prefix_len, len(ids) - suffix_len
    cand = [p for p in range(content_lo, content_hi)
            if int(ids[p]) not in tpl.special_token_ids]
    if not cand:
        return []
    if max_positions is None or max_positions < 0 or len(cand) <= max_positions:
        return cand
    step = len(cand) / max_positions
    return [cand[int(i * step)] for i in range(max_positions)]
