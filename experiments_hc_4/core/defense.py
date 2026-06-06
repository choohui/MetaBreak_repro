from __future__ import annotations

import numpy as np

from .metrics import binary_eval, per_letter_rates, prompt_eval
from .model import LoadedModel, refusal_success
from .template import find_literal_assistant_spans, find_regular_assistant_spans
from .thresholds import labels_from_rows, predict_rule


def flagged_base_positions(rows: list[dict], pred: np.ndarray) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for row, p in zip(rows, pred):
        if not p:
            continue
        sid = int(row["sample_index"])
        out.setdefault(sid, set()).add(int(row["base_position"]))
    return out


def _expand_header_like_positions(input_ids: list[int], positions: set[int], tpl) -> set[int]:
    """If a flagged token sits in an assistant-header-like span, remove every
    replace slot in that span. This keeps the action token-focused while avoiding
    the ineffective case where only one of the three control-like slots is cut."""
    out = set(int(p) for p in positions)
    literal = find_literal_assistant_spans(input_ids, tpl)
    regular = find_regular_assistant_spans(input_ids, tpl)
    last_literal_start = max((s.start for s in literal), default=-1)
    spans = regular + [s for s in literal if s.start != last_literal_start]
    if out:
        for span in spans:
            out.update({span.start + int(off) for off in tpl.replace_positions})
    for span in spans:
        span_replace = {span.start + int(off) for off in tpl.replace_positions}
        if out & span_replace:
            out.update(span_replace)
    return out


def clean_ids(input_ids: list[int], positions: set[int], action: str, tokenizer, tpl=None) -> list[int]:
    if action == "prompt_block":
        return list(input_ids)
    base_positions = _expand_header_like_positions(input_ids, positions, tpl) if tpl is not None else set(positions)
    rm = set()
    if action == "drop_token_pm1":
        for p in base_positions:
            rm.update({p - 1, p, p + 1})
    else:
        rm = set(base_positions)
    if action.startswith("drop_token"):
        return [int(t) for i, t in enumerate(input_ids) if i not in rm]
    if action == "mask_token":
        mask = getattr(tokenizer, "unk_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0
        return [int(mask) if i in rm else int(t) for i, t in enumerate(input_ids)]
    return list(input_ids)


def evaluate_rule_on_split(x: np.ndarray, names: list[str], rows: list[dict], rule: dict, split: str = "test") -> dict:
    y = labels_from_rows(rows)
    mask = np.asarray([r["split"] == split for r in rows]) & (y >= 0)
    pred = predict_rule(x[mask], names, rule)
    rr = [r for r, m in zip(rows, mask) if m]
    return {
        "token": binary_eval(pred, y[mask]),
        "per_letter": per_letter_rates(pred, [r["letter"] for r in rr]),
        "prompt": prompt_eval(rr, pred),
    }


def generate_defended(lm: LoadedModel, input_rows: list[dict], flags: dict[int, set[int]],
                      action: str, max_new_tokens: int) -> list[dict]:
    out = []
    for rec in input_rows:
        sid = int(rec["sample_index"])
        is_blocked = action == "prompt_block" and bool(flags.get(sid))
        if is_blocked:
            text = ""
            success = False
        else:
            ids = clean_ids([int(x) for x in rec["input_ids"]], flags.get(sid, set()),
                            action, lm.tokenizer, lm.template)
            text = lm.generate_from_ids(ids, max_new_tokens=max_new_tokens)
            success = refusal_success(text)
        out.append({
            "sample_index": sid,
            "variant": rec.get("variant"),
            "letter": rec.get("letter"),
            "action": action,
            "blocked": bool(is_blocked),
            "n_flagged_positions": len(flags.get(sid, set())),
            "success": bool(success),
            "response": text,
        })
    return out
