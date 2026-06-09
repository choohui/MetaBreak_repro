"""Build + capture category C (benign mimicry) by token-level injection.

C is the token-identity control: the *exact* attack replacement token id placed
in a *benign* context. The decoded string round-trips to a different id in
natural text (the root cause of the C=0 census), so we splice the id directly
into the carrier's ``input_ids`` and bypass re-tokenization. The injected
positions are known exactly, so labeling is position-exact (no id matching).

hc_2: C is captured inline in the main extraction (stage 03) — no retroactive
append step (hc_1's ``03c`` is gone), so C appears in the census normally.
"""

from __future__ import annotations

from pathlib import Path

from . import benign_gen, io
# Heavy deps (torch via template/src.model_configs) are imported lazily inside
# c_input_ids so prompt-building (build_c_prompts, used by the model-free stage
# 01) stays torch-free.


def _split_on_substring(text: str, sub: str):
    idx = text.find(sub)
    if idx < 0:
        return None
    return text[:idx], text[idx + len(sub):]


def build_c_prompts(cfg, repl: dict) -> list[dict]:
    """C rows to ``cfg.n_benign`` total: curated seeds (split on the mimicry
    string) first, then template fills. Each row carries ``carrier_head`` /
    ``carrier_tail`` and the ``inject_token_ids`` to splice between them."""
    inject_ids = [int(x) for x in repl.get("best_triple_ids", [])]
    repl_strs = repl.get("best_triple_decoded", [])
    mim_str = next((s for s in repl_strs if s and s.strip() and s != "�"), "ujících")

    target = cfg.n_benign if cfg.n_benign is not None else cfg.n
    rows: list[dict] = []

    if Path(cfg.benign_mimicry).exists():
        for r in io.read_jsonl(cfg.benign_mimicry):
            split = _split_on_substring(r.get("text", ""), mim_str)
            if split is None:
                continue  # seed without the mimicry string -> can't place the id
            head, tail = split
            rows.append({
                "id": r.get("id"),
                "source": "benign_mimicry_prompts",
                "carrier_head": head,
                "carrier_tail": tail,
                "inject_token_ids": inject_ids,
                "text": r.get("text", head + mim_str + tail),
            })

    need = max(0, target - len(rows))
    for g in benign_gen.generate_benign_mimicry(need, repl_strs):
        rows.append({
            "id": g["id"],
            "source": "benign_mimicry_generated",
            "carrier_head": g["carrier_head"],
            "carrier_tail": g["carrier_tail"],
            "inject_token_ids": inject_ids,
            "text": g["text"],
        })
    return rows[:target]


def c_input_ids(lm, head: str, tail: str, inject_ids) -> tuple[list[int], list[int]]:
    """``prefix + tok(head) + inject_ids + tok(tail) + suffix`` with the same
    template wrapper as every other prompt. Returns (input_ids, inject_positions)."""
    from .template import template_prefix_suffix_ids
    prefix_ids, suffix_ids = template_prefix_suffix_ids(lm.tokenizer)
    head_ids = [int(x) for x in lm.tokenizer(head, add_special_tokens=False)["input_ids"]]
    tail_ids = [int(x) for x in lm.tokenizer(tail, add_special_tokens=False)["input_ids"]]
    inject_ids = [int(x) for x in inject_ids]
    start = len(prefix_ids) + len(head_ids)
    inject_positions = list(range(start, start + len(inject_ids)))
    full = prefix_ids + head_ids + inject_ids + tail_ids + suffix_ids
    return full, inject_positions
