"""Build + capture category C (benign mimicry) by token-level injection.

C is the token-identity control: the *exact* attack replacement token id placed
in a *benign* context. The decoded string round-trips to a different id in
natural text (the root cause of the C=0 census), so we splice the id directly
into the carrier's ``input_ids`` and bypass re-tokenization. The injected
positions are known exactly, so labeling is position-exact (no id matching).

Shared by the standalone append script (``03c_append_benign_mimicry.py``) and the
full-run path (stage 03), so both produce identical C rows.
"""

from __future__ import annotations

from pathlib import Path

from . import benign_gen, io
from .labels import CAT_C, CAT_TO_LETTER
# Heavy deps (torch via capture/features, src.model_configs via template) are
# imported lazily inside the capture helpers so prompt-building (build_c_prompts,
# used by the model-free stage 01) stays torch-free.


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


def capture_c_rows(lm, c_rows: list[dict], cfg, start_row_id: int,
                   start_sample_index: int):
    """Forward-pass each C prompt; emit token rows (stage-03 shape) + hidden cubes.

    ``row_id`` is contiguous from ``start_row_id``; ``sample_index`` is offset
    past existing prompts so the per-prompt sink grouping (stage 06) stays correct.
    """
    from .capture import forward_capture_ids, sink_scores
    from .features import CaptureSignals, hidden_vector

    token_rows: list[dict] = []
    hidden_cubes = []
    rid = start_row_id
    for sidx, row in enumerate(c_rows):
        input_ids, inject_positions = c_input_ids(
            lm, row["carrier_head"], row["carrier_tail"], row["inject_token_ids"])
        cap = forward_capture_ids(lm, input_ids)
        sinks = sink_scores(cap)["mean_over_heads"]
        sig = CaptureSignals(cap, sinks)
        seq = len(cap.input_ids)
        for p in inject_positions:
            for off in cfg.pos_offsets:
                pos = p + off
                if pos >= seq:
                    continue
                rec = {
                    "row_id": rid,
                    "sample_index": start_sample_index + sidx,
                    "prompt_idx": row.get("id"),
                    "variant": "benign_mimicry",
                    "category": CAT_C,
                    "letter": CAT_TO_LETTER[CAT_C],
                    "base_position": int(p),
                    "position": int(pos),
                    "pos_offset": int(off),
                    "token_id": int(cap.input_ids[pos]),
                    "decoded": lm.tokenizer.convert_ids_to_tokens(int(cap.input_ids[pos])),
                    "seq_len": int(seq),
                }
                rec.update(sig.signals_at(pos))
                token_rows.append(rec)
                hidden_cubes.append(hidden_vector(cap, pos))
                rid += 1
    return token_rows, hidden_cubes
