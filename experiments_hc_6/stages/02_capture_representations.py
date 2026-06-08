from __future__ import annotations

import numpy as np
from tqdm import tqdm

from experiments_hc_6.config import ExpConfig
from experiments_hc_6.core import io
from experiments_hc_6.core.capture import chat_input_ids, forward_capture_ids, forward_capture_text, row_signals
from experiments_hc_6.core.labels import CAT_TO_LETTER, LETTER_TO_CAT, VARIANT_TO_LETTER
from experiments_hc_6.core.model import get_model, refusal_success
from experiments_hc_6.core.template import (
    content_bounds,
    find_literal_assistant_spans,
    find_regular_assistant_spans,
    template_prefix_suffix_ids,
)


def _encode_ids(tokenizer, text: str) -> set[int]:
    out = set()
    for s in (text, " " + text):
        try:
            out.update(int(x) for x in tokenizer(s, add_special_tokens=False)["input_ids"])
        except Exception:
            pass
    return out


def _sample_even(xs: list[int], k: int) -> list[int]:
    if k < 0 or len(xs) <= k:
        return list(xs)
    step = len(xs) / k
    return [xs[int(i * step)] for i in range(k)]


def _c_input_ids(lm, head: str, tail: str, inject_ids: list[int]) -> tuple[list[int], list[int]]:
    prefix, suffix = template_prefix_suffix_ids(lm.tokenizer)
    head_ids = [int(x) for x in lm.tokenizer(head, add_special_tokens=False)["input_ids"]]
    tail_ids = [int(x) for x in lm.tokenizer(tail, add_special_tokens=False)["input_ids"]]
    start = len(prefix) + len(head_ids)
    inj = [int(x) for x in inject_ids]
    return prefix + head_ids + inj + tail_ids + suffix, list(range(start, start + len(inj)))


def _mimic_input_ids(lm, text: str, repl_ids: list[int]) -> list[int]:
    """Build a malicious-mimicry (B) prompt at the token-ID level: tokenize the
    literal-header attack text, then substitute the special-token ids at the
    injected assistant header's replace positions with the mimicry ids. This
    keeps the 5-token header shape (``assistant``@2, ``\\n\\n``@4) intact, which a
    plain string-replace + re-tokenize destroys -- the replacement strings split
    into several tokens and shift the fixed slots -- so find_regular_assistant_spans
    can still locate the mimicked header. Only headers inside the content bounds
    (the injected attack header, not the template's own trailing header) are touched.
    """
    tpl = lm.template
    ids = chat_input_ids(lm, text)
    lo, hi = content_bounds(lm.tokenizer, ids)
    for span in find_literal_assistant_spans(ids, tpl):
        if not (lo <= span.start and span.start + len(span.ids) <= hi):
            continue
        for j, off in enumerate(tpl.replace_positions):
            if j < len(repl_ids):
                ids[span.start + off] = int(repl_ids[j])
    return ids


def _labels_for(lm, row: dict, input_ids: list[int], c_positions: list[int] | None, cfg: ExpConfig) -> dict[int, str]:
    tpl = lm.template
    lo, hi = content_bounds(lm.tokenizer, input_ids)
    labels: dict[int, str] = {}
    if c_positions:
        for p in c_positions:
            labels[p] = LETTER_TO_CAT["C"]
    else:
        variant = row["variant"]
        if variant == "malicious_special":
            for span in find_literal_assistant_spans(input_ids, tpl):
                if lo <= span.start and span.start + len(span.ids) <= hi:
                    for off in tpl.replace_positions:
                        labels[span.start + off] = LETTER_TO_CAT["D"]
        elif variant == "malicious_mimicry":
            for span in find_regular_assistant_spans(input_ids, tpl):
                if lo <= span.start and span.start + len(span.ids) <= hi:
                    for off in tpl.replace_positions:
                        labels[span.start + off] = LETTER_TO_CAT["B"]
        elif variant == "positioned_regular":
            slot_ids = _encode_ids(lm.tokenizer, str(row.get("slot_word", "")))
            for p in range(lo, hi):
                if int(input_ids[p]) in slot_ids:
                    labels[p] = LETTER_TO_CAT["F"]
        elif variant == "benign_special":
            attack_positions = set()
            for span in find_literal_assistant_spans(input_ids, tpl):
                if lo <= span.start and span.start + len(span.ids) <= hi:
                    for off in tpl.replace_positions:
                        attack_positions.add(span.start + off)
            for p in range(lo, hi):
                if int(input_ids[p]) in tpl.special_token_ids and p not in attack_positions:
                    labels[p] = LETTER_TO_CAT["E"]
    # Reference A: fixed template special tokens, capped per prompt.
    a_pos = [p for p, tid in enumerate(input_ids)
             if (p < lo or p >= hi) and int(tid) in tpl.special_token_ids and p not in labels]
    for p in _sample_even(a_pos, cfg.max_a_per_prompt):
        labels[p] = LETTER_TO_CAT["A"]
    # Ordinary G: sampled normal user-content tokens.
    if row["variant"] == "ordinary":
        cand = [p for p in range(lo, hi) if int(input_ids[p]) not in tpl.special_token_ids]
        for p in _sample_even(cand, cfg.ordinary_per_prompt):
            labels[p] = LETTER_TO_CAT["G"]
    return labels


def _target_slot_index(lm, input_ids: list[int], base_pos: int, c_positions: list[int] | None) -> int | None:
    if c_positions and base_pos in c_positions:
        return int(c_positions.index(base_pos))
    tpl = lm.template
    lo, hi = content_bounds(lm.tokenizer, input_ids)
    spans = find_literal_assistant_spans(input_ids, tpl) + find_regular_assistant_spans(input_ids, tpl)
    for span in spans:
        if not (lo <= span.start and span.start + len(span.ids) <= hi):
            continue
        rel = int(base_pos) - int(span.start)
        if rel in tpl.replace_positions:
            return int(list(tpl.replace_positions).index(rel))
    return None


def run(cfg: ExpConfig, lm=None) -> dict:
    lm = get_model(cfg, lm)
    prompts = io.read_jsonl(cfg.prompts_path)
    repl_ids = [int(x) for x in io.load_replacement(cfg.replacement_path).get("best_triple_ids", [])]
    token_rows: list[dict] = []
    input_rows: list[dict] = []
    responses: list[dict] = []
    hidden_rows: list[np.ndarray] = []

    for row in tqdm(prompts, desc="[02] capture"):
        c_positions = None
        if row["variant"] == "benign_mimicry" and row.get("carrier_head") is not None:
            input_ids, c_positions = _c_input_ids(
                lm, row["carrier_head"], row["carrier_tail"], row.get("inject_token_ids", []))
            cap = forward_capture_ids(lm, input_ids)
        elif row["variant"] == "malicious_mimicry":
            input_ids = _mimic_input_ids(lm, row["text"], row.get("inject_token_ids", repl_ids))
            cap = forward_capture_ids(lm, input_ids)
        else:
            cap = forward_capture_text(lm, row["text"])
            input_ids = cap.input_ids
        letter = VARIANT_TO_LETTER.get(row["variant"], "A")
        input_rows.append({
            "sample_index": int(row["sample_index"]),
            "idx": row.get("idx"),
            "variant": row["variant"],
            "letter": letter,
            "text": row["text"],
            "input_ids": input_ids,
        })
        labels = _labels_for(lm, row, input_ids, c_positions, cfg)
        for base_pos, category in sorted(labels.items()):
            for off in cfg.pos_offsets:
                pos = base_pos + off
                if pos >= len(input_ids):
                    continue
                rec = {
                    "row_id": len(token_rows),
                    "sample_index": int(row["sample_index"]),
                    "prompt_idx": row.get("idx"),
                    "variant": row["variant"],
                    "category": category,
                    "letter": CAT_TO_LETTER[category],
                    "is_attack_token": CAT_TO_LETTER[category] in ("B", "D"),
                    "base_position": int(base_pos),
                    "position": int(pos),
                    "pos_offset": int(off),
                    "target_token_index": _target_slot_index(lm, input_ids, base_pos, c_positions),
                    "token_id": int(input_ids[pos]),
                    "is_exact_special_token": int(input_ids[pos]) in set(int(x) for x in lm.template.special_token_ids),
                    "is_known_replacement_token": int(input_ids[pos]) in set(repl_ids),
                    "is_l2_neighbor_proxy": int(input_ids[pos]) in set(repl_ids),
                    "decoded": lm.tokenizer.convert_ids_to_tokens(int(input_ids[pos])),
                    "seq_len": len(input_ids),
                }
                rec.update(row_signals(cap, pos))
                token_rows.append(rec)
                hidden_rows.append(cap.hidden[pos])
        if not cfg.skip_generation and row["variant"] in {"malicious_mimicry", "malicious_special", "positioned_regular"}:
            text = lm.generate_from_ids(input_ids, max_new_tokens=cfg.max_new_tokens)
            responses.append({
                "sample_index": int(row["sample_index"]),
                "variant": row["variant"],
                "letter": letter,
                "success": refusal_success(text),
                "response": text,
            })

    hidden = np.stack(hidden_rows, axis=0).astype(np.float16) if hidden_rows else np.zeros((0, 0, 0), dtype=np.float16)
    io.write_jsonl(cfg.tokens_path, token_rows)
    io.write_jsonl(cfg.inputs_path, input_rows)
    io.write_jsonl(cfg.out_dir / "responses.jsonl", responses)
    np.savez_compressed(cfg.features_path, hidden=hidden)
    census = {}
    for r in token_rows:
        key = f"{r['letter']}_pos{r['pos_offset']}"
        census[key] = census.get(key, 0) + 1
    summary = {
        "n_tokens": len(token_rows),
        "n_inputs": len(input_rows),
        "hidden_shape": list(hidden.shape),
        "census": census,
        "responses": len(responses),
    }
    io.write_json(cfg.out_dir / "capture_summary.json", summary)
    print(f"[02] tokens={len(token_rows)} hidden={hidden.shape}")
    return summary

