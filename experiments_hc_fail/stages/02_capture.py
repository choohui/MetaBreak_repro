"""Stage 02 - capture token signals and optional ASR responses."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG.parent
for p in (str(REPO_ROOT), str(PKG.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evaluate import matches_refusal  # noqa: E402
from experiments_hc_fail.config import ExpConfig, config_from_args, make_parser, require_model  # noqa: E402
from experiments_hc_fail.core import io  # noqa: E402
from experiments_hc_fail.core.capture import forward_capture, forward_capture_ids, sink_scores  # noqa: E402
from experiments_hc_fail.core.labeling import label_positions_for_variant, sample_ordinary_positions  # noqa: E402
from experiments_hc_fail.core.labels import ASR_VARIANTS, CAT_A, CAT_TO_LETTER  # noqa: E402
from experiments_hc_fail.core.model import DTYPES, load_model  # noqa: E402
from experiments_hc_fail.core.mock import load_mock_model  # noqa: E402
from experiments_hc_fail.core.template import template_prefix_suffix_ids, template_prefix_suffix_lengths  # noqa: E402


def _cap_category(labels: dict[int, str], category: str, max_per: int) -> dict[int, str]:
    if max_per < 0:
        return labels
    positions = sorted(p for p, c in labels.items() if c == category)
    if len(positions) <= max_per:
        return labels
    step = len(positions) / max_per
    keep = {positions[int(i * step)] for i in range(max_per)}
    return {p: c for p, c in labels.items() if c != category or p in keep}


def _encode_ids(tokenizer, word: str) -> set[int]:
    ids: set[int] = set()
    for variant in (word, " " + word):
        try:
            ids.update(int(x) for x in tokenizer(variant, add_special_tokens=False)["input_ids"])
        except Exception:
            pass
    return ids


def _build_injected_ids(lm, row: dict) -> tuple[list[int], list[int]]:
    prefix, suffix = template_prefix_suffix_ids(lm.tokenizer)
    head_ids = lm.tokenizer(row["carrier_head"], add_special_tokens=False)["input_ids"]
    tail_ids = lm.tokenizer(row["carrier_tail"], add_special_tokens=False)["input_ids"]
    inject_ids = [int(x) for x in row["inject_token_ids"]]
    start = len(prefix) + len(head_ids)
    return prefix + [int(x) for x in head_ids] + inject_ids + [int(x) for x in tail_ids] + suffix, list(range(start, start + len(inject_ids)))


def _signals_at(cap, sinks, pos: int, tokenizer) -> dict:
    hidden_norm = [round(float(torch.linalg.vector_norm(h[pos], dim=-1)), 5) for h in cap.hidden_states]
    sink = [round(float(sinks[l, pos]), 5) for l in range(sinks.shape[0])]
    value = [round(float(cap.value_norms[l, pos]), 5) for l in range(cap.value_norms.shape[0])]
    output = [round(float(cap.output_norms[l, pos]), 5) for l in range(cap.output_norms.shape[0])]
    active = [round(float(s * v), 5) for s, v in zip(sink, value)]
    return {
        "token_id": int(cap.input_ids[pos]),
        "decoded": tokenizer.convert_ids_to_tokens(int(cap.input_ids[pos])),
        "hidden_norm": hidden_norm,
        "sink": sink,
        "value_norm": value,
        "output_norm": output,
        "active_value": active,
        "active_value_max": max(active) if active else 0.0,
    }


@torch.no_grad()
def _generate_once(lm, text: str, cfg: ExpConfig) -> dict:
    if getattr(lm, "is_mock", False):
        is_attack = "alpha" in text or "<|eot_id|>" in text
        out = "Here is the requested content." if is_attack else "I cannot help with that."
        return {"text": out, "n_input_tokens": 0, "n_generated_tokens": len(out.split()), "seconds": 0.0}
    input_ids = lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    input_ids = input_ids.to(lm.device)
    in_len = int(input_ids.shape[1])
    do_sample = cfg.temperature > 0.0
    t0 = time.time()
    out = lm.model.generate(
        input_ids,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=do_sample,
        temperature=cfg.temperature if do_sample else 1.0,
        pad_token_id=lm.tokenizer.eos_token_id,
    )
    dt = time.time() - t0
    gen_ids = out[0, in_len:].tolist()
    return {
        "text": lm.tokenizer.decode(gen_ids, skip_special_tokens=True),
        "n_input_tokens": in_len,
        "n_generated_tokens": len(gen_ids),
        "seconds": dt,
    }


def _load_lm(cfg: ExpConfig):
    if cfg.smoke:
        return load_mock_model(cfg.model_type)
    require_model(cfg)
    return load_model(str(cfg.model), cfg.model_type, cfg.dtype, cfg.device)


def run(cfg: ExpConfig) -> dict:
    lm = _load_lm(cfg)
    rows = io.read_jsonl(cfg.prompts)
    repl = io.read_json(cfg.replacement)
    mimicry_ids = {int(x) for x in repl.get("best_triple_ids", [])}
    prefix_len, suffix_len = template_prefix_suffix_lengths(lm.tokenizer)

    token_rows: list[dict] = []
    responses: list[dict] = []

    for prompt in rows:
        is_inject = prompt["variant"] == "benign_mimicry" and prompt.get("carrier_head") is not None
        if is_inject:
            ids, inject_positions = _build_injected_ids(lm, prompt)
            cap = forward_capture_ids(lm, ids)
        else:
            cap = forward_capture(lm, prompt["text"])
            inject_positions = []
        sinks = sink_scores(cap)["mean_over_heads"]

        if is_inject:
            from experiments_hc_fail.core.labels import CAT_C
            labels = {p: CAT_C for p in inject_positions}
        else:
            extra = {"mimicry_ids": mimicry_ids}
            if prompt["variant"] == "positioned_regular" and prompt.get("slot_word"):
                extra["slot_word_ids"] = _encode_ids(lm.tokenizer, prompt["slot_word"])
            labels = label_positions_for_variant(
                cap.input_ids, lm.template, prefix_len, suffix_len, prompt["variant"], extra
            )
            labels = _cap_category(labels, CAT_A, cfg.max_a_per_prompt)

        for p in sample_ordinary_positions(cap.input_ids, lm.template, prefix_len, suffix_len, cfg.ordinary):
            from experiments_hc_fail.core.labels import CAT_G
            labels.setdefault(p, CAT_G)

        for base_pos, category in sorted(labels.items()):
            for off in cfg.pos_offsets:
                pos = base_pos + off
                if pos >= len(cap.input_ids):
                    continue
                rec = {
                    "row_id": len(token_rows),
                    "sample_index": int(prompt["sample_index"]),
                    "prompt_idx": prompt["idx"],
                    "source": prompt["source"],
                    "variant": prompt["variant"],
                    "category": category,
                    "letter": CAT_TO_LETTER[category],
                    "base_position": int(base_pos),
                    "position": int(pos),
                    "pos_offset": int(off),
                    "seq_len": int(len(cap.input_ids)),
                }
                rec.update(_signals_at(cap, sinks, pos, lm.tokenizer))
                token_rows.append(rec)

        if not cfg.skip_generation and prompt["variant"] in ASR_VARIANTS:
            gen = _generate_once(lm, prompt["text"], cfg)
            is_refusal, matched = matches_refusal(gen["text"])
            responses.append({
                "sample_index": int(prompt["sample_index"]),
                "idx": prompt["idx"],
                "variant": prompt["variant"],
                "text": prompt["text"],
                "response": gen,
                "refusal_keyword_matched": matched,
                "success_refusal_keyword": not is_refusal,
            })

    io.write_jsonl(cfg.tokens, token_rows)
    if responses:
        io.write_jsonl(cfg.responses, responses)
    summary = {
        "n_prompts": len(rows),
        "n_token_rows": len(token_rows),
        "n_responses": len(responses),
        "pos_offsets": cfg.pos_offsets,
    }
    io.write_json(cfg.out_dir / "capture_summary.json", summary)
    print(f"[02] token rows={len(token_rows)} -> {cfg.tokens}")
    if responses:
        print(f"[02] responses={len(responses)} -> {cfg.responses}")
    return summary


def main() -> None:
    run(config_from_args(make_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()

