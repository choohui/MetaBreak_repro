"""Generation + judging helpers for the hc_7 steering stages.

``generate_steered`` runs one greedy (or sampled) generation with the steering
hook active, rebuilding the chat-template prompt the SAME way as
``src.attack.generate_once`` (so an attack-slot ``positions`` mask lines up with
the prompt token ids). Mock models cannot steer, so they fall back to
``mock_generate`` (steering has no observable effect — smoke checks plumbing only).
"""

from __future__ import annotations

import torch

from .steer import can_steer, steer


def build_prompt_ids(lm, text: str) -> torch.Tensor:
    ids = lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True, return_tensors="pt", return_dict=False)
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return ids.to(lm.device)


def _generate_real(lm, cfg, input_ids: torch.Tensor) -> str:
    out = lm.model.generate(
        input_ids,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.temperature > 0.0,
        temperature=cfg.temperature if cfg.temperature > 0.0 else 1.0,
        pad_token_id=lm.tokenizer.eos_token_id,
    )
    gen_ids = out[0, input_ids.shape[1]:].tolist()
    return lm.tokenizer.decode(gen_ids, skip_special_tokens=True)


def make_mask(seq_len: int, positions) -> torch.Tensor | None:
    """Bool mask [seq_len] for attack-slot positions; None if positions is None."""
    if positions is None:
        return None
    m = torch.zeros(seq_len, dtype=torch.bool)
    valid = [int(p) for p in positions if 0 <= int(p) < seq_len]
    if valid:
        m[valid] = True
    return m


def generate_steered(lm, cfg, text: str, *, block_idx, unit_v, coef: float,
                     positions=None) -> str:
    """Generate ``text``'s response with ``coef*unit_v`` steered into block
    ``block_idx``'s output. ``positions`` (token indices) -> attack-slot mask;
    ``None`` -> steer all positions. ``coef==0`` -> clean baseline (no hook)."""
    if getattr(lm, "is_mock", False) or not can_steer(lm) or block_idx is None:
        from .mock import mock_generate
        return mock_generate(lm, text)["text"]

    input_ids = build_prompt_ids(lm, text)
    with torch.no_grad():
        if coef == 0.0 or unit_v is None:
            return _generate_real(lm, cfg, input_ids)
        mask = make_mask(int(input_ids.shape[1]), positions)
        with steer(lm, block_idx, unit_v, coef, token_mask=mask):
            return _generate_real(lm, cfg, input_ids)


def repetition_rate(text: str) -> float:
    """Degeneracy proxy: 1 - distinct_bigrams / total_bigrams (0 = no repetition)."""
    toks = text.split()
    if len(toks) < 2:
        return 0.0
    bigrams = list(zip(toks[:-1], toks[1:]))
    return round(1.0 - len(set(bigrams)) / len(bigrams), 5)
