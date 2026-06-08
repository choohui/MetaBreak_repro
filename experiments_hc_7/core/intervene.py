"""Optional REAL token-exclusion intervention (stage 08 ``--real_intervention``).

The block-rate proxy (core.cascade.prompt_block_and_asr) is the headline ASR
metric and needs no model. This module is the stronger, model-dependent check:
for each attack prompt, rebuild the exact generation-prompt token ids (the same
space stage 03 labelled), DROP the flagged attack-slot tokens, RE-GENERATE, and
re-judge. It is never exercised by the smoke test (mock has no real generate).

Depends only on ``repro_mb/src`` (allowed) for the refusal judge.
"""

from __future__ import annotations

import torch

from .model import LoadedModel


def _gen_prompt_ids(lm: LoadedModel, text: str) -> list[int]:
    ids = lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True, return_tensors="pt", return_dict=False)
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return [int(x) for x in ids[0].tolist()]


def regenerate_excluding(lm: LoadedModel, text: str, drop_positions: set[int],
                         max_new_tokens: int = 256, temperature: float = 0.0) -> str:
    """Re-generate the assistant turn after deleting ``drop_positions`` from the
    prompt's token ids. Returns the decoded response text."""
    ids = _gen_prompt_ids(lm, text)
    keep = [ids[i] for i in range(len(ids)) if i not in drop_positions]
    if not keep:
        keep = ids
    t = torch.tensor([keep], device=lm.device)
    eos = getattr(lm.tokenizer, "eos_token_id", None)
    with torch.no_grad():
        out = lm.model.generate(
            t, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=eos,
        )
    gen = out[0, len(keep):]
    return lm.tokenizer.decode(gen, skip_special_tokens=True)


def real_intervention_asr(lm: LoadedModel, prompt_by_sample: dict[int, dict],
                          drop_by_sample: dict[int, set[int]], success: set[int],
                          max_new_tokens: int = 256, temperature: float = 0.0) -> dict:
    """Re-generate every successful attack prompt with its flagged tokens removed
    and re-judge. Returns ASR before/after + per-prompt records."""
    from src.evaluate import evaluate_one      # refusal-keyword judge (repro_mb/src)

    attack_samples = sorted(prompt_by_sample.keys())
    succeeded = [s for s in attack_samples if s in success]
    records = []
    still = 0
    for s in succeeded:
        rec = prompt_by_sample[s]
        drop = drop_by_sample.get(s, set())
        resp = regenerate_excluding(lm, rec["text"], drop, max_new_tokens, temperature)
        ev = evaluate_one(resp, rec["text"], None)
        survived = bool(ev["refusal_success"])    # still a successful attack after defense
        still += int(survived)
        records.append({"sample_index": s, "n_dropped": len(drop),
                        "still_success": survived, "response_text": resp[:400]})
    n_attack = len(attack_samples)
    n_succ = len(succeeded)
    return {
        "mode": "real_intervention",
        "n_attack_prompts": n_attack,
        "n_succeeded": n_succ,
        "asr_before": round(n_succ / n_attack, 5) if n_attack else None,
        "asr_after": round(still / n_attack, 5) if n_attack else None,
        "block_rate_among_successful": round((n_succ - still) / n_succ, 5) if n_succ else None,
        "records": records,
    }
