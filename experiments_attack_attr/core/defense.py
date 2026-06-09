"""choan.md §3.1 / §3.3 — token-level sanitizing defenses + their ASR effect.

The detector (stage 06) gives a per-token attack flag from the internal
representation (diff-means by default, choan §3.4). A defense turns that flag
into an action on the prompt's tokens and re-measures attack success:

  * ``mask``           (§3.1) — replace each flagged token with a NEUTRAL word
    (or unk / eos). choan: neutral-word masking lowers ASR (0.647->0.474) while
    unk/eos masking *raises* it.
  * ``drop_token``     — delete the flagged token.
  * ``drop_token_pm1`` (§3.3, HEADLINE) — delete the flagged token and its ±1
    neighbours (the special-token ±1 removal choan found most effective).

Two ASR views are produced:

  * **proxy** (always, model-free): an attack prompt is "blocked" if any of its
    attack-slot B/D tokens is flagged; ``asr_after`` = succeeded-and-unblocked.
    This is the drop-semantics lower bound, evaluated on the held-out TEST split.
  * **real** (only with ``--real_intervention`` and a real ``--model``): rebuild
    the prompt's generation token ids, apply the action to the flagged positions,
    RE-GENERATE, and re-judge. This is the honest number for ALL actions
    (mask/steer cannot be judged by the block-rate proxy).

Depends only on ``core`` + (for the real judge) ``repro_mb/src`` — never on a
sibling ``experiments_*`` folder.
"""

from __future__ import annotations

import numpy as np

from . import io
from . import thresholds as TH
from .cascade import per_type_rates, prompt_block_and_asr

_ATTACK_LETTERS = ("B", "D")


# --------------------------------------------------------------------------- #
# operating point + flagging
# --------------------------------------------------------------------------- #
def select_operating_point(cfg, off) -> dict | None:
    """The op-point that flags attack tokens for the §3 defenses. ``defense_family``
    chooses borderline (choan §3.4: the diff_means token detector specifically) or
    clean (cos_to_attack); falls back to whichever exists."""
    op = io.read_json(cfg.pos_dir(off) / "operating_points.json")
    fam = getattr(cfg, "defense_family", "borderline")
    if fam == "borderline":
        # prefer the diff_means-specific op (choan §3.4) over "best borderline"
        return op.get("diff_means") or op.get("borderline") or op.get("clean")
    return op.get(fam) or op.get("clean") or op.get("borderline")


def predict_flags(mats: dict, sel: dict | None) -> np.ndarray:
    """Boolean attack flag per row from the selected scalarizer/layer/threshold."""
    if not sel or sel.get("scalarizer") not in mats or sel.get("layer") is None:
        any_mat = next(iter(mats.values())) if mats else np.zeros((0, 1))
        return np.zeros(any_mat.shape[0], dtype=bool)
    col = mats[sel["scalarizer"]][:, sel["layer"]]
    return TH.predict(col, sel.get("threshold"), sel.get("direction"))


# --------------------------------------------------------------------------- #
# token-id surgery (mask / drop / drop±1)
# --------------------------------------------------------------------------- #
def neutral_mask_id(tokenizer, mode: str) -> int:
    """The id a masked token becomes. ``unk``/``eos`` reproduce choan's
    ASR-increasing controls; ``neutral`` is a benign filler word."""
    eos = getattr(tokenizer, "eos_token_id", None) or 0
    if mode == "eos":
        return int(eos)
    if mode == "unk":
        return int(getattr(tokenizer, "unk_token_id", None) or eos or 0)
    try:                                   # neutral filler word (choan §3.1)
        ids = tokenizer(" thing", add_special_tokens=False)["input_ids"]
        if ids:
            return int(ids[-1])
    except Exception:
        pass
    return int(eos)


def apply_action_to_ids(ids: list[int], positions: set[int], action: str,
                        mask_id: int) -> list[int]:
    """Return a new id sequence with ``action`` applied at ``positions``."""
    rm: set[int] = set()
    if action == "drop_token_pm1":
        for p in positions:
            rm.update({p - 1, p, p + 1})
    else:
        rm = set(int(p) for p in positions)
    rm = {p for p in rm if 0 <= p < len(ids)}
    if action.startswith("drop"):
        kept = [int(t) for i, t in enumerate(ids) if i not in rm]
        return kept or [int(t) for t in ids]
    if action == "mask":
        return [int(mask_id) if i in rm else int(t) for i, t in enumerate(ids)]
    return [int(t) for t in ids]


# --------------------------------------------------------------------------- #
# per-sample flagged positions (held-out attack prompts only)
# --------------------------------------------------------------------------- #
def flagged_positions_by_sample(cfg, off, rows, pred, split_mask):
    """Map sample_index -> set of flagged B/D token positions (the chat-template id
    space stage 03 labelled), restricted to ``split_mask``.

    ``rows`` carry only row_id/letter/sample_index (scalar_scores view); the actual
    token ``position`` is read from tokens.jsonl aligned by row_id."""
    toks = [r for r in io.read_jsonl(cfg.out_dir / "tokens.jsonl")
            if int(r["pos_offset"]) == off]
    summary = io.read_json(cfg.out_dir / "extract_summary.json")
    keep = set(summary.get("balanced_row_ids", [r["row_id"] for r in toks]))
    by_id = {r["row_id"]: r for r in toks if r["row_id"] in keep}
    drop_by_sample: dict[int, set] = {}
    for i, r in enumerate(rows):
        if not (split_mask[i] and pred[i]):
            continue
        if r["letter"] not in _ATTACK_LETTERS:
            continue
        tok = by_id.get(r["row_id"])
        if tok is None:
            continue
        drop_by_sample.setdefault(int(r["sample_index"]), set()).add(int(tok["position"]))
    return drop_by_sample


# --------------------------------------------------------------------------- #
# real regeneration ASR (mask / drop / drop±1)
# --------------------------------------------------------------------------- #
def _gen_prompt_ids(lm, text: str) -> list[int]:
    import torch
    ids = lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True, return_tensors="pt", return_dict=False)
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return [int(x) for x in ids[0].tolist()]


def generate_from_ids(lm, ids: list[int], max_new_tokens: int, temperature: float) -> str:
    import torch
    t = torch.tensor([list(int(x) for x in ids)], device=lm.device)
    eos = getattr(lm.tokenizer, "eos_token_id", None)
    with torch.no_grad():
        out = lm.model.generate(
            t, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=eos)
    return lm.tokenizer.decode(out[0, len(ids):], skip_special_tokens=True)


def real_action_asr(cfg, lm, action: str, prompt_by_sample: dict[int, dict],
                    drop_by_sample: dict[int, set], success: set[int]) -> dict:
    """Apply ``action`` to every (successful) held-out attack prompt and re-judge.

    Generic over mask/drop/drop±1 (steering uses :mod:`core.steer` instead)."""
    from src.evaluate import evaluate_one
    mask_id = neutral_mask_id(lm.tokenizer, getattr(cfg, "mask_mode", "neutral"))
    attack_samples = sorted(prompt_by_sample.keys())
    succeeded = [s for s in attack_samples if s in success]
    still, records = 0, []
    for s in succeeded:
        rec = prompt_by_sample[s]
        ids = _gen_prompt_ids(lm, rec["text"])
        new_ids = apply_action_to_ids(ids, drop_by_sample.get(s, set()), action, mask_id)
        resp = generate_from_ids(lm, new_ids, cfg.max_new_tokens, cfg.temperature)
        survived = bool(evaluate_one(resp, rec["text"], None)["refusal_success"])
        still += int(survived)
        records.append({"sample_index": s, "n_flagged": len(drop_by_sample.get(s, set())),
                        "still_success": survived, "response_text": resp[:400]})
    n_attack, n_succ = len(attack_samples), len(succeeded)
    return {
        "mode": f"real_{action}",
        "n_attack_prompts": n_attack,
        "n_succeeded": n_succ,
        "asr_before": round(n_succ / n_attack, 5) if n_attack else None,
        "asr_after": round(still / n_attack, 5) if n_attack else None,
        "block_rate_among_successful": round((n_succ - still) / n_succ, 5) if n_succ else None,
        "records": records,
    }


# --------------------------------------------------------------------------- #
# top-level runner (used by stages 07 mask + 09 drop)
# --------------------------------------------------------------------------- #
def run_offset(cfg, off, lm, action: str) -> dict:
    """Flag attack tokens with the §3 detector, apply ``action``, report ASR.

    Always writes the model-free proxy (drop-semantics, held-out). With a real
    model + ``--real_intervention`` also writes the real re-generated ASR."""
    from experiments_attack_attr.stages import scalar_common as sc
    from experiments_attack_attr.stages.analysis_common import success_set

    rows, mats, meta, arr = sc.load_scalar_scores(cfg, off)
    sel = select_operating_point(cfg, off)
    pred = predict_flags(mats, sel)
    is_test = arr["is_test"]
    success = success_set(cfg.out_dir, cfg.asr_judge)

    report = {"pos_offset": off, "action": action, "defense_family": cfg.defense_family,
              "mask_mode": cfg.mask_mode if action == "mask" else None,
              "eval_mode": meta["eval_mode"], "operating_point": sel,
              "asr_judge": cfg.asr_judge}

    def _proxy(mask):
        idx = np.where(mask)[0]
        sub_rows = [rows[i] for i in idx]
        res = prompt_block_and_asr(sub_rows, pred[mask], success)
        res["per_type"] = per_type_rates(sub_rows, pred[mask])
        return res

    proxy = {"test": _proxy(is_test), "full": _proxy(np.ones(len(rows), bool))}
    report["proxy"] = proxy
    t = proxy["test"]
    report["asr_before"] = t["asr_before"]
    report["asr_after_proxy"] = t["asr_after"]
    report["block_rate_among_successful"] = t["block_rate_among_successful"]

    if cfg.real_intervention and lm is not None and not getattr(lm, "is_mock", False):
        drop_by_sample = flagged_positions_by_sample(cfg, off, rows, pred, is_test)
        prompts = {int(r["sample_index"]): r
                   for r in io.read_jsonl(cfg.out_dir / "prompts.jsonl")}
        prompt_by_sample = {s: prompts[s] for s in drop_by_sample if s in prompts}
        real = real_action_asr(cfg, lm, action, prompt_by_sample, drop_by_sample, success)
        report["real"] = real
        report["asr_after"] = real["asr_after"]
        io.write_json(cfg.pos_dir(off) / f"real_asr_{action}.json", real)
    else:
        report["real_note"] = ("real_intervention off or mock model; proxy only "
                               "(mask/steer need a real model to judge ASR honestly)")

    io.write_json(cfg.pos_dir(off) / f"defense_{action}.json", report)
    print(f"[defense:{action}] pos{off}: proxy held-out ASR {t['asr_before']} -> "
          f"{t['asr_after']} (block_among_succ={t['block_rate_among_successful']})")
    return report
