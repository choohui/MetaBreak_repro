"""choan.md §3.2 — lightweight activation steering (partial-effect defense).

Instead of removing the flagged attack token, push its hidden state toward the
benign side at generation time: subtract ``alpha * unit(diff_means_direction)``
from the residual stream at the flagged positions during the prompt prefill,
then generate normally. choan's finding is that this only PARTIALLY helps — the
headline defense remains drop±1 (§3.3). This module is deliberately minimal (a
single-layer prefill hook + a small alpha sweep), not the full hc_7 causal study.

Real-model only: it needs actual decoder layers + ``model.generate``. Under the
mock model (smoke) there is nothing to hook, so the stage falls back to the
model-free flag-coverage proxy and records that steering was not exercised.

Direction provenance: stage 05 saves the per-layer unit ``diff_means`` direction
(``scalarizer_fit.npz`` -> ``dir__diff_means``). Steering uses that axis even when
the detector family is ``clean`` (cos_to_attack has no signed 1-D push direction).
"""

from __future__ import annotations

import numpy as np

from . import io
from .defense import (flagged_positions_by_sample, predict_flags,
                      select_operating_point)


def _decoder_layers(model):
    base = getattr(model, "model", model)
    return getattr(base, "layers", None)


def load_diff_means_dirs(cfg, off) -> np.ndarray | None:
    """Per-layer unit diff_means directions [L+1, dim] from stage 05, or None."""
    path = cfg.pos_dir(off) / "scalarizer_fit.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return z["dir__diff_means"] if "dir__diff_means" in z else None


class _PrefillSteerHook:
    """Forward hook on one decoder layer: during the PROMPT prefill (full-length
    forward), subtract ``alpha * dir`` from the output hidden state at ``positions``."""

    def __init__(self, layer_module, direction, positions, alpha, prompt_len):
        import torch
        self._h = None
        self._module = layer_module
        self._dir = torch.tensor(np.asarray(direction, dtype=np.float32))
        self._pos = [int(p) for p in positions]
        self._alpha = float(alpha)
        self._prompt_len = int(prompt_len)

    def __enter__(self):
        def hook(_m, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if hs.dim() != 3 or hs.shape[1] != self._prompt_len:
                return out  # only steer the full-prompt prefill, not cached steps
            d = self._dir.to(hs.dtype).to(hs.device)
            for p in self._pos:
                if 0 <= p < hs.shape[1]:
                    hs[0, p, :] = hs[0, p, :] - self._alpha * d
            if isinstance(out, tuple):
                return (hs,) + tuple(out[1:])
            return hs
        self._h = self._module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self._h is not None:
            self._h.remove()
            self._h = None


def run_offset(cfg, off, lm) -> dict:
    """Steer flagged held-out attack prompts and re-measure ASR over an alpha sweep.

    Writes ``defense_steer.json`` with a model-free flag-coverage proxy plus, on a
    real model with ``--real_intervention``, the per-alpha re-generated ASR."""
    from experiments_attack_attr.stages import scalar_common as sc
    from experiments_attack_attr.stages.analysis_common import success_set

    rows, mats, meta, arr = sc.load_scalar_scores(cfg, off)
    sel = select_operating_point(cfg, off)
    pred = predict_flags(mats, sel)
    is_test = arr["is_test"]
    success = success_set(cfg.out_dir, cfg.asr_judge)
    drop_by_sample = flagged_positions_by_sample(cfg, off, rows, pred, is_test)

    n_attack = len({int(r["sample_index"]) for r in rows if r["letter"] in ("B", "D")})
    n_flagged_prompts = len(drop_by_sample)
    report = {
        "pos_offset": off, "action": "steer", "eval_mode": meta["eval_mode"],
        "operating_point": sel, "asr_judge": cfg.asr_judge,
        "steer_alphas": cfg.steer_alphas,
        "flag_coverage": {"n_attack_prompts_full": n_attack,
                          "n_flagged_prompts_holdout": n_flagged_prompts},
    }

    can_steer = (cfg.real_intervention and lm is not None
                 and not getattr(lm, "is_mock", False))
    dirs = load_diff_means_dirs(cfg, off)
    layers = _decoder_layers(lm.model) if (can_steer and lm is not None) else None
    if not can_steer or dirs is None or layers is None:
        report["real_note"] = ("steering needs a real model + diff_means directions; "
                               "flag-coverage proxy only (choan §3.2 is real-model)")
        io.write_json(cfg.pos_dir(off) / "defense_steer.json", report)
        print(f"[defense:steer] pos{off}: flag-coverage only "
              f"({n_flagged_prompts}/{n_attack} attack prompts flagged); not steered")
        return report

    from src.evaluate import evaluate_one
    from .defense import _gen_prompt_ids, generate_from_ids

    # hidden-space best layer -> decoder index (hidden_states[l] is layer l-1 output)
    hl = sel["layer"] if (sel and sel.get("layer") is not None) else dirs.shape[0] - 1
    steer_layer = cfg.steer_layer if cfg.steer_layer is not None else max(0, int(hl) - 1)
    steer_layer = min(steer_layer, len(layers) - 1)
    direction = dirs[min(int(hl), dirs.shape[0] - 1)]

    prompts = {int(r["sample_index"]): r for r in io.read_jsonl(cfg.out_dir / "prompts.jsonl")}
    succeeded = [s for s in drop_by_sample if s in success]
    asr_before = round(len(succeeded) / n_attack, 5) if n_attack else None

    per_alpha = []
    for alpha in cfg.steer_alphas:
        still = 0
        for s in succeeded:
            rec = prompts.get(s)
            if rec is None:
                continue
            ids = _gen_prompt_ids(lm, rec["text"])
            with _PrefillSteerHook(layers[steer_layer], direction,
                                   drop_by_sample[s], alpha, len(ids)):
                resp = generate_from_ids(lm, ids, cfg.max_new_tokens, cfg.temperature)
            if evaluate_one(resp, rec["text"], None)["refusal_success"]:
                still += 1
        per_alpha.append({"alpha": alpha,
                          "asr_after": round(still / n_attack, 5) if n_attack else None})
        print(f"[defense:steer] pos{off} alpha={alpha}: ASR {asr_before} -> "
              f"{per_alpha[-1]['asr_after']}")
    report.update({"steer_layer": steer_layer, "asr_before": asr_before,
                   "per_alpha": per_alpha,
                   "asr_after": min((a["asr_after"] for a in per_alpha
                                     if a["asr_after"] is not None), default=None)})
    io.write_json(cfg.pos_dir(off) / "defense_steer.json", report)
    return report
