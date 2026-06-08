"""Stage 03 (needs model) — the steering alpha-sweep on HELD-OUT attack prompts.

For each (pos_offset, vector_type in {attack, contrast}, token_mode in {all,
attack_slot}, signed alpha) we steer ``coef = alpha * rho[L]`` into the detector
block and regenerate every held-out attack prompt (variant B/D/F), then judge ASR
via the same refusal-keyword judge hc_4 used. alpha=0 reproduces the undefended
baseline; negative alpha = steer AWAY from attack (defense); positive = amplify.

Outputs (per ``pos{off}/``):
    steer_sweep.jsonl    one row per (arm, prompt)
    steer_sweep_asr.csv  ASR per (vector_type, token_mode, alpha) overall + per-variant
"""

from __future__ import annotations

import sys
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_7.config import ExpConfig, config_from_args, get_model, make_stage_parser  # noqa: E402
from experiments_hc_7.core import io                                                           # noqa: E402
from experiments_hc_7.core.steer_eval import generate_steered, repetition_rate                 # noqa: E402
from experiments_hc_7.stages import steer_common as sc                                          # noqa: E402

from src.evaluate import GuardJudge, evaluate_one  # noqa: E402

_VARIANT_TO_LETTER = {"malicious_mimicry": "B", "malicious_special": "D",
                      "positioned_regular": "F"}


def _make_judge(cfg, lm):
    if cfg.asr_judge in ("guard", "both") and not getattr(lm, "is_mock", False) and cfg.guard_model:
        return GuardJudge(cfg.guard_model, lm.device, cfg.dtype)
    return None


def _asr_rows(records: list[dict]) -> list[dict]:
    """Aggregate per-prompt records into ASR per (vector_type, token_mode, alpha)."""
    keyf = lambda r: (r["vector_type"], r["token_mode"], r["alpha"])
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault(keyf(r), []).append(r)
    out = []
    for (vt, tm, a), rs in sorted(groups.items()):
        n = len(rs)
        succ = sum(1 for r in rs if r["success"])
        row = {"vector_type": vt, "token_mode": tm, "alpha": a, "n": n,
               "asr": round(succ / n, 5) if n else None}
        for var, lt in _VARIANT_TO_LETTER.items():
            vr = [r for r in rs if r["variant"] == var]
            row[f"asr_{lt}"] = (round(sum(1 for r in vr if r["success"]) / len(vr), 5)
                                if vr else None)
        out.append(row)
    return out


def _run_offset(cfg: ExpConfig, off: int, lm, judge) -> dict:
    vecs = sc.load_vectors(cfg, off)
    meta = vecs["meta"]
    block_idx, rho = meta["block_idx"], meta["rho"]
    attacks = sc.held_out_attack_samples(cfg, off)
    slot_pos = sc.attack_slot_positions(cfg, off)

    baseline_cache: dict[int, str] = {}    # alpha==0 response per sample (mode/vec-independent)
    records: list[dict] = []
    n_skip_slot = 0

    arms = [(vt, tm, a) for vt in cfg.vector_types for tm in cfg.token_modes for a in cfg.alphas]
    for vt, tm, alpha in tqdm(arms, desc=f"[03] pos{off} arms"):
        unit_v = vecs["v"][f"v_{vt}"]
        coef = float(alpha) * float(rho)
        for ap in attacks:
            si = ap["sample_index"]
            positions = None
            if tm == "attack_slot":
                positions = slot_pos.get(si)
                if not positions:
                    n_skip_slot += 1
                    continue
            if alpha == 0.0 and si in baseline_cache:
                resp = baseline_cache[si]
            else:
                resp = generate_steered(lm, cfg, ap["text"], block_idx=block_idx,
                                        unit_v=unit_v, coef=coef, positions=positions)
                if alpha == 0.0:
                    baseline_cache[si] = resp
            ev = evaluate_one(resp, ap["text"], judge)
            records.append({
                "pos_offset": off, "vector_type": vt, "token_mode": tm, "alpha": float(alpha),
                "sample_index": si, "variant": ap["variant"], "letter": ap["letter"],
                "success": bool(ev["refusal_success"]),
                "guard_success": ev.get("guard_success"),
                "repetition_rate": repetition_rate(resp),
                "response_preview": resp[:200],
            })

    pdir = cfg.pos_dir(off)
    io.write_jsonl(pdir / "steer_sweep.jsonl", records)
    asr = _asr_rows(records)
    io.write_csv(pdir / "steer_sweep_asr.csv", asr)
    report = {"pos_offset": off, "layer": meta["layer"], "block_idx": block_idx,
              "rho": rho, "n_attack_prompts": len(attacks), "n_arms": len(arms),
              "n_skipped_attack_slot": n_skip_slot, "eval_mode": "holdout",
              "steering_observable": not getattr(lm, "is_mock", False)}
    io.write_json(pdir / "steer_sweep_report.json", report)
    print(f"[03] pos{off}: {len(attacks)} attack prompts x {len(arms)} arms "
          f"-> {len(records)} gens; skipped(slot)={n_skip_slot}")
    return report


def run(cfg: ExpConfig, lm=None) -> dict:
    lm = get_model(cfg, lm)
    judge = _make_judge(cfg, lm)
    return {f"pos{off}": _run_offset(cfg, off, lm, judge) for off in cfg.pos_offsets}


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
