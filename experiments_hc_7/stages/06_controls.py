"""Stage 06 (needs model) — control conditions at alpha*.

At the selected alpha* (stage-07 rule, reusing stages 03/04 outputs; or
``--alpha_star``), re-run on the held-out attack prompts:

  * ``attack``        — the headline steering vector (reference).
  * ``random``        — a random unit direction at the SAME layer + |coef|.
                        Must NOT reduce ASR, ruling out "any perturbation of this
                        magnitude degrades generation / triggers refusals".
  * ``control_layer`` — ``v_attack`` injected at a DIFFERENT block. Weak/insignificant
                        reduction => the effect is specific to the detector layer.

Outputs (per ``pos{off}/``): ``controls.csv`` + ``controls.jsonl``.
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

import numpy as np  # noqa: E402

from experiments_hc_7.config import ExpConfig, config_from_args, get_model, make_stage_parser  # noqa: E402
from experiments_hc_7.core import io                                                           # noqa: E402
from experiments_hc_7.core.steer_eval import generate_steered                                  # noqa: E402
from experiments_hc_7.stages import steer_common as sc                                          # noqa: E402

from src.evaluate import evaluate_one  # noqa: E402


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n else v.astype(np.float32)


def _default_control_layer(layer: int, n_hidden: int) -> int:
    """A layer clearly distinct from the operating layer, clamped to [1, n_hidden-1]."""
    cand = 16 if abs(16 - layer) >= 4 else (layer // 2 if layer > 8 else layer + 8)
    cand = max(1, min(int(cand), n_hidden - 1))
    if cand == layer:
        cand = max(1, layer - 1) if layer > 1 else min(n_hidden - 1, layer + 1)
    return int(cand)


def _asr_on(cfg, lm, attacks, block_idx, unit_v, coef) -> tuple[float, list[dict]]:
    recs, succ = [], 0
    for ap in attacks:
        resp = generate_steered(lm, cfg, ap["text"], block_idx=block_idx,
                                unit_v=unit_v, coef=coef, positions=None)
        ok = bool(evaluate_one(resp, ap["text"], None)["refusal_success"])
        succ += int(ok)
        recs.append({"sample_index": ap["sample_index"], "variant": ap["variant"], "success": ok})
    return (succ / len(attacks) if attacks else None), recs


def _run_offset(cfg: ExpConfig, off: int, lm, hidden) -> dict:
    vecs = sc.load_vectors(cfg, off)
    meta = vecs["meta"]
    layer, block_idx, rho = meta["layer"], meta["block_idx"], meta["rho"]
    sel = sc.select_alpha_star(cfg, off)
    a_star = sel["alpha_star"]
    attacks = sc.held_out_attack_samples(cfg, off)

    rows: list[dict] = []
    all_recs: list[dict] = []
    if a_star is None:
        io.write_csv(cfg.pos_dir(off) / "controls.csv", rows)
        print(f"[06] pos{off}: no alpha* within budget; controls skipped.")
        return {"pos_offset": off, "alpha_star": None, "skipped": True}

    coef = float(a_star) * float(rho)
    n_hidden = hidden.shape[1]
    control_layers = cfg.control_layers or [_default_control_layer(layer, n_hidden)]

    arms = [("attack", layer, block_idx, vecs["v"]["v_attack"], coef),
            ("random", layer, block_idx, vecs["v"]["v_rand"], coef)]
    for lc in control_layers:
        if int(lc) == layer:
            continue
        v_lc = _unit(sc.attack_centroid(cfg, off, int(lc)))
        rho_lc = sc.rho_at_layer(cfg, off, int(lc), hidden)
        arms.append((f"control_layer{int(lc)}", int(lc), sc.block_index(int(lc)),
                     v_lc, float(a_star) * float(rho_lc)))

    for name, lyr, bidx, v, cf in tqdm(arms, desc=f"[06] pos{off} controls"):
        asr, recs = _asr_on(cfg, lm, attacks, bidx, v, cf)
        for r in recs:
            r.update({"arm": name, "layer": lyr})
        all_recs.extend(recs)
        rows.append({"arm": name, "layer": lyr, "alpha_star": a_star,
                     "coef": round(cf, 5), "n": len(attacks), "asr": round(asr, 5) if asr is not None else None})

    base_asr = sel.get("baseline_asr")
    for r in rows:
        r["delta_vs_baseline"] = (round(r["asr"] - base_asr, 5)
                                  if (r["asr"] is not None and base_asr is not None) else None)
    pdir = cfg.pos_dir(off)
    io.write_csv(pdir / "controls.csv", rows)
    io.write_jsonl(pdir / "controls.jsonl", all_recs)
    print(f"[06] pos{off}: alpha*={a_star} baseline_asr={base_asr} -> "
          f"{[(r['arm'], r['asr']) for r in rows]}")
    return {"pos_offset": off, "alpha_star": a_star, "baseline_asr": base_asr,
            "arms": rows, "steering_observable": not getattr(lm, "is_mock", False)}


def run(cfg: ExpConfig, lm=None) -> dict:
    lm = get_model(cfg, lm)
    hidden = sc.load_hidden(cfg)
    return {f"pos{off}": _run_offset(cfg, off, lm, hidden) for off in cfg.pos_offsets}


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
