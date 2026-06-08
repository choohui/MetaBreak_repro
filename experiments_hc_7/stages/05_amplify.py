"""Stage 05 (needs model) — causal AMPLIFICATION (the up-test).

The defense direction is only causal if pushing the OTHER way also moves
behavior. We take the held-out attack prompts that the model REFUSED at baseline
(alpha=0 failures — no headroom to go down) and apply POSITIVE alpha along the
same steering vector. If the refusals flip to successes ("rescue rate" rises with
alpha), steering controls attack success in BOTH directions, proving the cos_to_attack
direction is causal rather than a mere correlate.

Outputs (per ``pos{off}/``): ``amplify.csv`` + ``amplify.jsonl``.
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
from experiments_hc_7.core.steer_eval import generate_steered                                  # noqa: E402
from experiments_hc_7.stages import steer_common as sc                                          # noqa: E402

from src.evaluate import evaluate_one  # noqa: E402


def _run_offset(cfg: ExpConfig, off: int, lm) -> dict:
    vecs = sc.load_vectors(cfg, off)
    meta = vecs["meta"]
    block_idx, rho = meta["block_idx"], meta["rho"]
    attacks = sc.held_out_attack_samples(cfg, off)

    # baseline (alpha=0) -> the refused subset is our amplification target.
    refused = []
    for ap in tqdm(attacks, desc=f"[05] pos{off} baseline"):
        resp = generate_steered(lm, cfg, ap["text"], block_idx=block_idx,
                                unit_v=None, coef=0.0, positions=None)
        if not bool(evaluate_one(resp, ap["text"], None)["refusal_success"]):
            refused.append(ap)

    pos_alphas = [a for a in cfg.alphas if a > 0.0]
    records: list[dict] = []
    for vt in cfg.vector_types:
        unit_v = vecs["v"][f"v_{vt}"]
        for alpha in tqdm(pos_alphas, desc=f"[05] pos{off} {vt} amplify"):
            coef = float(alpha) * float(rho)
            for ap in refused:
                resp = generate_steered(lm, cfg, ap["text"], block_idx=block_idx,
                                        unit_v=unit_v, coef=coef, positions=None)
                ok = bool(evaluate_one(resp, ap["text"], None)["refusal_success"])
                records.append({"vector_type": vt, "alpha": float(alpha),
                                "sample_index": ap["sample_index"], "variant": ap["variant"],
                                "rescued": ok})

    rows = []
    keyf = lambda r: (r["vector_type"], r["alpha"])
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault(keyf(r), []).append(r)
    for (vt, a), rs in sorted(groups.items()):
        n = len(rs)
        rows.append({"vector_type": vt, "alpha": a, "n_baseline_refused": n,
                     "rescue_rate": round(sum(1 for r in rs if r["rescued"]) / n, 5) if n else None})
    pdir = cfg.pos_dir(off)
    io.write_csv(pdir / "amplify.csv", rows)
    io.write_jsonl(pdir / "amplify.jsonl", records)
    print(f"[05] pos{off}: {len(refused)}/{len(attacks)} baseline-refused; "
          f"positive alphas={pos_alphas}")
    return {"pos_offset": off, "n_baseline_refused": len(refused),
            "n_attack_prompts": len(attacks),
            "steering_observable": not getattr(lm, "is_mock", False)}


def run(cfg: ExpConfig, lm=None) -> dict:
    lm = get_model(cfg, lm)
    return {f"pos{off}": _run_offset(cfg, off, lm) for off in cfg.pos_offsets}


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
