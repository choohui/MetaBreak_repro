"""Stage 04 (needs model) — utility cost of steering on HELD-OUT BENIGN prompts.

Re-runs the same alpha-sweep (token_mode='all'; benign prompts have no attack
slot) on benign prompts and measures collateral damage:

  * over_refusal_rate = fraction of benign prompts the steering turns INTO a
    refusal (1 - refusal_success). A good defense steers attacks down WITHOUT
    refusing benign requests.
  * repetition_rate (degeneracy proxy) = mean 1 - distinct_bigrams/total_bigrams.

Outputs (per ``pos{off}/``): ``steer_utility.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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

from src.evaluate import evaluate_one  # noqa: E402


def _run_offset(cfg: ExpConfig, off: int, lm) -> dict:
    vecs = sc.load_vectors(cfg, off)
    meta = vecs["meta"]
    block_idx, rho = meta["block_idx"], meta["rho"]
    benign = sc.held_out_benign_samples(cfg, off, n=cfg.utility_n)

    records: list[dict] = []
    baseline_cache: dict[int, str] = {}
    arms = [(vt, a) for vt in cfg.vector_types for a in cfg.alphas]
    for vt, alpha in tqdm(arms, desc=f"[04] pos{off} utility"):
        unit_v = vecs["v"][f"v_{vt}"]
        coef = float(alpha) * float(rho)
        for bp in benign:
            si = bp["sample_index"]
            if alpha == 0.0 and si in baseline_cache:
                resp = baseline_cache[si]
            else:
                resp = generate_steered(lm, cfg, bp["text"], block_idx=block_idx,
                                        unit_v=unit_v, coef=coef, positions=None)
                if alpha == 0.0:
                    baseline_cache[si] = resp
            ev = evaluate_one(resp, bp["text"], None)
            records.append({"vector_type": vt, "alpha": float(alpha), "sample_index": si,
                            "over_refusal": (not bool(ev["refusal_success"])),
                            "repetition_rate": repetition_rate(resp)})

    # aggregate
    rows = []
    keyf = lambda r: (r["vector_type"], r["alpha"])
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault(keyf(r), []).append(r)
    for (vt, a), rs in sorted(groups.items()):
        n = len(rs)
        rows.append({
            "vector_type": vt, "alpha": a, "n": n,
            "over_refusal_rate": round(sum(1 for r in rs if r["over_refusal"]) / n, 5) if n else None,
            "mean_repetition_rate": round(float(np.mean([r["repetition_rate"] for r in rs])), 5) if n else None,
        })
    pdir = cfg.pos_dir(off)
    io.write_csv(pdir / "steer_utility.csv", rows)
    io.write_jsonl(pdir / "steer_utility.jsonl", records)
    print(f"[04] pos{off}: {len(benign)} benign x {len(arms)} arms -> {len(records)} gens")
    return {"pos_offset": off, "n_benign": len(benign), "n_arms": len(arms),
            "steering_observable": not getattr(lm, "is_mock", False)}


def run(cfg: ExpConfig, lm=None) -> dict:
    lm = get_model(cfg, lm)
    return {f"pos{off}": _run_offset(cfg, off, lm) for off in cfg.pos_offsets}


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
