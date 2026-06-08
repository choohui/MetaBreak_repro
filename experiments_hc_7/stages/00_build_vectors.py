"""Stage 00 (model-free) — build the steering vectors from the consumed hc_4
``cos_to_attack`` geometry.

Per pos_offset, at the detector's operating hidden-layer L, construct three unit
directions and save them for the generation stages:

  * ``v_attack``   = unit(dir__cos_to_attack[L])  — literally the cos_to_attack
                     direction (its cosine to this centroid IS the detector
                     scalar). PRIMARY steering vector.
  * ``v_contrast`` = unit(attack_centroid[L] - benign_centroid[L])  — the standard
                     contrastive / ActAdd direction, benign = TRAIN negative class
                     (C/E/F/G). Comparison arm.
  * ``v_rand``     = unit(N(0, I)) with a fixed per-pos seed — control direction.

Also records ``rho[L]`` (mean residual norm over held-out attack-slot tokens) so
the generation stages can scale alpha into an interpretable "fraction of the
layer's typical residual norm".

Outputs (per ``pos{off}/``): ``steer_vectors.npz`` + ``build_vectors.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_7.config import ExpConfig, config_from_args, make_stage_parser  # noqa: E402
from experiments_hc_7.core import io                                                # noqa: E402
from experiments_hc_7.stages import steer_common as sc                              # noqa: E402


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero vector cannot be unit-normalized")
    return (v / n).astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    da, db = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if da == 0 or db == 0:
        return float("nan")
    return round(float(np.dot(a, b) / (da * db)), 5)


def _run_offset(cfg: ExpConfig, off: int, hidden: np.ndarray) -> dict:
    op = sc.read_operating_point(cfg, off)
    layer = sc.resolve_layer(cfg, off, op)
    block_idx = sc.block_index(layer)

    c_attack = sc.attack_centroid(cfg, off, layer)          # [dim]
    c_benign = sc.benign_centroid(cfg, off, layer, hidden)  # [dim]
    rho = sc.rho_at_layer(cfg, off, layer, hidden)

    v_attack = _unit(c_attack)
    v_contrast = _unit(c_attack - c_benign)
    rng = np.random.default_rng(cfg.seed + 1000 * off + 7)
    v_rand = _unit(rng.standard_normal(c_attack.shape[0]))

    pdir = cfg.pos_dir(off)
    pdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pdir / "steer_vectors.npz",
                        v_attack=v_attack, v_contrast=v_contrast, v_rand=v_rand)
    meta = {
        "pos_offset": off,
        "scalarizer": op.get("scalarizer"),
        "layer": layer,
        "block_idx": block_idx,
        "threshold": op.get("threshold"),
        "direction": op.get("direction"),
        "dim": int(c_attack.shape[0]),
        "rho": round(rho, 5),
        "norms": {
            "attack_centroid": round(float(np.linalg.norm(c_attack)), 5),
            "benign_centroid": round(float(np.linalg.norm(c_benign)), 5),
            "attack_minus_benign": round(float(np.linalg.norm(c_attack - c_benign)), 5),
        },
        "cos_attack_contrast": _cos(v_attack, v_contrast),
        "cos_attack_rand": _cos(v_attack, v_rand),
        "rand_seed": int(cfg.seed + 1000 * off + 7),
        "eval_mode": "holdout",
        "source_results": str(cfg.source_results),
    }
    io.write_json(pdir / "build_vectors.json", meta)
    print(f"[00] pos{off}: layer={layer} block={block_idx} rho={rho:.3f} "
          f"cos(attack,contrast)={meta['cos_attack_contrast']}")
    return meta


def run(cfg: ExpConfig, lm=None) -> dict:
    hidden = sc.load_hidden(cfg)
    return {f"pos{off}": _run_offset(cfg, off, hidden) for off in cfg.pos_offsets}


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
