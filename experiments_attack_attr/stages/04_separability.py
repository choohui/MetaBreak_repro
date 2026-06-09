"""Stage 04 (choan.md §2.1) — is the attack separable in the internal rep?

Before reducing to a single scalar (§2.2), confirm the coarse claim: a
logistic-regression probe over the FULL per-layer hidden vector separates attack
(B,D) from benign (C,E,F,G) tokens. choan.md: "internal representation 을
logistic regression 했더니 잘 나왔다" — reported here as per-layer probe AUC
(GroupKFold by prompt, so a prompt never straddles train/test).

Model-free: reads the stage-03 balanced rows + hidden cube.

Outputs (under ``out_dir``):
    separability.json   - per-layer probe AUC + best layer, per pos_offset
    separability.csv    - flat (pos_offset, layer, auc, balanced_acc)
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # repro_mb (makes experiments_attack_attr importable)
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_attack_attr.config import ExpConfig, config_from_args, make_stage_parser  # noqa: E402
from experiments_attack_attr.core import io  # noqa: E402
from experiments_attack_attr.core.separability import per_layer_separability  # noqa: E402
from experiments_attack_attr.stages.analysis_common import load_artifacts  # noqa: E402


def _run_offset(cfg: ExpConfig, off: int) -> dict:
    rows, hidden, _success = load_artifacts(cfg.out_dir, cfg.asr_judge, balanced=True)
    rows = [r for r in rows if int(r["pos_offset"]) == off]
    rep = per_layer_separability(rows, hidden, folds=cfg.cv_folds, seed=cfg.seed)
    rep["pos_offset"] = off
    io.write_json(cfg.pos_dir(off) / "separability.json", rep)
    print(f"[04] pos{off}: probe best-layer={rep.get('best_layer')} "
          f"AUC={rep.get('best_auc')} ({rep.get('method')}/{rep.get('split')}, "
          f"n_pos={rep.get('n_pos')} n_neg={rep.get('n_neg')})")
    return rep


def run(cfg: ExpConfig, lm=None) -> dict:  # lm unused (model-free stage)
    out, flat = {}, []
    for off in cfg.pos_offsets:
        rep = _run_offset(cfg, off)
        out[f"pos{off}"] = rep
        for pl in rep.get("per_layer", []):
            flat.append({"pos_offset": off, "layer": pl.get("layer"),
                         "auc": pl.get("auc"), "balanced_acc": pl.get("balanced_acc"),
                         "method": pl.get("method"), "split": pl.get("split")})
    io.write_csv(cfg.out_dir / "separability.csv", flat,
                 columns=["pos_offset", "layer", "auc", "balanced_acc", "method", "split"])
    return out


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
