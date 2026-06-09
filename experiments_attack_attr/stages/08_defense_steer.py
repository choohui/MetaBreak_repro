"""Stage 08 (choan.md §3.2) — lightweight activation-steering defense.

Push the flagged attack token's hidden state toward the benign side at generation
time (subtract ``alpha * unit(diff_means)`` on the residual stream during the
prompt prefill), sweeping a small alpha grid. choan's finding: this only
PARTIALLY helps, so it is NOT the headline defense (that is drop±1, stage 09).

Real-model only (needs decoder layers + ``model.generate`` + ``--real_intervention``);
under the mock it records flag-coverage and notes that steering was not exercised.

Outputs (per ``pos{off}/``): defense_steer.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_attack_attr.config import ExpConfig, config_from_args, get_model, make_stage_parser  # noqa: E402
from experiments_attack_attr.core import steer  # noqa: E402


def run(cfg: ExpConfig, lm=None) -> dict:
    if cfg.real_intervention and lm is None:
        lm = get_model(cfg, None)
    return {f"pos{off}": steer.run_offset(cfg, off, lm) for off in cfg.pos_offsets}


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
