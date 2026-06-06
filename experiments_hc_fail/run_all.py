"""Run experiments_hc_4 end to end."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_hc_fail.config import ALL_STAGES, config_from_args, make_parser  # noqa: E402

STAGE_MODULES = {
    "00": "experiments_hc_4.stages.00_embedding",
    "01": "experiments_hc_4.stages.01_build_prompts",
    "02": "experiments_hc_4.stages.02_capture",
    "03": "experiments_hc_4.stages.03_active_pct_threshold",
    "04": "experiments_hc_4.stages.04_report",
}


def run(cfg):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for stage in cfg.stages:
        if stage not in STAGE_MODULES:
            raise SystemExit(f"unknown stage {stage!r}; choices={ALL_STAGES}")
        mod = importlib.import_module(STAGE_MODULES[stage])
        print("=" * 72)
        print(f"[run_all] stage {stage}: {STAGE_MODULES[stage]}")
        print("=" * 72)
        mod.run(cfg)
    print()
    print(f"[run_all] done. out_dir={cfg.out_dir}")
    return {"out_dir": str(cfg.out_dir)}


def main() -> None:
    run(config_from_args(make_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()

