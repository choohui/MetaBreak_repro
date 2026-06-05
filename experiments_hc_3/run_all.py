"""Run hc_3 stages in the recommended order.

Default:
    python -m experiments_hc_3.run_all

This copies compatible hc_2 artifacts into hc_3/results/hc3_active_sink, then
runs:
  08 Active SinkProbe
  09 Prompt-Level Aggregation
  10 Two-Branch Cascade
  11 Counterfactual paired-control validation
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_3.config import ALL_STAGES, config_from_args, make_parser, materialize_artifacts  # noqa: E402

STAGES_DIR = HERE / "stages"
STAGE_FILES = {
    "08": "08_active_sinkprobe.py",
    "09": "09_prompt_aggregation.py",
    "10": "10_two_branch_cascade.py",
    "11": "11_counterfactual_validation.py",
}


def load_stage(num: str):
    path = STAGES_DIR / STAGE_FILES[num]
    spec = importlib.util.spec_from_file_location(f"hc3_stage_{num}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def run(cfg) -> dict:
    materialize_artifacts(cfg)
    stages = [s for s in ALL_STAGES if s in set(cfg.stages)]
    print(f"[hc3] source_out_dir={cfg.source_out_dir}")
    print(f"[hc3] out_dir={cfg.out_dir}")
    print(f"[hc3] stages={stages}")
    results = {}
    for num in tqdm(stages, desc="[hc3] stages", unit="stage"):
        print(f"\n===== stage {num}: {STAGE_FILES[num]} =====")
        results[num] = load_stage(num).run(cfg)
    print("\n[hc3] done.")
    return results


def main() -> None:
    p = make_parser(__doc__)
    p.add_argument("--stages", default=",".join(ALL_STAGES),
                   help="Comma list, e.g. 08,09,10,11")
    args = p.parse_args()
    args.stages = [s.strip() for s in str(args.stages).split(",") if s.strip()]
    run(config_from_args(args))


if __name__ == "__main__":
    main()

