"""Model-free smoke test for hc_3."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_3.config import DEFAULT_SOURCE_OUT, ExpConfig  # noqa: E402
from experiments_hc_3.core import io  # noqa: E402
import experiments_hc_3.run_all as run_all  # noqa: E402


def main() -> int:
    out_dir = HERE / "results" / "_smoke"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cfg = ExpConfig(source_out_dir=DEFAULT_SOURCE_OUT, out_dir=out_dir,
                    pos_offsets=[0], balanced=True, asr_judge="both")
    run_all.run(cfg)
    pdir = cfg.pos_dir(0)
    required = [
        "active_sinkprobe_report.json",
        "prompt_aggregation_report.json",
        "two_branch_cascade_report.json",
        "counterfactual_validation_report.json",
    ]
    missing = [name for name in required if not (pdir / name).exists()]
    if missing:
        print(f"missing outputs: {missing}")
        return 1
    report = io.read_json(pdir / "active_sinkprobe_report.json")
    if report.get("n_rows", 0) <= 0:
        print("active sinkprobe has no rows")
        return 1
    print("OK - hc_3 smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

