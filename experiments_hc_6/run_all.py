from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments_hc_6.config import ALL_STAGES, config_from_args, make_parser  # noqa: E402
from experiments_hc_6.core.model import get_model  # noqa: E402

STAGE_FILES = {
    "00": "00_prepare_data.py",
    "01": "01_build_prompts.py",
    "02": "02_capture_representations.py",
    "03": "03_split_balance.py",
    "04": "04_fit_detectors.py",
    "05": "05_threshold_stability.py",
    "06": "06_counterfactual_eval.py",
    "07": "07_mask_candidate_search.py",
    "08": "08_mask_eval.py",
    "09": "09_fit_steering_vectors.py",
    "10": "10_steering_eval.py",
    "11": "11_stress_controls.py",
    "12": "12_report.py",
}
NEEDS_MODEL = {"00", "02", "07", "08", "10"}


def load_stage(num: str):
    path = HERE / "stages" / STAGE_FILES[num]
    spec = importlib.util.spec_from_file_location(f"hc6_stage_{num}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def run(cfg, lm=None) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    stages = [s for s in ALL_STAGES if s in set(cfg.stages)]
    if lm is None and (set(stages) & NEEDS_MODEL):
        lm = get_model(cfg)
    print(f"[hc6] out_dir={cfg.out_dir}")
    print(f"[hc6] stages={stages} smoke={cfg.smoke}")
    results = {}
    for num in tqdm(stages, desc="[hc6] stages", unit="stage"):
        print(f"\n===== stage {num}: {STAGE_FILES[num]} =====")
        use_lm = lm if num in NEEDS_MODEL else None
        results[num] = load_stage(num).run(cfg, use_lm)
    print("[hc6] done")
    return results


def main() -> None:
    p = make_parser(__doc__)
    p.add_argument("--stages", default=",".join(ALL_STAGES))
    args = p.parse_args()
    args.stages = [s.strip() for s in str(args.stages).split(",") if s.strip()]
    run(config_from_args(args))


if __name__ == "__main__":
    main()

