"""One-shot orchestrator — load the model ONCE and run the selected stages.

Pipeline (experiments_attack_attr — a clean reproduction of choan.md §0–§3.4):
    00  §1   embedding separability                [needs model]
    01  §0+§2.0 build 7-type prompts (A-G)
    02  baseline ASR (B/D/F)                        [needs model]
    03  §2   capture internal representations       [needs model]
    04  §2.1 internal-rep logistic-probe separability
    05  §2.2 scalar signals (clean① + borderline②)
    06  §2.2 detector: threshold fit (train) + held-out eval
    07  §3.1 masking defense                        [model for real ASR]
    08  §3.2 steering defense (partial)             [model for real ASR]
    09  §3.3 drop±1 defense (HEADLINE)              [model for real ASR]
    10  §3.4 consolidated report

Run as a module from ``repro_mb`` (the directory above this package):
    python -m experiments_attack_attr.run_all --model /path/to/Llama-3.1-8B-Instruct --n 150
    python -m experiments_attack_attr.run_all --smoke --n 4          # model-free smoke run
    python -m experiments_attack_attr.run_all --model ... --stages 05,06 --real_intervention
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent          # .../repro_mb/experiments_attack_attr
REPO_ROOT = HERE.parent                          # .../repro_mb
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_attack_attr.config import ALL_STAGES, config_from_args, get_model, make_stage_parser  # noqa: E402

STAGES_DIR = HERE / "stages"
STAGE_FILES = {
    "00": "00_embedding_analysis.py",
    "01": "01_build_prompts.py",
    "02": "02_run_asr.py",
    "03": "03_capture.py",
    "04": "04_separability.py",
    "05": "05_scalars.py",
    "06": "06_detect.py",
    "07": "07_defense_mask.py",
    "08": "08_defense_steer.py",
    "09": "09_defense_drop.py",
    "10": "10_report.py",
}
# Stages that always run a forward pass.
BASE_NEEDS_MODEL = {"00", "02", "03"}
# Defense stages need the model ONLY to measure real ASR; otherwise model-free proxy.
DEFENSE_STAGES = {"07", "08", "09"}


def needs_model(cfg) -> set[str]:
    nm = set(BASE_NEEDS_MODEL)
    if getattr(cfg, "real_intervention", False):
        nm |= DEFENSE_STAGES
    return nm


def load_stage(num: str):
    path = STAGES_DIR / STAGE_FILES[num]
    spec = importlib.util.spec_from_file_location(f"attack_attr_stage_{num}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def run(cfg, lm=None) -> dict:
    stages = [s for s in ALL_STAGES if s in set(cfg.stages)]
    nm = needs_model(cfg)
    print(f"[run_all] out_dir={cfg.out_dir}")
    print(f"[run_all] stages={stages}  smoke={cfg.smoke}  needs_model={sorted(nm & set(stages))}")

    if lm is None and (nm & set(stages)):
        lm = get_model(cfg, None)
        print(f"[run_all] model loaded (mock={getattr(lm, 'is_mock', False)})")

    results: dict[str, dict] = {}
    for num in tqdm(stages, desc="[run_all] stages", unit="stage"):
        mod = load_stage(num)
        use_lm = lm if num in nm else None
        print(f"\n===== stage {num} : {STAGE_FILES[num]} =====")
        results[num] = mod.run(cfg, use_lm)
    print("\n[run_all] done.")
    return results


def main() -> None:
    p = make_stage_parser(__doc__)
    p.add_argument("--stages", default=",".join(ALL_STAGES),
                   help="Comma list of stages to run (default: all).")
    args = p.parse_args()
    args.stages = [s.strip() for s in str(args.stages).split(",") if s.strip()]
    run(config_from_args(args))


if __name__ == "__main__":
    main()
