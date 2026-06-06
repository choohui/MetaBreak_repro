"""One-shot orchestrator — load the model ONCE and run the selected stages.

Pipeline (hc_4_claude — a non-logistic scalar-signal + threshold per-token defense):
    00  embedding analysis              [needs model]
    01  build 7-type prompts (A-G)
    02  run ASR (B/D/F)                 [needs model]
    03  extract representations         [needs model]  7-way balanced cap (A-G equal)
    04  scalarize                       fit-on-train scalars + honest train AUC
    05  threshold fit + stability       train-only operating-point selection
    06  held-out evaluation             the hc_2 failure scenario, reported honestly
    07  counterfactual paired controls  B-C, B-F, D-E, D-F, F-G
    08  token-exclusion defense + ASR   block-rate proxy (+ optional real re-gen)
    09  robustness / ablations          normalisation / family / sink-gate arms

Run as a module from ``repro_mb`` (the directory above this package):
    python -m experiments_hc_4_claude.run_all --model /path/to/Llama-3.1-8B-Instruct --n 150
    python -m experiments_hc_4_claude.run_all --smoke --n 4          # model-free smoke run
    python -m experiments_hc_4_claude.run_all --model ... --stages 04,05,06
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent          # .../repro_mb/experiments_hc_4_claude
REPO_ROOT = HERE.parent                          # .../repro_mb
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_4_claude.config import ALL_STAGES, config_from_args, get_model, make_stage_parser  # noqa: E402

STAGES_DIR = HERE / "stages"
STAGE_FILES = {
    "00": "00_embedding_analysis.py",
    "01": "01_build_prompts.py",
    "02": "02_run_asr.py",
    "03": "03_extract_representations.py",
    "04": "04_scalarize.py",
    "05": "05_threshold_fit.py",
    "06": "06_holdout_eval.py",
    "07": "07_counterfactual.py",
    "08": "08_token_defense.py",
    "09": "09_robustness.py",
}
BASE_NEEDS_MODEL = {"00", "02", "03"}


def needs_model(cfg) -> set[str]:
    nm = set(BASE_NEEDS_MODEL)
    if getattr(cfg, "real_intervention", False):
        nm.add("08")
    return nm


def load_stage(num: str):
    path = STAGES_DIR / STAGE_FILES[num]
    spec = importlib.util.spec_from_file_location(f"hc4c_stage_{num}", path)
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
