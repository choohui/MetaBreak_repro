"""One-shot orchestrator — load the model ONCE and run the selected stages.

Stages (Main.md sections):
    00  embedding analysis            (§1)   [needs model]
    01  build 7-type prompts          (§2.1)
    02  run ASR (B/D/F)               (§2.1)  [needs model]
    03  extract representations       (§2.2)  [needs model]
    04  cosine + logreg analysis      (§2.3)
    05  single-threshold defense      (§2.3)
    06  sink-range reduction          (§3)            <- runs together with §2

Examples:
    python run_all.py --model /path/to/Llama-3.1-8B-Instruct --n 10
    python run_all.py --smoke --n 3            # model-free smoke run
    python run_all.py --model ... --stages 03,04,05
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import ALL_STAGES, config_from_args, get_model, make_stage_parser  # noqa: E402

STAGE_FILES = {
    "00": "00_embedding_analysis.py",
    "01": "01_build_prompts.py",
    "02": "02_run_asr.py",
    "03": "03_extract_representations.py",
    "04": "04_analyze_cosine_logreg.py",
    "05": "05_threshold_defense.py",
    "06": "06_sink_range.py",
}
NEEDS_MODEL = {"00", "02", "03"}


def load_stage(num: str):
    path = HERE / STAGE_FILES[num]
    spec = importlib.util.spec_from_file_location(f"hc1_stage_{num}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def run(cfg, lm=None) -> dict:
    stages = [s for s in ALL_STAGES if s in set(cfg.stages)]
    print(f"[run_all] out_dir={cfg.out_dir}")
    print(f"[run_all] stages={stages}  smoke={cfg.smoke}")

    # Load the victim model once if any model-needing stage is selected.
    if lm is None and (NEEDS_MODEL & set(stages)):
        lm = get_model(cfg, None)
        print(f"[run_all] model loaded (mock={getattr(lm, 'is_mock', False)})")

    results: dict[str, dict] = {}
    for num in stages:
        mod = load_stage(num)
        use_lm = lm if num in NEEDS_MODEL else None
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
