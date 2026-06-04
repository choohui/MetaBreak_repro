"""One-shot orchestrator — load the model ONCE and run the selected stages.

Stages (Main.md sections) — the full campaign from "what signal" to a tested defense:
    00  embedding analysis            (§1)   [needs model]
    01  build 7-type prompts          (§2.1)  C carriers integrated from the start
    02  run ASR (B/D/F)               (§2.1)  [needs model]  keyword / guard judge
    03  extract representations       (§2.2)  [needs model]  balanced + raw census
    04  probe + cosine                (§2.3)  naive + prompt-level (GroupKFold) AUC
    05  single-threshold defense      (§2.3)  + operating_points.json
    06  sink-filter sweep             (§3)    1st-stage gate effectiveness
    07  cascade defense               (§4)    2-stage detector, block-rate / FPR / ASR

Run as a module from ``repro_mb`` (the directory above this package):
    python -m experiments_hc_2.run_all --model /path/to/Llama-3.1-8B-Instruct --n 150
    python -m experiments_hc_2.run_all --smoke --n 3            # model-free smoke run
    python -m experiments_hc_2.run_all --model ... --stages 03,04,05
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent          # .../repro_mb/experiments_hc_2
REPO_ROOT = HERE.parent                          # .../repro_mb
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_2.config import ALL_STAGES, config_from_args, get_model, make_stage_parser  # noqa: E402

STAGES_DIR = HERE / "stages"
STAGE_FILES = {
    "00": "00_embedding_analysis.py",
    "01": "01_build_prompts.py",
    "02": "02_run_asr.py",
    "03": "03_extract_representations.py",
    "04": "04_probe_cosine.py",
    "05": "05_threshold_defense.py",
    "06": "06_sink_filter.py",
    "07": "07_cascade_defense.py",
}
NEEDS_MODEL = {"00", "02", "03"}


def load_stage(num: str):
    path = STAGES_DIR / STAGE_FILES[num]
    spec = importlib.util.spec_from_file_location(f"hc2_stage_{num}", path)
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
    for num in tqdm(stages, desc="[run_all] stages", unit="stage"):
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
