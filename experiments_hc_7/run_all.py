"""One-shot orchestrator for experiments_hc_7 — load the model ONCE, run stages.

Pipeline (causal activation-steering along cos_to_attack; consumes hc_4_claude):
    00  build steering vectors          [model-free]  attack / contrast / random + rho
    03  steering alpha-sweep on attacks [needs model] held-out ASR(alpha)
    04  utility on benign prompts       [needs model] over-refusal + degeneracy
    05  amplification (causal up-test)  [needs model] +alpha rescues refused attacks
    06  controls at alpha*              [needs model] random + control-layer arms
    07  analysis                        [model-free]  dose-response, Pareto, CIs, head-to-head
    08  report + figures                [model-free]

Run from ``repro_mb`` (the directory above this package):
    python -m experiments_hc_7.run_all --model /path/to/Llama-3.1-8B-Instruct \
        --source_results experiments_hc_4_claude/results/hc4_claude_llama31_8b
    python -m experiments_hc_7.run_all --smoke           # model-free + mock plumbing
    python -m experiments_hc_7.run_all --model ... --stages 00,03,07,08
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

from experiments_hc_7.config import ALL_STAGES, config_from_args, get_model, make_stage_parser  # noqa: E402

STAGES_DIR = HERE / "stages"
STAGE_FILES = {
    "00": "00_build_vectors.py",
    "03": "03_steer_sweep.py",
    "04": "04_steer_utility.py",
    "05": "05_amplify.py",
    "06": "06_controls.py",
    "07": "07_analysis.py",
    "08": "08_report.py",
}
BASE_NEEDS_MODEL = {"03", "04", "05", "06"}


def needs_model(cfg) -> set[str]:
    return set(BASE_NEEDS_MODEL)


def load_stage(num: str):
    path = STAGES_DIR / STAGE_FILES[num]
    spec = importlib.util.spec_from_file_location(f"hc7_stage_{num}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def run(cfg, lm=None) -> dict:
    stages = [s for s in ALL_STAGES if s in set(cfg.stages)]
    nm = needs_model(cfg)
    print(f"[run_all] out_dir={cfg.out_dir}")
    print(f"[run_all] source={cfg.source_results}")
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
