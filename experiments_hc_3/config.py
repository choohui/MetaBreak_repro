"""Configuration for the hc_3 active-sink/cascade experiments.

hc_3 is intentionally model-free by default: it consumes stage-03 artifacts
(`tokens.jsonl`, `features.npz`, `extract_summary.json`, optional `asr.jsonl`)
from hc_2 or another compatible extraction run, then writes its own reports.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SOURCE_OUT = REPO_ROOT / "experiments_hc_2" / "results" / "hc2_llama31_8b"
DEFAULT_OUT = HERE / "results" / "hc3_active_sink"

ALL_STAGES = ["08", "09", "10", "11"]


@dataclass
class ExpConfig:
    source_out_dir: Path = DEFAULT_SOURCE_OUT
    out_dir: Path = DEFAULT_OUT
    pos_offsets: list[int] = field(default_factory=lambda: [0, 1])
    asr_judge: str = "both"
    balanced: bool = True
    fpr: float = 0.01
    token_recall: float = 0.95
    top_ks: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    seed: int = 0
    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))

    def __post_init__(self) -> None:
        self.source_out_dir = Path(self.source_out_dir)
        self.out_dir = Path(self.out_dir)

    def pos_dir(self, pos_offset: int) -> Path:
        return self.out_dir / f"pos{pos_offset}"


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source_out_dir", default=str(DEFAULT_SOURCE_OUT),
                   help="Existing extraction artifact directory to analyze.")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT),
                   help="hc_3 output directory.")
    p.add_argument("--pos_offsets", default="0,1")
    p.add_argument("--asr_judge", default="both", choices=["keyword", "guard", "both"])
    p.add_argument("--balanced", action=argparse.BooleanOptionalAction, default=True,
                   help="Use extract_summary balanced_row_ids when available.")
    p.add_argument("--fpr", type=float, default=0.01,
                   help="Prompt/cascade operating FPR target.")
    p.add_argument("--token_recall", type=float, default=0.95,
                   help="Token recall target for high-recall candidate generation.")
    p.add_argument("--top_ks", default="1,2,3,5,10",
                   help="Comma list of within-prompt top-k sink ranks.")
    p.add_argument("--seed", type=int, default=0)


def config_from_args(args: argparse.Namespace) -> ExpConfig:
    pos_offsets = [int(x) for x in str(args.pos_offsets).split(",") if x.strip()]
    top_ks = [int(x) for x in str(args.top_ks).split(",") if x.strip()]
    return ExpConfig(
        source_out_dir=Path(args.source_out_dir),
        out_dir=Path(args.out_dir),
        pos_offsets=pos_offsets,
        asr_judge=args.asr_judge,
        balanced=args.balanced,
        fpr=args.fpr,
        token_recall=args.token_recall,
        top_ks=top_ks,
        seed=args.seed,
        stages=getattr(args, "stages", None) or list(ALL_STAGES),
    )


def make_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    add_common_args(p)
    return p


def materialize_artifacts(cfg: ExpConfig) -> None:
    """Copy compatible hc_2 artifacts into hc_3's output directory.

    The analysis stages can read directly from `source_out_dir`, but copying the
    core artifacts makes each hc_3 run self-contained and protects it from later
    edits to the source experiment.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tokens.jsonl", "features.npz", "extract_summary.json",
                 "asr.jsonl", "asr_summary.json", "prompts.jsonl"):
        src = cfg.source_out_dir / name
        if src.exists():
            shutil.copy2(src, cfg.out_dir / name)


