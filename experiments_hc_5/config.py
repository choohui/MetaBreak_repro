from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = HERE / "data"
DEFAULT_OUT_BASE = HERE / "results"
ALL_STAGES = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"]


@dataclass
class ExpConfig:
    model: str | None = None
    model_type: str = "llama"
    run_name: str = "hc5_token_sanitize"
    out_dir: Path | None = None
    n: int = 150
    n_benign: int | None = None
    seed: int = 0
    smoke: bool = False
    device: str | None = None
    dtype: str = "bfloat16"
    max_new_tokens: int = 128
    skip_generation: bool = False
    balance_mode: str = "letter_pos_split"
    split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
    split_group_key: str = "sample_index"
    pos_offsets: list[int] = field(default_factory=lambda: [0, 1])
    fpr_targets: list[float] = field(default_factory=lambda: [0.001, 0.01, 0.05])
    scalar_rule_max_terms: int = 3
    defense_actions: list[str] = field(default_factory=lambda: [
        "no_op", "mask_token", "drop_token", "drop_token_pm1", "drop_detected_span", "prompt_block"
    ])
    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))
    neighbor_rank: int = 1
    top_ks: list[int] = field(default_factory=lambda: [1, 3, 5])
    ordinary_per_prompt: int = 6
    max_a_per_prompt: int = 2
    stress_n: int = 20

    q_path: Path = DATA_DIR / "Q.txt"
    tm1_path: Path = DATA_DIR / "Q_TM-1_Llama.txt"
    benign_mimicry_path: Path = DATA_DIR / "benign_mimicry_prompts.jsonl"
    benign_special_path: Path = DATA_DIR / "benign_special_prompts.jsonl"
    positioned_words_path: Path = DATA_DIR / "positioned_regular_words.txt"

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir) if self.out_dir else DEFAULT_OUT_BASE / self.run_name
        self.n_benign = self.n if self.n_benign is None else self.n_benign
        self.q_path = Path(self.q_path)
        self.tm1_path = Path(self.tm1_path)
        self.benign_mimicry_path = Path(self.benign_mimicry_path)
        self.benign_special_path = Path(self.benign_special_path)
        self.positioned_words_path = Path(self.positioned_words_path)

    @property
    def replacement_path(self) -> Path:
        return self.out_dir / "replacement.json"

    @property
    def prompts_path(self) -> Path:
        return self.out_dir / "prompts.jsonl"

    @property
    def tokens_path(self) -> Path:
        return self.out_dir / "tokens.jsonl"

    @property
    def inputs_path(self) -> Path:
        return self.out_dir / "capture_inputs.jsonl"

    @property
    def features_path(self) -> Path:
        return self.out_dir / "features.npz"


def _int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _str_list(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _stage_list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        raw = [str(x).strip() for x in value if str(x).strip()]
    else:
        raw = _str_list(str(value))
    return [x.zfill(2) if x.isdigit() else x for x in raw]


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None)
    p.add_argument("--model_type", default="llama")
    p.add_argument("--run_name", default="hc5_token_sanitize")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--n_benign", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--skip_generation", action="store_true")
    p.add_argument("--pos_offsets", default="0,1")
    p.add_argument("--fpr_targets", default="0.001,0.01,0.05")
    p.add_argument("--scalar_rule_max_terms", type=int, default=3)
    p.add_argument("--defense_actions", default="no_op,mask_token,drop_token,drop_token_pm1,drop_detected_span,prompt_block")
    p.add_argument("--neighbor_rank", type=int, default=1)
    p.add_argument("--top_ks", default="1,3,5")
    p.add_argument("--ordinary_per_prompt", type=int, default=6)
    p.add_argument("--max_a_per_prompt", type=int, default=2)
    p.add_argument("--stress_n", type=int, default=20)


def make_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    add_common_args(p)
    return p


def config_from_args(args: argparse.Namespace) -> ExpConfig:
    return ExpConfig(
        model=args.model,
        model_type=args.model_type,
        run_name=args.run_name,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        n=args.n,
        n_benign=args.n_benign,
        seed=args.seed,
        smoke=args.smoke,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        skip_generation=args.skip_generation,
        pos_offsets=_int_list(args.pos_offsets),
        fpr_targets=_float_list(args.fpr_targets),
        scalar_rule_max_terms=args.scalar_rule_max_terms,
        defense_actions=_str_list(args.defense_actions),
        stages=_stage_list(getattr(args, "stages", None)) if getattr(args, "stages", None) else list(ALL_STAGES),
        neighbor_rank=args.neighbor_rank,
        top_ks=_int_list(args.top_ks),
        ordinary_per_prompt=args.ordinary_per_prompt,
        max_a_per_prompt=args.max_a_per_prompt,
        stress_n=args.stress_n,
    )
