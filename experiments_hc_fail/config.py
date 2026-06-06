"""Configuration and CLI helpers for experiments_hc_4."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA = HERE / "data"
DEFAULT_RESULTS = HERE / "results"

ALL_STAGES = ["00", "01", "02", "03", "04"]


@dataclass
class ExpConfig:
    model_type: str = "llama"
    model: str | None = None
    run_name: str = "active_value_sweep"
    out_dir: Path = DEFAULT_RESULTS / "active_value_sweep"
    data_dir: Path = DATA
    n: int = 50
    n_benign: int | None = None
    pos_offsets: list[int] = field(default_factory=lambda: [0])
    keep_pcts: list[float] = field(default_factory=lambda: [5.0, 10.0, 20.0, 30.0, 50.0, 100.0])
    fpr: float = 0.01
    seed: int = 0
    topk: int = 200
    ordinary: int = 6
    max_a_per_prompt: int = 2
    max_new_tokens: int = 256
    temperature: float = 0.0
    dtype: str = "bfloat16"
    device: str | None = None
    skip_embedding: bool = False
    skip_generation: bool = False
    smoke: bool = False
    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))

    @property
    def replacement(self) -> Path:
        return self.out_dir / "replacement.json"

    @property
    def prompts(self) -> Path:
        return self.out_dir / "prompts.jsonl"

    @property
    def tokens(self) -> Path:
        return self.out_dir / "active_value_rows.jsonl"

    @property
    def responses(self) -> Path:
        return self.out_dir / "responses.jsonl"

    @property
    def report_json(self) -> Path:
        return self.out_dir / "pct_threshold_report.json"

    @property
    def report_md(self) -> Path:
        return self.out_dir / "pct_threshold_report.md"

    @property
    def sweep_csv(self) -> Path:
        return self.out_dir / "sweep_summary.csv"


def _parse_list(s: str, typ):
    return [typ(x) for x in str(s).split(",") if str(x).strip()]


def make_parser(description: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model_type", default="llama")
    p.add_argument("--model", default=None)
    p.add_argument("--run_name", default="active_value_sweep")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--data_dir", default=str(DATA))
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--n_benign", type=int, default=None)
    p.add_argument("--pos_offsets", default="0")
    p.add_argument("--keep_pcts", default="5,10,20,30,50,100")
    p.add_argument("--fpr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk", type=int, default=200)
    p.add_argument("--ordinary", type=int, default=6)
    p.add_argument("--max_a_per_prompt", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--skip_embedding", action="store_true")
    p.add_argument("--skip_generation", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--stages", default=",".join(ALL_STAGES))
    return p


def config_from_args(args: argparse.Namespace) -> ExpConfig:
    out = Path(args.out_dir) if args.out_dir else DEFAULT_RESULTS / args.run_name
    return ExpConfig(
        model_type=args.model_type,
        model=args.model,
        run_name=args.run_name,
        out_dir=out,
        data_dir=Path(args.data_dir),
        n=args.n,
        n_benign=args.n_benign,
        pos_offsets=_parse_list(args.pos_offsets, int),
        keep_pcts=_parse_list(args.keep_pcts, float),
        fpr=args.fpr,
        seed=args.seed,
        topk=args.topk,
        ordinary=args.ordinary,
        max_a_per_prompt=args.max_a_per_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        dtype=args.dtype,
        device=args.device,
        skip_embedding=args.skip_embedding,
        skip_generation=args.skip_generation,
        smoke=args.smoke,
        stages=_parse_list(args.stages, str),
    )


def require_model(cfg: ExpConfig) -> None:
    if not cfg.smoke and not cfg.model:
        raise SystemExit("--model is required unless --smoke is set.")

