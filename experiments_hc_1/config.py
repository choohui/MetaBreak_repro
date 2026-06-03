"""Shared configuration + CLI for every ``experiments_hc_1`` stage.

A single :class:`ExpConfig` is built from argparse and threaded through all
stages so individual scripts and ``run_all.py`` agree on paths and knobs.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../repro_mb/experiments_hc_1
REPO_ROOT = HERE.parent                          # .../repro_mb
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = HERE / "data"
DEFAULT_OUT = HERE / "results" / "hc1_llama31_8b"

DEFAULT_TM1 = REPO_ROOT / "prompts" / "Q_TM-1_Llama.txt"
DEFAULT_Q = REPO_ROOT / "prompts" / "Q.txt"
DEFAULT_REPLACEMENT = REPO_ROOT / "results" / "llama" / "replacement.json"
DEFAULT_BENIGN_SPECIAL = DATA_DIR / "benign_special_prompts.jsonl"
DEFAULT_BENIGN_MIMICRY = DATA_DIR / "benign_mimicry_prompts.jsonl"
DEFAULT_POSITIONED_WORDS = DATA_DIR / "positioned_regular_words.txt"

ALL_STAGES = ["00", "01", "02", "03", "04", "05", "06"]


@dataclass
class ExpConfig:
    model: str | None = None
    model_type: str = "llama"
    dtype: str = "bfloat16"
    device: str | None = None

    n: int = 50
    n_benign: int | None = None        # C/E target count (default: = n)
    out_dir: Path = DEFAULT_OUT

    tm1: Path = DEFAULT_TM1
    q: Path = DEFAULT_Q
    replacement: Path = DEFAULT_REPLACEMENT
    benign_special: Path = DEFAULT_BENIGN_SPECIAL
    benign_mimicry: Path = DEFAULT_BENIGN_MIMICRY
    positioned_words: Path = DEFAULT_POSITIONED_WORDS

    pos_offsets: list[int] = field(default_factory=lambda: [0, 1])
    ordinary: int = 6                  # G tokens sampled per prompt (-1 = all)
    max_a_per_prompt: int = 2          # A (reference) tokens kept per prompt (-1 = all)
    cap_per_type: int | None = None    # global cap per (category, pos_offset); None = no cap
    no_hidden: bool = False
    limit: int | None = None

    guard_model: str | None = None
    max_new_tokens: int = 256
    temperature: float = 0.0

    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))

    # smoke knobs
    smoke: bool = False
    smoke_layers: int = 4
    smoke_dim: int = 64
    smoke_heads: int = 4

    # sink-range (stage 06)
    sink_range_mode: str = "header_slots"   # or "topk"
    sink_range_topk: int = 8

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        for attr in ("tm1", "q", "replacement", "benign_special",
                     "benign_mimicry", "positioned_words"):
            setattr(self, attr, Path(getattr(self, attr)))

    def pos_dir(self, pos_offset: int) -> Path:
        return self.out_dir / f"pos{pos_offset}"


def add_common_args(p: argparse.ArgumentParser, require_model: bool = False) -> None:
    p.add_argument("--model", default=None, required=require_model,
                   help="Local HF path to the victim model (Llama-3.1-8B-Instruct). "
                        "Required for model stages unless --smoke.")
    p.add_argument("--model_type", default="llama")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None, help="cuda / cpu (auto if omitted).")
    p.add_argument("--n", type=int, default=50,
                   help="Base prompts per attack variant / carriers (B/D/F/G).")
    p.add_argument("--n_benign", type=int, default=None,
                   help="Target prompt count for C/E control types (default: = n).")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--tm1", default=str(DEFAULT_TM1))
    p.add_argument("--q", default=str(DEFAULT_Q))
    p.add_argument("--replacement", default=str(DEFAULT_REPLACEMENT))
    p.add_argument("--benign_special", default=str(DEFAULT_BENIGN_SPECIAL))
    p.add_argument("--benign_mimicry", default=str(DEFAULT_BENIGN_MIMICRY))
    p.add_argument("--positioned_words", default=str(DEFAULT_POSITIONED_WORDS))
    p.add_argument("--pos_offsets", default="0,1",
                   help="Comma list of position offsets analyzed (0=slot, 1=next).")
    p.add_argument("--ordinary", type=int, default=6,
                   help="G (ordinary) tokens sampled per prompt; -1 = all content.")
    p.add_argument("--max_a_per_prompt", type=int, default=2,
                   help="A (system-special / reference) tokens kept per prompt; -1 = all. "
                        "Limits over-collection of the repeated template specials.")
    p.add_argument("--cap_per_type", type=int, default=None,
                   help="Global cap on rows per (category, pos_offset); evenly "
                        "downsampled. None = keep all. Use to balance A/G vs others.")
    p.add_argument("--no_hidden", action="store_true",
                   help="Do not store the hidden-state cube (skips cos_to_ref/logreg).")
    p.add_argument("--limit", type=int, default=None,
                   help="Debug: cap the number of prompts processed.")
    p.add_argument("--guard_model", default=None,
                   help="Optional local HF path to Llama-Guard for the ASR judge.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--smoke", action="store_true",
                   help="Use a fake model/tokenizer (no real weights needed).")
    p.add_argument("--smoke_layers", type=int, default=4)
    p.add_argument("--smoke_dim", type=int, default=64)
    p.add_argument("--smoke_heads", type=int, default=4)
    p.add_argument("--sink_range_mode", default="header_slots",
                   choices=["header_slots", "topk"])
    p.add_argument("--sink_range_topk", type=int, default=8)


def config_from_args(args: argparse.Namespace) -> ExpConfig:
    pos_offsets = [int(x) for x in str(args.pos_offsets).split(",") if x.strip() != ""]
    return ExpConfig(
        model=args.model,
        model_type=args.model_type,
        dtype=args.dtype,
        device=args.device,
        n=args.n,
        n_benign=args.n_benign,
        out_dir=Path(args.out_dir),
        tm1=Path(args.tm1),
        q=Path(args.q),
        replacement=Path(args.replacement),
        benign_special=Path(args.benign_special),
        benign_mimicry=Path(args.benign_mimicry),
        positioned_words=Path(args.positioned_words),
        pos_offsets=pos_offsets,
        ordinary=args.ordinary,
        max_a_per_prompt=args.max_a_per_prompt,
        cap_per_type=args.cap_per_type,
        no_hidden=args.no_hidden,
        limit=args.limit,
        guard_model=args.guard_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        stages=getattr(args, "stages", None) or list(ALL_STAGES),
        smoke=args.smoke,
        smoke_layers=args.smoke_layers,
        smoke_dim=args.smoke_dim,
        smoke_heads=args.smoke_heads,
        sink_range_mode=args.sink_range_mode,
        sink_range_topk=args.sink_range_topk,
    )


def get_model(cfg: ExpConfig, lm=None):
    """Return a LoadedModel: the shared ``lm`` if given, else load (or mock)."""
    if lm is not None:
        return lm
    if cfg.smoke:
        from core.mock import build_mock_loaded_model
        return build_mock_loaded_model(cfg.smoke_layers, cfg.smoke_dim, cfg.smoke_heads)
    if not cfg.model:
        raise SystemExit("This stage needs a model: pass --model <path> or --smoke.")
    from core.model import load_model
    return load_model(cfg.model, cfg.model_type, cfg.dtype, cfg.device)


def make_stage_parser(description: str, require_model: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    add_common_args(p, require_model=require_model)
    return p
