"""Shared configuration + CLI for every ``experiments_hc_7`` stage.

hc_7 asks the CAUSAL question hc_4_claude left open: the ``cos_to_attack``
direction SEPARATES attack tokens (held-out AUC 0.96/0.99) — but does steering
generation ALONG/AGAINST that exact direction causally change ASR, and at what
utility cost? The pipeline CONSUMES hc_4_claude's results (fitted centroids, the
saved train/held-out split, prompts, baseline ASR) via ``source_results`` so the
steering vector is identical to the validated detector and evaluation stays on
the same held-out prompts. ``core/`` is copied from hc_4_claude (self-contained).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../repro_mb/experiments_hc_7
REPO_ROOT = HERE.parent                          # .../repro_mb
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = HERE / "results" / "hc7_steer_llama31_8b"
# Where the validated hc_4_claude detector artifacts live (consumed read-only).
DEFAULT_SOURCE = REPO_ROOT / "experiments_hc_4_claude" / "results" / "hc4_claude_llama31_8b"

# 00 build vectors (model-free) ; 03-06 generation (model) ; 07-08 analysis (model-free).
ALL_STAGES = ["00", "03", "04", "05", "06", "07", "08"]
ASR_JUDGE_MODES = ["keyword", "guard", "both"]
VECTOR_TYPES = ["attack", "contrast"]            # primary + contrastive control
TOKEN_MODES = ["all", "attack_slot"]

# Signed steering grid (units of rho[L] = layer's typical residual norm).
# Negative = steer AWAY from attack (defense) ; positive = amplify (causal check).
DEFAULT_ALPHAS = [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]


@dataclass
class ExpConfig:
    model: str | None = None
    model_type: str = "llama"
    dtype: str = "bfloat16"
    device: str | None = None

    out_dir: Path = DEFAULT_OUT
    source_results: Path = DEFAULT_SOURCE        # hc_4_claude results to consume

    asr_judge: str = "keyword"
    guard_model: str | None = None
    max_new_tokens: int = 256
    temperature: float = 0.0

    pos_offsets: list[int] = field(default_factory=lambda: [0, 1])
    # Detector operating-layer per pos_offset (from hc_4 operating_points.json).
    # Overridden at load time by the actual operating_points unless --steer_layers given.
    steer_layers: dict[int, int] | None = None

    # --- steering sweep knobs ---------------------------------------------- #
    alphas: list[float] = field(default_factory=lambda: list(DEFAULT_ALPHAS))
    vector_types: list[str] = field(default_factory=lambda: list(VECTOR_TYPES))
    token_modes: list[str] = field(default_factory=lambda: list(TOKEN_MODES))
    utility_n: int = 50                           # benign prompts for utility/amplify
    over_refusal_budget: float = 0.05             # alpha* picks max ASR-drop within this
    alpha_star: float | None = None               # explicit alpha* for stages 06 (else from 07)
    control_layers: list[int] | None = None       # control-layer arm (hidden-layer indices)

    # --- rigor knobs ------------------------------------------------------- #
    seed: int = 0
    n_bootstrap: int = 1000
    n_perm: int = 1000
    limit: int | None = None                      # debug: cap prompts per arm

    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))

    # smoke knobs
    smoke: bool = False
    smoke_layers: int = 4
    smoke_dim: int = 64
    smoke_heads: int = 4

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.source_results = Path(self.source_results)

    def pos_dir(self, pos_offset: int) -> Path:
        return self.out_dir / f"pos{pos_offset}"

    def src_pos_dir(self, pos_offset: int) -> Path:
        return self.source_results / f"pos{pos_offset}"


def add_common_args(p: argparse.ArgumentParser, require_model: bool = False) -> None:
    p.add_argument("--model", default=None, required=require_model,
                   help="Local HF path to the victim model (Llama-3.1-8B-Instruct). "
                        "Required for generation stages 03-06 unless --smoke.")
    p.add_argument("--model_type", default="llama")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None, help="cuda / cpu (auto if omitted).")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--source_results", default=str(DEFAULT_SOURCE),
                   help="hc_4_claude results dir to consume (fitted centroids, split, prompts, ASR).")
    p.add_argument("--asr_judge", default="keyword", choices=ASR_JUDGE_MODES)
    p.add_argument("--guard_model", default=None, help="Optional local HF path to Llama-Guard.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--pos_offsets", default="0,1", help="Comma list of position offsets.")
    p.add_argument("--steer_layers", default=None,
                   help="Override detector layer per pos as 'off:layer,...' (e.g. '0:32,1:6'). "
                        "Default: read from each pos's operating_points.json.")
    p.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS),
                   help="Comma list of signed steering coefficients (units of rho[L]).")
    p.add_argument("--vector_types", default=",".join(VECTOR_TYPES),
                   help="Comma list from {attack, contrast}.")
    p.add_argument("--token_modes", default=",".join(TOKEN_MODES),
                   help="Comma list from {all, attack_slot}.")
    p.add_argument("--utility_n", type=int, default=50,
                   help="Benign prompts sampled for utility (stage 04) / amplify (stage 05).")
    p.add_argument("--over_refusal_budget", type=float, default=0.05,
                   help="alpha* = max ASR-reduction alpha with over-refusal increase <= this.")
    p.add_argument("--alpha_star", type=float, default=None,
                   help="Explicit alpha* for the controls stage (default: taken from stage 07).")
    p.add_argument("--control_layers", default=None,
                   help="Comma list of hidden-layer indices for the control-layer arm (stage 06).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None, help="Debug: cap prompts per arm.")
    p.add_argument("--smoke", action="store_true", help="Use a fake model (no real weights).")
    p.add_argument("--smoke_layers", type=int, default=4)
    p.add_argument("--smoke_dim", type=int, default=64)
    p.add_argument("--smoke_heads", type=int, default=4)


def _split_floats(s, default):
    if s is None:
        return list(default)
    items = [float(x) for x in str(s).split(",") if x.strip() != ""]
    return items or list(default)


def _split_strs(s, default):
    if s is None:
        return list(default)
    items = [x.strip() for x in str(s).split(",") if x.strip()]
    return items or list(default)


def _parse_steer_layers(s):
    if not s:
        return None
    out: dict[int, int] = {}
    for pair in str(s).split(","):
        pair = pair.strip()
        if not pair:
            continue
        off, lyr = pair.split(":")
        out[int(off)] = int(lyr)
    return out


def config_from_args(args: argparse.Namespace) -> ExpConfig:
    pos_offsets = [int(x) for x in str(args.pos_offsets).split(",") if x.strip() != ""]
    control_layers = ([int(x) for x in str(args.control_layers).split(",") if x.strip() != ""]
                      if args.control_layers else None)
    return ExpConfig(
        model=args.model,
        model_type=args.model_type,
        dtype=args.dtype,
        device=args.device,
        out_dir=Path(args.out_dir),
        source_results=Path(args.source_results),
        asr_judge=args.asr_judge,
        guard_model=args.guard_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        pos_offsets=pos_offsets,
        steer_layers=_parse_steer_layers(args.steer_layers),
        alphas=_split_floats(args.alphas, DEFAULT_ALPHAS),
        vector_types=_split_strs(args.vector_types, VECTOR_TYPES),
        token_modes=_split_strs(args.token_modes, TOKEN_MODES),
        utility_n=args.utility_n,
        over_refusal_budget=args.over_refusal_budget,
        alpha_star=args.alpha_star,
        control_layers=control_layers,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        n_perm=args.n_perm,
        limit=args.limit,
        stages=getattr(args, "stages", None) or list(ALL_STAGES),
        smoke=args.smoke,
        smoke_layers=args.smoke_layers,
        smoke_dim=args.smoke_dim,
        smoke_heads=args.smoke_heads,
    )


def get_model(cfg: ExpConfig, lm=None):
    """Return a LoadedModel: the shared ``lm`` if given, else load (or mock)."""
    if lm is not None:
        return lm
    if cfg.smoke:
        from experiments_hc_7.core.mock import build_mock_loaded_model
        return build_mock_loaded_model(cfg.smoke_layers, cfg.smoke_dim, cfg.smoke_heads)
    if not cfg.model:
        raise SystemExit("This stage needs a model: pass --model <path> or --smoke.")
    from experiments_hc_7.core.model import load_model
    return load_model(cfg.model, cfg.model_type, cfg.dtype, cfg.device)


def make_stage_parser(description: str, require_model: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    add_common_args(p, require_model=require_model)
    return p
