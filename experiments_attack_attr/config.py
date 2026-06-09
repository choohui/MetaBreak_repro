"""Shared configuration + CLI for every ``experiments_attack_attr`` stage.

A single :class:`ExpConfig` is built from argparse and threaded through all
stages so individual scripts and ``run_all.py`` agree on paths and knobs.

This folder is a single, structurally-clean reproduction of ``choan.md`` §0-§3.4
("Metabreak semantic-mimicry attack attribution + defense"): detect the attack
tokens inside a prompt from the victim model's INTERNAL representation
(diff-means / cos-to-attack), then sanitize by dropping the flagged token ±1.

Self-containment contract (the hard requirement): it depends ONLY on shared
non-experiment code under ``repro_mb/`` — ``src/`` (the canonical MetaBreak
attack implementation) and ``prompts/`` (Q.txt / Q_TM-1_Llama.txt). It does NOT
import or read from any sibling ``experiments_*`` folder. Datasets
(benign_*.jsonl, positioned words) and the mimicry ``replacement.json`` are
vendored in ``data/``; stage 00 regenerates ``replacement.json`` locally on a
real model run via ``src.embedding`` (so a non-Llama model works too).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../repro_mb/experiments_attack_attr
REPO_ROOT = HERE.parent                          # .../repro_mb
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = HERE / "data"
DEFAULT_OUT = HERE / "results" / "attack_attr_llama31_8b"

DEFAULT_TM1 = REPO_ROOT / "prompts" / "Q_TM-1_Llama.txt"
DEFAULT_Q = REPO_ROOT / "prompts" / "Q.txt"
# Vendored (self-contained). Stage 00 regenerates ``out_dir/replacement.json`` on
# a real model run; ``replacement_path()`` prefers that copy when present.
DEFAULT_REPLACEMENT = DATA_DIR / "replacement.json"
DEFAULT_BENIGN_SPECIAL = DATA_DIR / "benign_special_prompts.jsonl"
DEFAULT_BENIGN_MIMICRY = DATA_DIR / "benign_mimicry_prompts.jsonl"
DEFAULT_POSITIONED_WORDS = DATA_DIR / "positioned_regular_words.txt"

# choan.md-aligned stages (one section -> one stage):
#   00 §1 embedding | 01 §0+§2.0 prompts | 02 baseline ASR | 03 §2 capture
#   04 §2.1 separability | 05 §2.2 scalars | 06 §2.2 detect (threshold+held-out)
#   07 §3.1 mask | 08 §3.2 steer | 09 §3.3 drop±1 | 10 §3.4 report
ALL_STAGES = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

# Defense stages and the token action each applies (choan.md §3.1/§3.2/§3.3).
DEFENSE_STAGES = {"07": "mask", "08": "steer", "09": "drop_pm1"}

ASR_JUDGE_MODES = ["keyword", "guard", "both"]
SCALARIZER_SETS = ["clean", "borderline", "all"]
NORMALIZE_MODES = ["none", "zscore", "rank", "robust"]
DEFENSE_FAMILIES = ["borderline", "clean"]   # which op-point flags tokens
MASK_MODES = ["neutral", "unk", "eos"]       # §3.1 replacement token

# Default threshold selectors fit on TRAIN (see core/thresholds.py).
DEFAULT_THRESHOLD_METHODS = ["youden", "fpr@1", "fpr@5", "eer", "pct_benign@99"]


@dataclass
class ExpConfig:
    model: str | None = None
    model_type: str = "llama"
    dtype: str = "bfloat16"
    device: str | None = None

    n: int = 50
    n_benign: int | None = None        # C/E target count (default: = n)
    out_dir: Path = DEFAULT_OUT

    asr_judge: str = "keyword"         # keyword | guard | both

    tm1: Path = DEFAULT_TM1
    q: Path = DEFAULT_Q
    replacement: Path = DEFAULT_REPLACEMENT
    benign_special: Path = DEFAULT_BENIGN_SPECIAL
    benign_mimicry: Path = DEFAULT_BENIGN_MIMICRY
    positioned_words: Path = DEFAULT_POSITIONED_WORDS

    pos_offsets: list[int] = field(default_factory=lambda: [0, 1])
    ordinary: int = 6                  # G tokens sampled per prompt (-1 = all)
    max_a_per_prompt: int = 4          # A pre-cap per prompt before balancing (-1 = all)
    cap_per_type: int | None = None    # explicit global cap per (category, pos_offset)
    balanced: bool = True              # auto-cap types to equal counts
    balance_a: bool = True             # include A in the 7-way (A-G) equalisation
    no_hidden: bool = False
    limit: int | None = None

    guard_model: str | None = None
    max_new_tokens: int = 256
    temperature: float = 0.0

    # --- scalar-signal knobs (choan.md §2.2) -------------------------------- #
    # "all" by default: choan §2.2 reports BOTH the clean headline (cos_to_attack)
    # and the borderline token-detector (diff_means) that the §3 defenses flag with.
    scalarizer_set: str = "all"        # clean | borderline | all
    scalarizers: list[str] | None = None   # explicit override; None -> use the set
    normalize: str = "none"            # none | zscore | rank | robust (per-prompt)
    threshold_methods: list[str] = field(
        default_factory=lambda: list(DEFAULT_THRESHOLD_METHODS))
    fn_fp_cost: float = 1.0            # cost selector: cost(FN)/cost(FP)
    sink_gate_pct: float = 100.0       # optional 1st-stage sink gate (100 = off)

    # --- defense knobs (choan.md §3.1/§3.2/§3.3) ---------------------------- #
    defense_family: str = "borderline"  # op-point that flags tokens: borderline
                                        # (diff_means, choan §3.4) or clean (cos_to_attack)
    mask_mode: str = "neutral"          # §3.1: neutral word (default) / unk / eos
    steer_alphas: list[float] = field(default_factory=lambda: [2.0, 4.0, 8.0])
    steer_layer: int | None = None      # §3.2: residual layer (None -> diff_means best)
    real_intervention: bool = False     # actually apply the action + RE-GENERATE to
                                        # measure ASR (needs model); else block-rate proxy

    # --- evaluation / rigor knobs ------------------------------------------ #
    holdout_frac: float = 1.0 / 3.0    # prompt-level held-out test fraction
    seed: int = 0
    cv_folds: int = 5
    n_bootstrap: int = 1000            # bootstrap resamples for CIs
    n_perm: int = 1000                 # permutation-test iterations

    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))

    # smoke knobs
    smoke: bool = False
    smoke_layers: int = 4
    smoke_dim: int = 64
    smoke_heads: int = 4

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        for attr in ("tm1", "q", "replacement", "benign_special",
                     "benign_mimicry", "positioned_words"):
            setattr(self, attr, Path(getattr(self, attr)))

    def pos_dir(self, pos_offset: int) -> Path:
        return self.out_dir / f"pos{pos_offset}"

    def replacement_path(self) -> Path:
        """Mimicry signature path. Prefer the copy stage 00 regenerated into
        ``out_dir`` (correct for whatever ``--model`` was used); fall back to the
        vendored ``data/replacement.json`` (Llama-3.1; also the smoke default)."""
        local = self.out_dir / "replacement.json"
        return local if local.exists() else self.replacement


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
    p.add_argument("--max_a_per_prompt", type=int, default=4,
                   help="A (reference) tokens kept per prompt BEFORE balancing; -1 = all.")
    p.add_argument("--cap_per_type", type=int, default=None,
                   help="Explicit global cap on rows per (category, pos_offset). "
                        "Overrides --balanced's auto-cap when given.")
    p.add_argument("--balanced", action=argparse.BooleanOptionalAction, default=True,
                   help="Auto-cap types to equal per-(type,offset) counts. "
                        "--no-balanced keeps raw counts.")
    p.add_argument("--balance_a", action=argparse.BooleanOptionalAction, default=True,
                   help="Include A (reference) in the equalisation so ALL SEVEN types "
                        "A-G have equal counts. --no-balance_a reverts to B..G only.")
    p.add_argument("--asr_judge", default="keyword", choices=ASR_JUDGE_MODES,
                   help="ASR success judge: keyword / guard (needs --guard_model) / both.")
    p.add_argument("--no_hidden", action="store_true",
                   help="Do not store the hidden-state cube (disables hidden-based signals).")
    p.add_argument("--limit", type=int, default=None,
                   help="Debug: cap the number of prompts processed.")
    p.add_argument("--guard_model", default=None,
                   help="Optional local HF path to Llama-Guard for the ASR judge.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)

    # scalar-signal knobs (§2.2)
    p.add_argument("--scalarizer_set", default="clean", choices=SCALARIZER_SETS,
                   help="Which scalar family defines the headline: clean (pure "
                        "measurement / OOD, incl. cos_to_attack), borderline (fitted "
                        "1-D: diff_means/lda_1d/pca_sep_proj), or all.")
    p.add_argument("--scalarizers", default=None,
                   help="Comma list explicitly choosing scalarizers (overrides --scalarizer_set).")
    p.add_argument("--normalize", default="none", choices=NORMALIZE_MODES,
                   help="Per-prompt normalisation applied to every scalar at score time "
                        "(none|zscore|rank|robust) — the anti-distribution-shift wrapper.")
    p.add_argument("--threshold_methods", default=",".join(DEFAULT_THRESHOLD_METHODS),
                   help="Comma list of TRAIN-fit threshold selectors "
                        "(youden, fpr@1, fpr@5, fpr@10, eer, pct_benign@95, pct_benign@99, cost).")
    p.add_argument("--fn_fp_cost", type=float, default=1.0,
                   help="cost selector: cost(false-negative)/cost(false-positive).")
    p.add_argument("--sink_gate_pct", type=float, default=100.0,
                   help="Optional 1st-stage sink gate: keep top X%% per prompt by sink "
                        "before thresholding (100 = gate off).")

    # defense knobs (§3.1/§3.2/§3.3)
    p.add_argument("--defense_family", default="borderline", choices=DEFENSE_FAMILIES,
                   help="Which op-point flags attack tokens for the §3 defenses: "
                        "borderline (diff_means, choan §3.4 headline) or clean (cos_to_attack).")
    p.add_argument("--mask_mode", default="neutral", choices=MASK_MODES,
                   help="§3.1 masking: replace the flagged token with a neutral word "
                        "(default), the unk token, or the eos token.")
    p.add_argument("--steer_alphas", default="2,4,8",
                   help="§3.2: comma list of steering strengths to sweep along -diff_means.")
    p.add_argument("--steer_layer", type=int, default=None,
                   help="§3.2: residual layer to steer (default: the diff_means best layer).")
    p.add_argument("--real_intervention", action="store_true",
                   help="§3 defenses (07/08/09): actually apply the action and RE-GENERATE "
                        "to measure ASR (needs --model). Default is the model-free proxy.")

    # rigor knobs
    p.add_argument("--holdout_frac", type=float, default=1.0 / 3.0,
                   help="Prompt-level held-out test fraction (the generalisation scenario).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--n_bootstrap", type=int, default=1000,
                   help="Bootstrap resamples for AUC / threshold confidence intervals.")
    p.add_argument("--n_perm", type=int, default=1000,
                   help="Permutation-test iterations for the selected operating point.")

    # smoke knobs
    p.add_argument("--smoke", action="store_true",
                   help="Use a fake model/tokenizer (no real weights needed).")
    p.add_argument("--smoke_layers", type=int, default=4)
    p.add_argument("--smoke_dim", type=int, default=64)
    p.add_argument("--smoke_heads", type=int, default=4)


def _split_list(s, default):
    if s is None:
        return None
    items = [x.strip() for x in str(s).split(",") if x.strip()]
    return items or list(default)


def config_from_args(args: argparse.Namespace) -> ExpConfig:
    pos_offsets = [int(x) for x in str(args.pos_offsets).split(",") if x.strip() != ""]
    scalarizers = _split_list(args.scalarizers, [])
    threshold_methods = _split_list(args.threshold_methods, DEFAULT_THRESHOLD_METHODS)
    return ExpConfig(
        model=args.model,
        model_type=args.model_type,
        dtype=args.dtype,
        device=args.device,
        n=args.n,
        n_benign=args.n_benign,
        out_dir=Path(args.out_dir),
        asr_judge=args.asr_judge,
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
        balanced=args.balanced,
        balance_a=args.balance_a,
        no_hidden=args.no_hidden,
        limit=args.limit,
        guard_model=args.guard_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        scalarizer_set=args.scalarizer_set,
        scalarizers=scalarizers,
        normalize=args.normalize,
        threshold_methods=threshold_methods,
        fn_fp_cost=args.fn_fp_cost,
        sink_gate_pct=args.sink_gate_pct,
        defense_family=args.defense_family,
        mask_mode=args.mask_mode,
        steer_alphas=[float(x) for x in str(args.steer_alphas).split(",") if x.strip() != ""],
        steer_layer=args.steer_layer,
        real_intervention=args.real_intervention,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
        cv_folds=args.cv_folds,
        n_bootstrap=args.n_bootstrap,
        n_perm=args.n_perm,
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
        from experiments_attack_attr.core.mock import build_mock_loaded_model
        return build_mock_loaded_model(cfg.smoke_layers, cfg.smoke_dim, cfg.smoke_heads)
    if not cfg.model:
        raise SystemExit("This stage needs a model: pass --model <path> or --smoke.")
    from experiments_attack_attr.core.model import load_model
    return load_model(cfg.model, cfg.model_type, cfg.dtype, cfg.device)


def make_stage_parser(description: str, require_model: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    add_common_args(p, require_model=require_model)
    return p
