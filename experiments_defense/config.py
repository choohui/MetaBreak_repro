"""Experiment configuration for the §4 multi-model defense comparison."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results"

ALL_MODELS = ["llama", "qwen", "gemma"]
ALL_DEFENSES = ["ours", "llama_guard", "jbshield", "guard_slm"]


@dataclass
class ExpConfig:
    models: list[str] = field(default_factory=lambda: ["llama"])
    defenses: list[str] = field(default_factory=lambda: list(ALL_DEFENSES))
    model_paths: dict[str, str] = field(default_factory=dict)   # model_type -> HF path
    guard_model: str | None = None                              # Llama-Guard-3 path
    dtype: str = "bfloat16"
    device: str | None = None

    n_attack: int | None = None       # cap on attack prompts (None = all)
    n_benign: int | None = 200
    n_gsm8k: int = 50
    frac_train: float = 0.7
    mimicry: bool = True              # re-mimic attacks via replacement.json if present

    max_new_tokens: int = 256
    out_dir: Path = DEFAULT_OUT
    run_name: str | None = None
    smoke: bool = False

    def slug(self) -> str:
        if self.run_name:
            return self.run_name
        return "def_smoke" if self.smoke else (
            "def_all" if set(self.models) == set(ALL_MODELS) else "def_" + "_".join(self.models))

    @property
    def result_dir(self) -> Path:
        return Path(self.out_dir) / self.slug()


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", default="llama",
                   help="comma list of model families (or 'all'): llama,qwen,gemma")
    p.add_argument("--defenses", default=",".join(ALL_DEFENSES),
                   help="comma list: ours,llama_guard,jbshield,guard_slm")
    p.add_argument("--model_path", action="append", default=[],
                   help="model_type=PATH (repeatable), e.g. --model_path llama=/models/Llama-3.1-8B-Instruct")
    p.add_argument("--models_file", default=None,
                   help="JSON registry of paths: {\"llama\":\"/p\",\"qwen\":\"/p\",\"gemma\":\"/p\",\"guard_model\":\"/p\"}. "
                        "Defaults to experiments_defense/models.json if present. --model_path overrides it.")
    p.add_argument("--guard_model", default=None, help="path to Llama-Guard-3-8B")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--n_attack", type=int, default=None)
    p.add_argument("--n_benign", type=int, default=200)
    p.add_argument("--n_gsm8k", type=int, default=50)
    p.add_argument("--frac_train", type=float, default=0.7)
    p.add_argument("--no_mimicry", action="store_true",
                   help="keep literal special-token attacks (skip replacement.json re-mimicry)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--run_name", default=None)
    p.add_argument("--smoke", action="store_true")


def _load_registry(args: argparse.Namespace) -> tuple[dict[str, str], str | None]:
    """Read the JSON path registry (explicit --models_file, else default
    models.json if present). Returns (model_paths, guard_model)."""
    import json
    path = args.models_file or (str(HERE / "models.json")
                                if (HERE / "models.json").exists() else None)
    if not path:
        return {}, None
    reg = json.loads(Path(path).read_text(encoding="utf-8"))
    guard = reg.pop("guard_model", None)
    return {str(k): str(v) for k, v in reg.items()}, guard


def config_from_args(args: argparse.Namespace) -> ExpConfig:
    paths, reg_guard = _load_registry(args)           # registry first ...
    for item in getattr(args, "model_path", []) or []:  # ... CLI overrides it
        if "=" not in item:
            raise SystemExit(f"--model_path expects model_type=PATH, got {item!r}")
        k, v = item.split("=", 1)
        paths[k.strip()] = v.strip()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if models == ["all"]:
        models = list(ALL_MODELS)
    return ExpConfig(
        models=models,
        defenses=[d.strip() for d in args.defenses.split(",") if d.strip()],
        model_paths=paths,
        guard_model=args.guard_model or reg_guard,
        dtype=args.dtype,
        device=args.device,
        n_attack=args.n_attack,
        n_benign=args.n_benign,
        n_gsm8k=args.n_gsm8k,
        frac_train=args.frac_train,
        mimicry=not args.no_mimicry,
        max_new_tokens=args.max_new_tokens,
        out_dir=Path(args.out_dir),
        run_name=args.run_name,
        smoke=args.smoke,
    )
