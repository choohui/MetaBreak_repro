"""MetaBreak TM-1 reproduction — Stage 2: apply Semantic Mimicry to prompts.

Reads `prompts/Q_TM-1_{ModelType}.txt` (450 user queries that contain the
literal chat-template special tokens for the chosen family — produced by
`build_prompts.py`) and rewrites each occurrence of those special-token
strings with the regular-token replacement strings produced by
`embedding.py`.

Output:
  * `prompt_mimicked.jsonl` : one record per prompt, with both the original
    and the rewritten version, plus a flag indicating whether the rewrite
    re-tokenizes to all-regular IDs (a sanity check that special-token
    sanitization would have nothing to flag).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

try:
    from .model_configs import known_model_types, resolve_config
except ImportError:  # standalone-script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_configs import known_model_types, resolve_config  # type: ignore[no-redef]


def load_prompts(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    parts = raw.split("\ntest split\n")
    return [p for p in parts if p.strip()]


def apply_mimicry(
    prompt: str,
    special_strs: list[str],
    replacement_strs: list[str],
) -> str:
    """Replace each special-token string in `prompt` with its regular-token
    replacement. Order matters when one special string is a substring of
    another (e.g. Llama's `<|eot_id|>` vs `<|start_header_id|>` share `<|`);
    we apply replacements longest-first to avoid the longer match being
    eaten by a shorter one.
    """
    assert len(special_strs) == len(replacement_strs)
    pairs = sorted(
        zip(special_strs, replacement_strs),
        key=lambda x: len(x[0]),
        reverse=True,
    )
    out = prompt
    for s, r in pairs:
        out = out.replace(s, r)
    return out


def run(args: argparse.Namespace) -> None:
    """Apply mimicry given a pre-built Namespace (no sys.argv parsing).

    Designed to be called directly from ``run.py`` or other orchestrators.
    """
    prompts_path = Path(args.prompts) if getattr(args, "prompts", None) else _default_prompts(args.model_type)
    repl_path    = Path(args.replacement) if getattr(args, "replacement", None) else _default_replacement(args.model_type)
    out_path     = Path(args.output) if getattr(args, "output", None) else _default_output(args.model_type)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(repl_path, "r", encoding="utf-8") as f:
        repl = json.load(f)
    replacement_strs = repl["best_triple_decoded"]
    print(f"[mimicry] model_type={args.model_type} (replacement.json says "
          f"{repl.get('model_type')!r})")
    if repl.get("model_type") and repl["model_type"] != args.model_type:
        raise ValueError(
            f"[mimicry] replacement.json was produced for model_type="
            f"{repl['model_type']!r} but --model_type={args.model_type!r}. "
            f"Re-run embedding.py with the correct --model_type or point "
            f"--replacement at the right file."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cfg = resolve_config(args.model_type, tokenizer)
    special_strs = repl.get("target_token_strs") or cfg.target_token_strs
    print(f"[mimicry] replacement: {repl['best_triple_ids']} "
          f"({[repr(s) for s in replacement_strs]})")
    print(f"[mimicry] specials   : {special_strs}")

    prompts = load_prompts(prompts_path)
    n = min(args.n, len(prompts))
    print(f"[mimicry] loaded {len(prompts)} prompts from {prompts_path}; "
          f"rewriting first {n}")

    n_clean = 0
    special_set = cfg.special_token_ids
    with open(out_path, "w", encoding="utf-8") as f:
        for idx in range(n):
            original = prompts[idx]
            rewritten = apply_mimicry(original, special_strs, replacement_strs)

            ids = tokenizer(rewritten, add_special_tokens=False)["input_ids"]
            has_special = any(int(i) in special_set for i in ids)
            if not has_special:
                n_clean += 1

            rec = {
                "idx": idx,
                "model_type": args.model_type,
                "original": original,
                "mimicked": rewritten,
                "mimicked_token_ids": ids,
                "mimicked_has_special_id": has_special,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[mimicry] wrote {n} records to {out_path}")
    print(f"[mimicry] clean (no special IDs after retokenization): "
          f"{n_clean}/{n}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_type", required=True,
                   help="Family slug. Known: " + ", ".join(known_model_types()))
    p.add_argument("--model", required=True,
                   help="Local HF path to the victim model (tokenizer only).")
    p.add_argument("--prompts", default=None,
                   help="Path to Q_TM-1_<Model>.txt. Default: "
                        "prompts/Q_TM-1_<Model>.txt.")
    p.add_argument("--replacement", default=None,
                   help="Output of embedding.py. Default: "
                        "results/<model_type>/replacement.json.")
    p.add_argument("--output", default=None,
                   help="Where to write rewritten prompts (jsonl). Default: "
                        "results/<model_type>/prompt_mimicked.jsonl.")
    p.add_argument("--n", type=int, default=10,
                   help="Only process the first N prompts (default 10).")
    return p.parse_args()


def _default_prompts(model_type: str) -> Path:
    suffix = model_type[:1].upper() + model_type[1:]
    return Path(__file__).resolve().parent.parent / "prompts" / f"Q_TM-1_{suffix}.txt"


def _default_replacement(model_type: str) -> Path:
    return Path(__file__).resolve().parent.parent / "results" / model_type / "replacement.json"


def _default_output(model_type: str) -> Path:
    return Path(__file__).resolve().parent.parent / "results" / model_type / "prompt_mimicked.jsonl"


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
