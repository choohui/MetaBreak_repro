"""MetaBreak TM-1 reproduction — Step 2: apply Semantic Mimicry to prompts.

Reads `Q_TM-1_Llama.txt` (450 user queries that contain the literal special
tokens `<|eot_id|>`, `<|start_header_id|>`, `<|end_header_id|>`) and rewrites
each occurrence with the regular-token replacement strings produced by
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
from pathlib import Path

from transformers import AutoTokenizer


SPECIAL_STR = ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"]


def load_prompts(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    parts = raw.split("\ntest split\n")
    # Strip trailing newline/empty splits.
    return [p for p in parts if p.strip()]


def apply_mimicry(prompt: str, replacement_strs: list[str]) -> str:
    """Replace the 3 special-token strings with their decoded regular-token
    counterparts. Order matters: do `<|eot_id|>` last so it does not interfere
    with `<|start_header_id|>` / `<|end_header_id|>` which share '<' / '|'."""
    eot, start, end = SPECIAL_STR
    r_eot, r_start, r_end = replacement_strs
    out = prompt.replace(start, r_start)
    out = out.replace(end, r_end)
    out = out.replace(eot, r_eot)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", required=True,
        help="Local HF path to Llama-3.1-8B-Instruct (tokenizer only).",
    )
    p.add_argument(
        "--prompts",
        default=str("repro_mb/Q_TM-1_Llama.txt"),
        help="Path to Q_TM-1_Llama.txt.",
    )
    p.add_argument(
        "--replacement",
        default="./repro_mb_out/replacement.json",
        help="Output of embedding.py (best replacement triple).",
    )
    p.add_argument(
        "--output",
        default="./repro_mb_out/prompt_mimicked.jsonl",
        help="Where to write the rewritten prompts (jsonl).",
    )
    p.add_argument(
        "--n", type=int, default=10,
        help="Only process the first N prompts (default 10).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repl_path = Path(args.replacement)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(repl_path, "r", encoding="utf-8") as f:
        repl = json.load(f)
    replacement_strs = repl["best_triple_decoded"]
    print(f"[mimicry] replacement triple: {repl['best_triple_ids']} "
          f"-> {[repr(s) for s in replacement_strs]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prompts = load_prompts(Path(args.prompts))
    n = min(args.n, len(prompts))
    print(f"[mimicry] loaded {len(prompts)} prompts; rewriting first {n}")

    n_clean = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx in range(n):
            original = prompts[idx]
            rewritten = apply_mimicry(original, replacement_strs)

            # Sanity: does the rewritten prompt still contain any
            # chat-template special tokens after re-tokenization?
            ids = tokenizer(rewritten, add_special_tokens=False)["input_ids"]
            has_special = any(i >= 128000 for i in ids)
            if not has_special:
                n_clean += 1

            rec = {
                "idx": idx,
                "original": original,
                "mimicked": rewritten,
                "mimicked_token_ids": ids,
                "mimicked_has_special_id": has_special,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[mimicry] wrote {n} records to {out_path}")
    print(f"[mimicry] clean (no special IDs after retokenization): "
          f"{n_clean}/{n}")


if __name__ == "__main__":
    main()
