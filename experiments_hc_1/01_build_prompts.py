"""Stage 01 (Main.md §2.1) — build the 7-type prompt set.

Produces ``prompts.jsonl`` (model-free). Each row:
    {sample_index, idx, source, variant, position_kind, slot_word, text}

Variants and how their analyzed positions become categories A..G are documented
in :mod:`core.labels` / :mod:`core.labeling`. The single TM-1 tail-attack
structure is used (per the approved plan); F (positioned_regular) is the
position control built by replacing each attack header with a benign slot word.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import ExpConfig, config_from_args, make_stage_parser  # noqa: E402
from core import benign_gen, io  # noqa: E402
from core.mimicry import apply_mimicry, load_prompts  # noqa: E402


def _assistant_header(special_strs: list[str]) -> str:
    """Reconstruct the Llama assistant header from its special-token strings:
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n."""
    eot, start, end = special_strs[0], special_strs[1], special_strs[2]
    return f"{eot}{start}assistant{end}\n\n"


def _load_words(path: Path) -> list[str]:
    words = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.append(line)
    return words


def build_rows(cfg: ExpConfig) -> list[dict]:
    repl = io.load_replacement(cfg.replacement)
    special_strs = repl["target_token_strs"]
    repl_strs = repl["best_triple_decoded"]
    assistant_header = repl.get("assistant_header") or _assistant_header(special_strs)

    tm1 = load_prompts(cfg.tm1)
    carriers = load_prompts(cfg.q)
    slot_words = _load_words(cfg.positioned_words)
    n = min(cfg.n, len(tm1))

    rows: list[dict] = []

    def add(idx, source, variant, position_kind, text, slot_word=None):
        rows.append({
            "sample_index": len(rows),
            "idx": idx,
            "source": source,
            "variant": variant,
            "position_kind": position_kind,
            "slot_word": slot_word,
            "text": text,
        })

    # D / B / F — derived from the first n TM-1 attack prompts.
    for i in range(n):
        d_text = tm1[i]
        add(i, "Q_TM-1_Llama", "malicious_special", "tm1_tail", d_text)
        add(i, "Q_TM-1_Llama", "malicious_mimicry", "tm1_tail",
            apply_mimicry(d_text, special_strs, repl_strs))
        word = slot_words[i % len(slot_words)] if slot_words else "pineapple"
        f_text = d_text.replace(assistant_header, f" {word} ")
        add(i, "Q_TM-1_Llama", "positioned_regular", "tm1_tail", f_text, slot_word=word)

    target = cfg.n_benign if cfg.n_benign is not None else cfg.n

    # C — benign sentences containing the mimicry token (token-identity control).
    #     curated seeds first, then template-generated to reach `target`.
    c_seeds = io.read_jsonl(cfg.benign_mimicry)
    c_gen = benign_gen.generate_benign_mimicry(max(0, target - len(c_seeds)), repl_strs)
    for r in (c_seeds + c_gen)[:target]:
        gen = str(r.get("id", "")).startswith("gen_")
        add(r.get("id"), "benign_mimicry_generated" if gen else "benign_mimicry_prompts",
            "benign_mimicry", "natural", r["text"])

    # E — benign sentences containing a literal special token.
    e_seeds = io.read_jsonl(cfg.benign_special)
    e_gen = benign_gen.generate_benign_special(max(0, target - len(e_seeds)))
    for r in (e_seeds + e_gen)[:target]:
        gen = str(r.get("id", "")).startswith("gen_")
        add(r.get("id"), "benign_special_generated" if gen else "benign_special_prompts",
            "benign_special", "natural", r["text"])

    # G — clean carrier questions (ordinary body tokens sampled at extract time).
    for i in range(min(cfg.n, len(carriers))):
        add(i, "Q", "ordinary", "none", carriers[i].strip())

    return rows


def run(cfg: ExpConfig, lm=None) -> dict:  # lm unused (model-free stage)
    rows = build_rows(cfg)
    out = cfg.out_dir / "prompts.jsonl"
    io.write_jsonl(out, rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["variant"]] = counts.get(r["variant"], 0) + 1
    print(f"[01] wrote {len(rows)} prompts -> {out}")
    print(f"[01] per-variant counts: {counts}")
    return {"n_rows": len(rows), "counts": counts, "path": str(out)}


def main() -> None:
    args = make_stage_parser(__doc__).parse_args()
    run(config_from_args(args))


if __name__ == "__main__":
    main()
