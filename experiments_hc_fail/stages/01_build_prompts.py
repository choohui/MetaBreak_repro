"""Stage 01 - build A-G prompt rows from local data seeds."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG.parent
for p in (str(REPO_ROOT), str(PKG.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments_hc_fail.config import ExpConfig, config_from_args, make_parser  # noqa: E402
from experiments_hc_fail.core import io  # noqa: E402


def _load_words(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _assistant_header(repl: dict) -> str:
    if repl.get("assistant_header"):
        return str(repl["assistant_header"])
    eot, start, end = repl["target_token_strs"][:3]
    return f"{eot}{start}assistant{end}\n\n"


def _apply_mimicry(prompt: str, special_strs: list[str], replacement_strs: list[str]) -> str:
    pairs = sorted(zip(special_strs, replacement_strs), key=lambda x: len(x[0]), reverse=True)
    out = prompt
    for s, r in pairs:
        out = out.replace(s, r)
    return out


def build_rows(cfg: ExpConfig) -> list[dict]:
    repl = io.read_json(cfg.replacement)
    data = cfg.data_dir
    tm1 = io.load_prompt_blocks(data / "Q_TM-1_Llama.txt")
    q = io.load_prompt_blocks(data / "Q.txt")
    words = _load_words(data / "positioned_regular_words.txt")
    benign_special = io.read_jsonl(data / "benign_special_prompts.jsonl")
    benign_mimicry = io.read_jsonl(data / "benign_mimicry_prompts.jsonl")
    n = min(cfg.n, len(tm1))
    n_benign = cfg.n_benign if cfg.n_benign is not None else cfg.n

    special_strs = list(repl["target_token_strs"])
    replacement_strs = list(repl["best_triple_decoded"])
    replacement_ids = [int(x) for x in repl["best_triple_ids"]]
    assistant_header = _assistant_header(repl)

    rows: list[dict] = []

    def add(idx, source, variant, text, slot_word=None, extra=None):
        row = {
            "sample_index": len(rows),
            "idx": idx,
            "source": source,
            "variant": variant,
            "text": text,
            "slot_word": slot_word,
        }
        if extra:
            row.update(extra)
        rows.append(row)

    for i in range(n):
        d_text = tm1[i]
        add(i, "Q_TM-1_Llama", "malicious_special", d_text)
        add(i, "Q_TM-1_Llama", "malicious_mimicry", _apply_mimicry(d_text, special_strs, replacement_strs))
        word = words[i % len(words)] if words else "pineapple"
        add(i, "Q_TM-1_Llama", "positioned_regular", d_text.replace(assistant_header, f" {word} "), slot_word=word)

    for i, seed in enumerate(benign_mimicry[:n_benign]):
        text = str(seed["text"])
        head = text + "\nToken under discussion:"
        tail = "\nExplain the word in this benign context."
        rendered = head + "".join(replacement_strs) + tail
        add(seed.get("id", i), "benign_mimicry_prompts", "benign_mimicry", rendered,
            extra={"carrier_head": head, "carrier_tail": tail, "inject_token_ids": replacement_ids})

    for i, seed in enumerate(benign_special[:n_benign]):
        add(seed.get("id", i), "benign_special_prompts", "benign_special", str(seed["text"]))

    for i in range(min(cfg.n, len(q))):
        add(i, "Q", "ordinary", q[i].strip())

    return rows


def run(cfg: ExpConfig) -> dict:
    rows = build_rows(cfg)
    io.write_jsonl(cfg.prompts, rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["variant"]] = counts.get(r["variant"], 0) + 1
    print(f"[01] wrote {len(rows)} prompts -> {cfg.prompts}")
    print(f"[01] counts={counts}")
    return {"n_rows": len(rows), "counts": counts, "path": str(cfg.prompts)}


def main() -> None:
    run(config_from_args(make_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
