"""Stage 03 (Main.md §2.2/2.3) — capture internal representations per prompt.

For every prompt: one forward pass -> sink scores -> 7-type position labeling ->
for each labeled position (and pos_offset) record the per-layer scalar signals
and the hidden-state vector.

Outputs (under ``out_dir``):
    tokens.jsonl        - one row per analyzed (position, pos_offset)
    features.npz        - hidden cube  ``hidden`` : [N, L+1, dim] float16
    extract_summary.json- per-category census + tensor shapes
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import ExpConfig, config_from_args, get_model, make_stage_parser  # noqa: E402
from core import io  # noqa: E402
from core.capture import forward_capture, sink_scores  # noqa: E402
from core.features import CaptureSignals, hidden_vector  # noqa: E402
from core.labeling import label_positions_for_variant, sample_ordinary_positions  # noqa: E402
from core.labels import CAT_A, CAT_G, CAT_TO_LETTER  # noqa: E402
from core.template import template_prefix_suffix_lengths  # noqa: E402


def _even_subset(items: list, k: int) -> list:
    """Deterministically keep ``k`` evenly-spaced elements of ``items``."""
    if k < 0 or len(items) <= k:
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _cap_category(labels: dict[int, str], category: str, max_per: int) -> dict[int, str]:
    """Keep at most ``max_per`` positions of ``category`` (evenly spaced)."""
    if max_per < 0:
        return labels
    cat_positions = sorted(p for p, c in labels.items() if c == category)
    if len(cat_positions) <= max_per:
        return labels
    keep = set(_even_subset(cat_positions, max_per))
    return {p: c for p, c in labels.items() if c != category or p in keep}


def _apply_cap(token_rows: list[dict], hidden_cubes: list, cap: int, store_hidden: bool):
    """Evenly downsample each (category, pos_offset) group to <= ``cap`` rows."""
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(token_rows):
        groups.setdefault((r["category"], r["pos_offset"]), []).append(i)
    keep: set[int] = set()
    for idxs in groups.values():
        keep.update(_even_subset(idxs, cap))
    keep_sorted = sorted(keep)
    new_rows = [token_rows[i] for i in keep_sorted]
    new_hidden = [hidden_cubes[i] for i in keep_sorted] if store_hidden else hidden_cubes
    return new_rows, new_hidden


def _encode_ids(tokenizer, word: str) -> set[int]:
    ids: set[int] = set()
    for variant in (word, " " + word):
        try:
            ids.update(int(x) for x in tokenizer(variant, add_special_tokens=False)["input_ids"])
        except Exception:
            pass
    return ids


def run(cfg: ExpConfig, lm=None) -> dict:
    prompts = io.read_jsonl(cfg.out_dir / "prompts.jsonl")
    if cfg.limit:
        prompts = prompts[: cfg.limit]
    lm = get_model(cfg, lm)
    tpl = lm.template

    repl = io.load_replacement(cfg.replacement)
    mimicry_ids = {int(x) for x in repl.get("best_triple_ids", [])}
    prefix_len, suffix_len = template_prefix_suffix_lengths(lm.tokenizer)

    store_hidden = not cfg.no_hidden
    token_rows: list[dict] = []
    hidden_cubes: list[np.ndarray] = []
    n_hidden_layers = n_attn_layers = 0

    for row in tqdm(prompts, desc="[03] extract"):
        cap = forward_capture(lm, row["text"])
        sinks = sink_scores(cap)["mean_over_heads"]
        sig = CaptureSignals(cap, sinks)
        n_hidden_layers = sig.n_hidden_layers
        n_attn_layers = sig.n_attn_layers

        extra = {"mimicry_ids": mimicry_ids}
        if row["variant"] == "positioned_regular" and row.get("slot_word"):
            extra["slot_word_ids"] = _encode_ids(lm.tokenizer, row["slot_word"])

        labels = label_positions_for_variant(
            cap.input_ids, tpl, prefix_len, suffix_len, row["variant"], extra)

        # Cap the repeated A (reference) tokens collected per prompt.
        labels = _cap_category(labels, CAT_A, cfg.max_a_per_prompt)

        for p in sample_ordinary_positions(cap.input_ids, tpl, prefix_len, suffix_len,
                                            max_positions=cfg.ordinary):
            labels.setdefault(p, CAT_G)

        seq = len(cap.input_ids)
        for p, category in sorted(labels.items()):
            for off in cfg.pos_offsets:
                pos = p + off
                if pos >= seq:
                    continue
                rec = {
                    "row_id": len(token_rows),
                    "sample_index": row["sample_index"],
                    "prompt_idx": row["idx"],
                    "variant": row["variant"],
                    "category": category,
                    "letter": CAT_TO_LETTER[category],
                    "base_position": int(p),
                    "position": int(pos),
                    "pos_offset": int(off),
                    "token_id": int(cap.input_ids[pos]),
                    "decoded": lm.tokenizer.convert_ids_to_tokens(int(cap.input_ids[pos])),
                    "seq_len": int(seq),
                }
                rec.update(sig.signals_at(pos))
                token_rows.append(rec)
                if store_hidden:
                    hidden_cubes.append(hidden_vector(cap, pos))

    # Global per-(category, pos_offset) cap: evenly downsample over-represented
    # types (chiefly A and G) so we do not keep/analyze every token.
    if cfg.cap_per_type is not None:
        token_rows, hidden_cubes = _apply_cap(
            token_rows, hidden_cubes, cfg.cap_per_type, store_hidden)

    census: dict[str, int] = {}
    for new_id, r in enumerate(token_rows):
        r["row_id"] = new_id                      # re-index to match hidden cube
        census[r["category"]] = census.get(r["category"], 0) + 1

    io.write_jsonl(cfg.out_dir / "tokens.jsonl", token_rows)
    if store_hidden and hidden_cubes:
        hidden = np.stack(hidden_cubes, axis=0)
    else:
        hidden = np.zeros((0, 0, 0), dtype=np.float16)
    np.savez_compressed(cfg.out_dir / "features.npz", hidden=hidden)

    summary = {
        "n_rows": len(token_rows),
        "n_hidden_layers": int(n_hidden_layers),   # L+1
        "n_attn_layers": int(n_attn_layers),       # L
        "hidden_dim": int(hidden.shape[2]) if hidden.ndim == 3 and hidden.size else 0,
        "stored_hidden": bool(store_hidden),
        "census": census,
        "pos_offsets": cfg.pos_offsets,
    }
    io.write_json(cfg.out_dir / "extract_summary.json", summary)
    print(f"[03] {len(token_rows)} token rows; hidden={hidden.shape}; census={census}")
    return summary


def main() -> None:
    args = make_stage_parser(__doc__).parse_args()
    run(config_from_args(args))


if __name__ == "__main__":
    main()
