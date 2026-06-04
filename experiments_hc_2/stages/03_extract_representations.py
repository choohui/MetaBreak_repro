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
REPO_ROOT = HERE.parent.parent  # repro_mb (makes experiments_hc_2 importable)
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_2.config import ExpConfig, config_from_args, get_model, make_stage_parser  # noqa: E402
from experiments_hc_2.core import io  # noqa: E402
from experiments_hc_2.core import benign_inject  # noqa: E402
from experiments_hc_2.core.capture import forward_capture, forward_capture_ids, sink_scores  # noqa: E402
from experiments_hc_2.core.features import CaptureSignals, hidden_vector  # noqa: E402
from experiments_hc_2.core.labeling import label_positions_for_variant, sample_ordinary_positions  # noqa: E402
from experiments_hc_2.core.labels import (  # noqa: E402
    CAT_A, CAT_B, CAT_C, CAT_D, CAT_E, CAT_F, CAT_G, CAT_TO_LETTER,
)
from experiments_hc_2.core.template import template_prefix_suffix_lengths  # noqa: E402


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


def _balanced_keep_ids(token_rows: list[dict], cap: int) -> list[int]:
    """row_ids of the balanced subset: each (category, pos_offset) group evenly
    downsampled to <= ``cap``. The FULL set is still saved; this is just the index
    of the balanced view (used by §2 stages 04/05). The gate stages (06/07) use
    the full set so the per-prompt token distribution stays realistic."""
    groups: dict[tuple, list[int]] = {}
    for r in token_rows:
        groups.setdefault((r["category"], r["pos_offset"]), []).append(r["row_id"])
    keep: set[int] = set()
    for ids in groups.values():
        keep.update(_even_subset(sorted(ids), cap))
    return sorted(keep)


def _balanced_cap(token_rows: list[dict]) -> int | None:
    """Auto-cap = the smallest non-empty (category, pos_offset) group among the
    defense types B..G (hc_2 fix for the type-count imbalance, Main.md §2.1).

    Capping every group to this value makes all analyzed types equal-sized for
    the main (balanced) dataset. A (reference) is included so it does not dominate.
    """
    from collections import Counter
    balance_cats = {CAT_B, CAT_C, CAT_D, CAT_E, CAT_F, CAT_G}
    counts = Counter((r["category"], r["pos_offset"]) for r in token_rows)
    vals = [n for (cat, _off), n in counts.items() if cat in balance_cats and n > 0]
    return min(vals) if vals else None


def _census(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r["category"]] = out.get(r["category"], 0) + 1
    return out


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
        # Category C uses token-level injection (exact attack id in a benign
        # carrier); its labeled positions are known, so it bypasses id-matching.
        is_c_inject = (row["variant"] == "benign_mimicry"
                       and row.get("carrier_head") is not None)
        if is_c_inject:
            input_ids, inject_positions = benign_inject.c_input_ids(
                lm, row["carrier_head"], row["carrier_tail"], row["inject_token_ids"])
            cap = forward_capture_ids(lm, input_ids)
        else:
            cap = forward_capture(lm, row["text"])
        sinks = sink_scores(cap)["mean_over_heads"]
        sig = CaptureSignals(cap, sinks)
        n_hidden_layers = sig.n_hidden_layers
        n_attn_layers = sig.n_attn_layers

        if is_c_inject:
            labels = {p: CAT_C for p in inject_positions}
        else:
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

    # Assign row_id over the FULL set; the hidden cube is in the same order.
    for i, r in enumerate(token_rows):
        r["row_id"] = i
    raw_census = _census(token_rows)

    # Balancing produces only an INDEX into the full set (the balanced view used
    # by §2 stages 04/05). The full set is always saved so the §3/§4 sink gate
    # (stages 06/07) sees a realistic body-dominated per-prompt distribution
    # rather than a pre-thinned one (validity fix; see README "balanced vs raw").
    if cfg.cap_per_type is not None:
        cap, cap_mode = cfg.cap_per_type, "explicit"
    elif cfg.balanced:
        cap, cap_mode = _balanced_cap(token_rows), "balanced"
    else:
        cap, cap_mode = None, "none"
    if cap is not None:
        balanced_row_ids = _balanced_keep_ids(token_rows, cap)
    else:
        balanced_row_ids = [r["row_id"] for r in token_rows]
    balanced_census = _census([token_rows[i] for i in balanced_row_ids])

    io.write_jsonl(cfg.out_dir / "tokens.jsonl", token_rows)   # FULL (raw) set
    if store_hidden and hidden_cubes:
        hidden = np.stack(hidden_cubes, axis=0)
    else:
        hidden = np.zeros((0, 0, 0), dtype=np.float16)
    np.savez_compressed(cfg.out_dir / "features.npz", hidden=hidden)

    summary = {
        "n_rows": len(token_rows),                 # full set
        "n_balanced_rows": len(balanced_row_ids),
        "n_hidden_layers": int(n_hidden_layers),   # L+1
        "n_attn_layers": int(n_attn_layers),       # L
        "hidden_dim": int(hidden.shape[2]) if hidden.ndim == 3 and hidden.size else 0,
        "stored_hidden": bool(store_hidden),
        "cap_mode": cap_mode,
        "cap_applied": cap,
        "raw_census": raw_census,                  # FULL set (gate stages use this)
        "census": balanced_census,                 # balanced view (§2 probe/threshold)
        "balanced_row_ids": balanced_row_ids,      # index of the balanced subset
        "pos_offsets": cfg.pos_offsets,
    }
    io.write_json(cfg.out_dir / "extract_summary.json", summary)
    print(f"[03] {len(token_rows)} full token rows; balanced subset "
          f"{len(balanced_row_ids)} (cap_mode={cap_mode}, cap={cap}); hidden={hidden.shape}"
          f"\n[03] raw_census={raw_census}\n[03] balanced_census={balanced_census}")
    return summary


def main() -> None:
    args = make_stage_parser(__doc__).parse_args()
    run(config_from_args(args))


if __name__ == "__main__":
    main()
