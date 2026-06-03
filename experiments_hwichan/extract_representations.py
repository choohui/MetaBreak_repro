"""Forward every labelled prompt and extract per-token internal features.

For each input we run ``common.forward_capture`` once and then, for every token
position that belongs to one of the study categories (plus its immediately
following position, ``pos_offset==1``), we record:

  * the hidden-state vector from every layer (saved to ``features.npz``),
  * per-layer hidden-state L2 norm,
  * per-layer attention sink score (mean over heads),
  * per-layer value-vector norm ``||V||`` and attention-output norm ``||O||``.

Layer indexing convention (documented for the analysis stage):
  * ``hidden`` cube has ``L+1`` layers: index 0 = token embeddings,
    index ``l`` (>=1) = output of decoder layer ``l``.
  * ``sink`` / ``value_norm`` / ``output_norm`` lists have ``L`` entries:
    entry ``i`` is produced inside decoder layer ``i`` and aligns with
    hidden layer ``i+1``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:  # progress bar is optional; degrade gracefully if tqdm is absent
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else []

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_hwichan.common import (  # noqa: E402
    CAT_ORDINARY,
    LoadedModel,
    forward_capture,
    label_token_categories,
    load_model,
    read_jsonl,
    sample_ordinary_positions,
    sink_scores,
    template_prefix_suffix_lengths,
    write_json,
    write_jsonl,
)


def extract(
    lm: LoadedModel,
    prompt_rows: list[dict],
    ordinary_positions_per_prompt: int = 4,
    desc: str = "extract",
    store_hidden: bool = True,
) -> tuple[list[dict], np.ndarray]:
    """Forward every prompt and emit per-token features.

    ``ordinary_positions_per_prompt``: number of E (ordinary) tokens sampled per
    prompt; pass ``-1`` to label **every** ordinary content token (Tier-2 honest
    full-sequence coverage). ``store_hidden=False`` skips the [N, L+1, dim] hidden
    cube (``features.npz``) so the full-E run stays small on disk; the scalar
    features (sink / hidden_norm / value_norm / output_norm) are unaffected, but
    ``cos_to_D`` will be unavailable downstream.
    """
    prefix_len, suffix_len = template_prefix_suffix_lengths(lm)
    tpl = lm.template

    token_rows: list[dict] = []
    hidden_cubes: list[np.ndarray] = []  # each [L+1, dim] float16

    progress = tqdm(prompt_rows, desc=desc, unit="prompt", dynamic_ncols=True)
    for sample_i, row in enumerate(progress):
        cap = forward_capture(lm, row["text"])
        seq = len(cap.input_ids)
        sinks = sink_scores(cap)["mean_over_heads"]   # [L, seq]
        value_norms = cap.value_norms                 # [L, seq]
        output_norms = cap.output_norms               # [L, seq]
        hidden = cap.hidden_states                     # list[L+1] of [seq, dim]
        n_hidden_layers = len(hidden)

        labels = label_token_categories(
            cap.input_ids, tpl, prefix_len, suffix_len, row["variant"]
        )
        # Always seed a few ordinary (E) content tokens as the negative baseline,
        # so both experiments have an ordinary-regular comparison even when no
        # dedicated "ordinary" variant is present (e.g. experiment 1). setdefault
        # never overwrites an A/B/C/D label.
        if ordinary_positions_per_prompt != 0:
            max_pos = None if ordinary_positions_per_prompt < 0 else ordinary_positions_per_prompt
            for p in sample_ordinary_positions(
                cap.input_ids, tpl, prefix_len, suffix_len,
                max_positions=max_pos,
            ):
                labels.setdefault(p, CAT_ORDINARY)

        # hidden-state norms for all positions, all layers, once.
        hidden_norms_all = [
            torch.linalg.vector_norm(h, dim=-1) for h in hidden
        ]  # list[L+1] of [seq]

        for p, category in sorted(labels.items()):
            for pos_offset in (0, 1):
                pos = p + pos_offset
                if pos >= seq:
                    continue
                if store_hidden:
                    cube = np.stack(
                        [h[pos].numpy().astype(np.float16) for h in hidden], axis=0
                    )  # [L+1, dim]
                    hidden_cubes.append(cube)
                token_rows.append(
                    {
                        "row_id": len(token_rows),
                        "sample_index": sample_i,
                        "prompt_idx": row.get("idx"),
                        "source": row.get("source"),
                        "variant": row.get("variant"),
                        "position_kind": row.get("position_kind"),
                        "category": category,
                        "position": int(pos),
                        "pos_offset": pos_offset,
                        "token_id": int(cap.input_ids[pos]),
                        "decoded": lm.tokenizer.decode([int(cap.input_ids[pos])]),
                        "seq_len": seq,
                        "hidden_norm": [
                            round(float(hn[pos]), 5) for hn in hidden_norms_all
                        ],
                        "sink": [round(float(sinks[l, pos]), 8) for l in range(sinks.shape[0])],
                        "value_norm": [
                            round(float(value_norms[l, pos]), 5)
                            for l in range(value_norms.shape[0])
                        ],
                        "output_norm": [
                            round(float(output_norms[l, pos]), 5)
                            for l in range(output_norms.shape[0])
                        ],
                        "n_hidden_layers": n_hidden_layers,
                    }
                )

        if hasattr(progress, "set_postfix"):
            progress.set_postfix(tokens=len(token_rows), refresh=False)

    if hidden_cubes:
        hidden_arr = np.stack(hidden_cubes, axis=0)  # [N, L+1, dim] float16
    else:
        hidden_arr = np.zeros((0, 0, 0), dtype=np.float16)
    return token_rows, hidden_arr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--model_type", default="llama")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--prompts", required=True, help="exp{1,2}_prompts.jsonl")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--limit", type=int, default=None, help="cap number of prompt rows")
    p.add_argument(
        "--ordinary", type=int, default=4,
        help="E (ordinary) tokens sampled per prompt; -1 = ALL content tokens "
             "(Tier-2 honest full-sequence coverage).",
    )
    p.add_argument(
        "--no_hidden", action="store_true",
        help="skip the [N, L+1, dim] hidden cube (features.npz). Use with "
             "--ordinary -1 to keep the full-E run small; disables cos_to_D.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lm = load_model(args.model, args.model_type, args.dtype, args.device)
    prompt_rows = read_jsonl(args.prompts, limit=args.limit)
    token_rows, hidden_arr = extract(
        lm, prompt_rows,
        ordinary_positions_per_prompt=args.ordinary,
        store_hidden=not args.no_hidden,
    )

    write_jsonl(out_dir / "tokens.jsonl", token_rows)
    if not args.no_hidden:
        np.savez_compressed(out_dir / "features.npz", hidden=hidden_arr)

    # quick category census for sanity
    census: dict[str, int] = {}
    for r in token_rows:
        key = f"{r['category']}|off{r['pos_offset']}"
        census[key] = census.get(key, 0) + 1
    summary = {
        "model": args.model,
        "prompts": args.prompts,
        "n_prompt_rows": len(prompt_rows),
        "n_token_rows": len(token_rows),
        "hidden_shape": list(hidden_arr.shape),
        "category_census": census,
    }
    write_json(out_dir / "extract_summary.json", summary)
    print(f"[extract] tokens={len(token_rows)} hidden={list(hidden_arr.shape)}")
    print(f"[extract] census={census}")


if __name__ == "__main__":
    main()
