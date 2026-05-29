"""MetaBreak TM-1 reproduction — Stage 1: Semantic Mimicry replacement search.

Given a model family (`--model_type`), recovers the chat-template
`assistant_header` (e.g. for Llama-3.x:
``<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n``) and
searches the embedding table for regular-token replacements for every
special token in that header.

Concretely: the header tokenizes (with `add_special_tokens=False`) into
N positions. Some positions are special tokens (model_configs marks them
in `replace_positions`); the rest are literal regular tokens kept as-is
(`fixed_positions`). We brute-force over the top-`k` L2-nearest candidates
per special position and keep the candidate-tuple whose concatenated
string re-tokenizes to the same N-token shape AND minimises the sum of
L2 distances between the candidate input-embeddings and the original
special-token input-embeddings.

This mirrors `MetaBreak/moderator/embeddings.py` but is family-agnostic:
the 3-special-tokens / 5-position case used in the paper for Llama is
recovered automatically; Qwen / Gemma / Phi work via the same code path.

Reference: MetaBreak: Jailbreaking Online LLM Services via Special Token
Manipulation, §5.3 (token substitution).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .model_configs import (
        ModelCfg,
        known_model_types,
        resolve_config,
    )
except ImportError:  # standalone-script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_configs import (  # type: ignore[no-redef]
        ModelCfg,
        known_model_types,
        resolve_config,
    )


def find_most_similar_token_l2(
    token_id: int,
    embedding_weights: torch.Tensor,
    k: int = 200,
) -> tuple[list[int], list[float]]:
    """Top-k nearest tokens by L2 distance on the input-embedding table."""
    input_embedding = embedding_weights[token_id].unsqueeze(0)
    d = embedding_weights.to(torch.float32) - input_embedding.to(torch.float32)
    distances = torch.norm(d, dim=1)
    distances[token_id] = float("inf")  # self exclusion
    top_k_values, top_k_indices = torch.topk(distances, k, largest=False)
    return top_k_indices.tolist(), top_k_values.tolist()


def _assemble_prompt(
    fixed_positions: list[int],
    fixed_strs: list[str],
    replace_positions: list[int],
    candidate_strs: list[str],
    expected_n_tokens: int,
) -> str:
    """Interleave fixed and candidate strings back into one concatenated
    string in original position order.
    """
    parts: list[str | None] = [None] * expected_n_tokens
    for pos, s in zip(fixed_positions, fixed_strs):
        parts[pos] = s
    for pos, cand in zip(replace_positions, candidate_strs):
        parts[pos] = cand
    return "".join(p or "" for p in parts)


def search_best_tuple(
    cfg: ModelCfg,
    tokenizer,
    embedding_weights: torch.Tensor,
    topk: int = 200,
    verbose: bool = True,
) -> dict:
    """Brute-force the best replacement tuple.

    Loop structure: nested over each special position's top-k candidates
    via `itertools.product`. For each tuple, build the candidate prompt
    by interleaving with `cfg.fixed_strs`, re-tokenize, keep only those
    that match `cfg.expected_n_tokens`, score by sum-of-L2 over the
    special positions, return the argmin.
    """
    n_special = len(cfg.replace_positions)
    if n_special == 0:
        raise RuntimeError(
            "model_configs reports zero special tokens in the "
            "assistant_header — nothing to search for."
        )

    # Top-k candidate IDs per special position (parallel to replace_positions).
    cand_lists: list[list[int]] = []
    for tid in cfg.target_token_ids:
        ids, _ = find_most_similar_token_l2(tid, embedding_weights, topk)
        cand_lists.append(ids)

    # Pre-fetch target embeddings for the scoring loop.
    target_emb = [
        embedding_weights[tid].to(torch.float32) for tid in cfg.target_token_ids
    ]

    best_similarity = float("inf")
    best_tuple_ids: list[int] | None = None
    best_decoded: list[str] | None = None
    n_evaluated = 0
    n_kept = 0

    # Pre-filter each candidate list to remove special / out-of-vocab IDs
    # once, instead of inside the hot loop.
    special_set = cfg.special_token_ids
    filtered_lists: list[list[tuple[int, str]]] = []
    for cand_ids in cand_lists:
        kept: list[tuple[int, str]] = []
        for tid in cand_ids:
            if int(tid) in special_set:
                continue
            kept.append((int(tid), tokenizer.decode(int(tid))))
        filtered_lists.append(kept)
        if verbose:
            print(f"[embedding] filtered candidates: {len(kept)}/{len(cand_ids)}"
                  f" (dropped {len(cand_ids) - len(kept)} special/added tokens)")

    for tup in itertools.product(*filtered_lists):
        cand_ids_seq = [x[0] for x in tup]
        cand_strs_seq = [x[1] for x in tup]
        n_evaluated += 1
        temp_prompt = _assemble_prompt(
            cfg.fixed_positions, cfg.fixed_strs,
            cfg.replace_positions, cand_strs_seq,
            cfg.expected_n_tokens,
        )
        temp_ids = tokenizer(temp_prompt, add_special_tokens=False)["input_ids"]
        if len(temp_ids) != cfg.expected_n_tokens:
            continue
        n_kept += 1

        sim = 0.0
        for pos, t_emb in zip(cfg.replace_positions, target_emb):
            cand_emb = embedding_weights[temp_ids[pos]].to(torch.float32)
            sim += float(torch.norm(t_emb - cand_emb))

        if sim < best_similarity:
            best_similarity = sim
            best_tuple_ids = cand_ids_seq
            best_decoded = cand_strs_seq
            if verbose:
                print(
                    f"  new best: ids={best_tuple_ids} "
                    f"strs={[repr(s) for s in best_decoded]} "
                    f"L2_sum={sim:.4f}"
                )

    if best_tuple_ids is None:
        raise RuntimeError(
            f"No valid {n_special}-tuple found that re-tokenizes to "
            f"{cfg.expected_n_tokens} tokens. Try larger --topk."
        )

    return {
        "model_type":             cfg.model_type,
        "auto_detected":          cfg.auto_detected,
        "assistant_header":       cfg.assistant_header,
        "expected_n_tokens":      cfg.expected_n_tokens,
        "target_token_ids":       cfg.target_token_ids,
        "target_token_strs":      cfg.target_token_strs,
        "fixed_positions":        cfg.fixed_positions,
        "fixed_strs":             cfg.fixed_strs,
        "replace_positions":      cfg.replace_positions,
        "best_triple_ids":        best_tuple_ids,
        "best_triple_decoded":    best_decoded,
        "best_similarity_l2_sum": best_similarity,
        "topk":                   topk,
        "n_evaluated":            n_evaluated,
        "n_kept_5tok":            n_kept,
    }


def run(args: argparse.Namespace) -> None:
    """Run embedding search given a pre-built Namespace (no sys.argv parsing).

    Designed to be called directly from ``run.py`` or other orchestrators.
    All fields of *args* must be set; see ``parse_args()`` for the full list.
    """
    device: str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.output) if getattr(args, "output", None) else default_output_path(args.model_type)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[embedding] loading tokenizer + model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    cfg = resolve_config(args.model_type, tokenizer)
    print(f"[embedding] model_type        : {cfg.model_type} "
          f"(auto_detected={cfg.auto_detected})")
    print(f"[embedding] assistant_header  : {cfg.assistant_header!r}")
    print(f"[embedding] target_token_strs : {cfg.target_token_strs}")
    print(f"[embedding] target_token_ids  : {cfg.target_token_ids}")
    print(f"[embedding] fixed (pos/str)   : "
          f"{list(zip(cfg.fixed_positions, [repr(s) for s in cfg.fixed_strs]))}")
    print(f"[embedding] expected_n_tokens : {cfg.expected_n_tokens}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    # We only need the input-embedding matrix, but loading the full causal LM
    # is the simplest reliable path; del model below keeps peak memory bounded.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
    )
    embedding_layer = model.get_input_embeddings()
    print(f"[embedding] embedding shape: {tuple(embedding_layer.weight.shape)}")
    embedding_weights = embedding_layer.weight.detach().cpu().clone()
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(f"[embedding] searching best {len(cfg.replace_positions)}-tuple "
          f"with topk={args.topk} ...")
    result = search_best_tuple(
        cfg, tokenizer, embedding_weights, topk=args.topk, verbose=True,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[embedding] best tuple ids      : {result['best_triple_ids']}")
    print(f"[embedding] best tuple strings  : {result['best_triple_decoded']}")
    print(f"[embedding] L2-sum              : {result['best_similarity_l2_sum']:.4f}")
    print(f"[embedding] saved to            : {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model_type", required=True,
        help="Family slug. Known: " + ", ".join(known_model_types())
             + ". Unknown values trigger tokenizer-based auto-detection.",
    )
    p.add_argument(
        "--model", required=True,
        help="Local HF path to the victim model (tokenizer + embedding "
             "table). Examples: /path/to/Llama-3.1-8B-Instruct, "
             "/path/to/Qwen2.5-7B-Instruct.",
    )
    p.add_argument(
        "--output", default=None,
        help="Where to write the replacement JSON. Default: "
             "results/<model_type>/replacement.json relative to repro_mb/.",
    )
    p.add_argument("--topk", type=int, default=200)
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--dtype", default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    return p.parse_args()


def default_output_path(model_type: str) -> Path:
    return Path(__file__).resolve().parent.parent / "results" / model_type / "replacement.json"


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
