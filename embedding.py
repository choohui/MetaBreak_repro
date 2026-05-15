"""MetaBreak TM-1 reproduction — Step 1: Semantic Mimicry replacement search.

Finds the best `(i, j, k)` regular-token replacement for the Llama-3.1 chat
template special tokens

    <|eot_id|>           id=128009
    <|start_header_id|>  id=128006
    <|end_header_id|>    id=128007

so that the concatenated string `decode(i) + decode(j) + 'assistant' +
decode(k) + '\\n\\n'` re-tokenizes back to a 5-token sequence whose 1st,
2nd and 4th input-embeddings are closest (sum of L2 distances) to the
target special-token input-embeddings.

This mirrors MetaBreak (Wu et al., 2025) `moderator/embeddings.py` but:
  * works on Llama-3.1-8B-Instruct (same tokenizer vocab as 70B/3.3, so
    the special-token IDs are reused),
  * has a configurable `--topk` candidate pool (default 200),
  * writes the result to a JSON file so downstream stages can pick it up.

Reference: MetaBreak: Jailbreaking Online LLM Services via Special Token
Manipulation, §5.3 (token substitution).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer


# Llama-3.1/3.3 chat-template special-token IDs (same across the family).
TARGET_TOKEN_IDS = [128009, 128006, 78191, 128007, 271]
# meaning: [<|eot_id|>, <|start_header_id|>, 'assistant', <|end_header_id|>, '\n\n']
# We search replacements for positions 0 (eot), 1 (start_hdr), 3 (end_hdr).
# Positions 2 ('assistant') and 4 ('\n\n') stay as plain regular tokens.


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


def search_best_triple(
    tokenizer,
    embedding_weights: torch.Tensor,
    topk: int = 200,
    verbose: bool = True,
) -> dict:
    """Brute-force `(i, j, k)` triple that minimises sum of L2 distances to the
    three special tokens, subject to re-tokenization keeping the 5-token shape.
    """
    eot_id, start_id, _assistant_id, end_id, _nl_id = TARGET_TOKEN_IDS

    cand_i, _ = find_most_similar_token_l2(eot_id, embedding_weights, topk)
    cand_j, _ = find_most_similar_token_l2(start_id, embedding_weights, topk)
    cand_k, _ = find_most_similar_token_l2(end_id, embedding_weights, topk)

    best_similarity = float("inf")
    best_triple = None
    best_decoded = None
    n_evaluated = 0
    n_kept = 0

    target_emb = {
        eot_id: embedding_weights[eot_id].to(torch.float32),
        start_id: embedding_weights[start_id].to(torch.float32),
        end_id: embedding_weights[end_id].to(torch.float32),
    }

    for i in cand_i:
        if i >= 128000:  # skip the special-token block in Llama-3.x
            continue
        ci = tokenizer.decode(i)
        for j in cand_j:
            if j >= 128000:
                continue
            cj = tokenizer.decode(j)
            for kk in cand_k:
                if kk >= 128000:
                    continue
                ck = tokenizer.decode(kk)
                n_evaluated += 1
                temp_prompt = ci + cj + "assistant" + ck + "\n\n"
                temp_ids = tokenizer(temp_prompt, add_special_tokens=False)[
                    "input_ids"
                ]
                if len(temp_ids) != 5:
                    continue
                n_kept += 1

                sim = 0.0
                for pos, target_id in zip(
                    (0, 1, 3),
                    (eot_id, start_id, end_id),
                ):
                    delta = (
                        target_emb[target_id]
                        - embedding_weights[temp_ids[pos]].to(torch.float32)
                    )
                    sim += float(torch.norm(delta))
                if sim < best_similarity:
                    best_similarity = sim
                    best_triple = [i, j, kk]
                    best_decoded = [ci, cj, ck]
                    if verbose:
                        print(
                            f"  new best: ids={best_triple} "
                            f"strs={[repr(s) for s in best_decoded]} "
                            f"L2_sum={sim:.4f}"
                        )

    if best_triple is None:
        raise RuntimeError(
            "No valid (i,j,k) triple found. Try larger --topk."
        )

    return {
        "target_token_ids": [eot_id, start_id, end_id],
        "target_token_strs": [
            tokenizer.convert_ids_to_tokens(eot_id),
            tokenizer.convert_ids_to_tokens(start_id),
            tokenizer.convert_ids_to_tokens(end_id),
        ],
        "best_triple_ids": best_triple,
        "best_triple_decoded": best_decoded,
        "best_similarity_l2_sum": best_similarity,
        "topk": topk,
        "n_evaluated": n_evaluated,
        "n_kept_5tok": n_kept,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", required=True,
        help="Local HF path to Llama-3.1-8B-Instruct.",
    )
    p.add_argument(
        "--output", default="./repro_mb_out/replacement.json",
        help="Where to write the best replacement JSON.",
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


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[embedding] loading tokenizer + model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    # We only need the input-embedding matrix, but loading the full causal LM
    # is the simplest reliable path; embedding extraction + del model below
    # keeps peak memory bounded.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
    )
    embedding_layer = model.get_input_embeddings()
    print(f"[embedding] embedding shape: {tuple(embedding_layer.weight.shape)}")
    embedding_weights = embedding_layer.weight.detach().cpu().clone()
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Sanity-check: confirm target IDs decode to the expected special strings.
    expected = {
        128009: "<|eot_id|>",
        128006: "<|start_header_id|>",
        128007: "<|end_header_id|>",
    }
    for tid, want in expected.items():
        got = tokenizer.convert_ids_to_tokens(tid)
        if got != want:
            raise RuntimeError(
                f"Special-token ID mismatch: id={tid} -> {got!r} "
                f"(expected {want!r}). Are you sure --model points to a "
                f"Llama-3.1/3.3 instruct tokenizer?"
            )

    print(f"[embedding] searching best (i,j,k) with topk={args.topk} ...")
    result = search_best_triple(
        tokenizer, embedding_weights, topk=args.topk, verbose=True,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[embedding] best triple ids     : {result['best_triple_ids']}")
    print(f"[embedding] best triple strings : {result['best_triple_decoded']}")
    print(f"[embedding] L2-sum              : {result['best_similarity_l2_sum']:.4f}")
    print(f"[embedding] saved to            : {out_path}")


if __name__ == "__main__":
    main()
