"""Generate MetaBreak responses with the L2 mimicry guard enabled."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from metabreak_l2_guard import (  # noqa: E402
    DEFAULT_REFUSAL,
    L2MimicryGuard,
    load_known_mimicry_spans,
)


def build_inputs(tokenizer, user_content: str, device: str) -> torch.Tensor:
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    )
    if not isinstance(out, torch.Tensor):
        out = out["input_ids"]
    return out.to(device)


@torch.no_grad()
def generate_once(
    model,
    tokenizer,
    user_content: str,
    *,
    max_new_tokens: int,
    device: str,
    temperature: float,
    do_sample: bool,
) -> dict:
    input_ids = build_inputs(tokenizer, user_content, device)
    in_len = int(input_ids.shape[1])
    t0 = time.time()
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    dt = time.time() - t0
    gen_ids = out[0, in_len:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {
        "text": text,
        "blocked_by_defense": False,
        "n_input_tokens": in_len,
        "n_generated_tokens": len(gen_ids),
        "seconds": dt,
    }


def blocked_response() -> dict:
    return {
        "text": DEFAULT_REFUSAL,
        "blocked_by_defense": True,
        "n_input_tokens": 0,
        "n_generated_tokens": 0,
        "seconds": 0.0,
    }


def defend_or_generate(
    *,
    guard: L2MimicryGuard,
    model,
    tokenizer,
    user_content: str,
    max_new_tokens: int,
    device: str,
    temperature: float,
    do_sample: bool,
) -> tuple[dict, dict]:
    decision = guard.inspect_text(user_content)
    if decision["blocked"]:
        return blocked_response(), decision
    return (
        generate_once(
            model,
            tokenizer,
            user_content,
            max_new_tokens=max_new_tokens,
            device=device,
            temperature=temperature,
            do_sample=do_sample,
        ),
        decision,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local Llama-3.x HF model path.")
    p.add_argument("--prompts", default=str(REPO_ROOT / "repro_mb_results" / "prompt_mimicked.jsonl"))
    p.add_argument("--output", default=str(HERE / "results" / "defended" / "responses.jsonl"))
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--also_baseline", action="store_true")
    p.add_argument("--neighbor_rank", type=int, default=256)
    p.add_argument("--threshold_margin", type=float, default=0.0)
    p.add_argument("--structural_min_spans", type=int, default=2)
    p.add_argument("--replacement", default=None)
    p.add_argument("--block_repeated_structure", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    print(f"[defended_attack] loading {args.model} ({args.dtype}) onto {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    known_mimicry_spans = load_known_mimicry_spans(
        tokenizer,
        Path(args.replacement) if args.replacement else None,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    guard = L2MimicryGuard.from_model(
        tokenizer,
        model,
        neighbor_rank=args.neighbor_rank,
        threshold_margin=args.threshold_margin,
        structural_min_spans=args.structural_min_spans,
        known_mimicry_spans=known_mimicry_spans,
        block_repeated_structure=args.block_repeated_structure,
    )
    print(f"[defended_attack] guard thresholds: {guard.thresholds.as_dict()}")

    do_sample = args.temperature > 0.0
    n_done = 0
    n_blocked_mim = 0
    n_blocked_base = 0

    with open(args.prompts, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            idx = rec["idx"]
            mimicked = rec["mimicked"]
            original = rec["original"]

            print(f"[defended_attack] [{idx}] mimicked -> guard/generate ...")
            mim_res, mim_decision = defend_or_generate(
                guard=guard,
                model=model,
                tokenizer=tokenizer,
                user_content=mimicked,
                max_new_tokens=args.max_new_tokens,
                device=device,
                temperature=args.temperature,
                do_sample=do_sample,
            )
            if mim_decision["blocked"]:
                n_blocked_mim += 1

            base_res = None
            base_decision = None
            if args.also_baseline:
                print(f"[defended_attack] [{idx}] baseline -> guard/generate ...")
                base_res, base_decision = defend_or_generate(
                    guard=guard,
                    model=model,
                    tokenizer=tokenizer,
                    user_content=original,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    temperature=args.temperature,
                    do_sample=do_sample,
                )
                if base_decision["blocked"]:
                    n_blocked_base += 1

            out_rec = {
                "idx": idx,
                "original_user_content": original,
                "mimicked_user_content": mimicked,
                "mimicked_response": mim_res,
                "baseline_response": base_res,
                "defense_mimicked": mim_decision,
                "defense_baseline": base_decision,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1

            print(
                f"[defended_attack] [{idx}] blocked="
                f"{mim_decision['blocked']} reason={mim_decision['reason']}"
            )

    print(
        f"[defended_attack] done. wrote {n_done} records to {out_path}; "
        f"blocked mimicked={n_blocked_mim}, baseline={n_blocked_base}"
    )


if __name__ == "__main__":
    main()
