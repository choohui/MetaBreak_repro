"""MetaBreak TM-1 reproduction — Stage 3: send mimicked prompts to the LLM.

For each record in `prompt_mimicked.jsonl` we generate model responses for:

  * the *mimicked* prompt (regular-token replacements, our actual attack), and
  * (optional) the *original* prompt (special-token literals kept; the
    no-defense baseline equivalent to MetaBreak `local_test.py` chat path).

The prompt is wrapped via `tokenizer.apply_chat_template` with role `user`,
which mirrors what an online chat API (e.g. Ollama `/api/chat`) does and is
the threat model MetaBreak targets — the user-supplied content is what an
adversary controls. `apply_chat_template` is family-agnostic (HF maintains
chat templates for Llama/Qwen/Gemma/Phi out of the box), so no model-specific
branching is needed here.

Output:
  * `responses.jsonl` : one record per prompt with the two responses (where
    applicable) plus token counts and timing info.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .model_configs import known_model_types
except ImportError:  # standalone-script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_configs import known_model_types  # type: ignore[no-redef]


def build_inputs(tokenizer, user_content: str, device: str) -> torch.Tensor:
    """Wrap `user_content` as a single user turn and tokenize with the
    family's chat template. The user-content special-token literals
    (e.g. `<|eot_id|>` for Llama, `<|im_end|>` for Qwen) ARE parsed as
    special IDs by the tokenizer — this is by design and matches the
    threat model.
    """
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    )
    # Newer transformers may still return a BatchEncoding even with
    # return_dict=False; unwrap if so.
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
        # Llama-3 instruct's generation stops naturally at <|eot_id|> via
        # tokenizer.eos_token_id (which is set to <|eot_id|> in instruct
        # tokenizers).
    )
    dt = time.time() - t0
    gen_ids = out[0, in_len:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {
        "text": text,
        "n_input_tokens": in_len,
        "n_generated_tokens": len(gen_ids),
        "seconds": dt,
    }


def run(args: argparse.Namespace) -> None:
    """Run the attack stage given a pre-built Namespace (no sys.argv parsing).

    Designed to be called directly from ``run.py`` or other orchestrators.
    """
    device: str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prompts_path = Path(args.prompts) if getattr(args, "prompts", None) else _default_prompts(args.model_type)
    out_path     = Path(args.output) if getattr(args, "output", None) else _default_output(args.model_type)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    print(f"[attack] loading {args.model} ({args.dtype}) onto {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    do_sample = args.temperature > 0.0

    n_done = 0
    with open(prompts_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            idx = rec["idx"]
            mimicked = rec["mimicked"]
            original = rec["original"]

            print(f"[attack] [{idx}] mimicked -> generating ...")
            mim_res = generate_once(
                model, tokenizer, mimicked,
                max_new_tokens=args.max_new_tokens,
                device=device,
                temperature=args.temperature,
                do_sample=do_sample,
            )

            base_res = None
            if getattr(args, "also_baseline", False):
                print(f"[attack] [{idx}] baseline -> generating ...")
                base_res = generate_once(
                    model, tokenizer, original,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    temperature=args.temperature,
                    do_sample=do_sample,
                )

            out_rec = {
                "idx": idx,
                "model_type": getattr(args, "model_type", "unknown"),
                "original_user_content": original,
                "mimicked_user_content": mimicked,
                "mimicked_response": mim_res,
                "baseline_response": base_res,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1

            preview = mim_res["text"][:160].replace("\n", " ")
            print(f"[attack] [{idx}] mim out (160c): {preview!r}")

    print(f"[attack] done. wrote {n_done} records to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_type", required=True,
                   help="Family slug. Known: " + ", ".join(known_model_types())
                        + ". Used for output annotation and default paths.")
    p.add_argument("--model", required=True,
                   help="Local HF path to the victim model.")
    p.add_argument("--prompts", default=None,
                   help="Path to prompt_mimicked.jsonl. Default: "
                        "results/<model_type>/prompt_mimicked.jsonl.")
    p.add_argument("--output", default=None,
                   help="Path to responses.jsonl. Default: "
                        "results/<model_type>/responses.jsonl.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 -> greedy decoding (default).")
    p.add_argument("--device", default=None,
                   help='cuda / cpu (auto-detect if omitted).')
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--also_baseline", action="store_true",
                   help="Also generate for the un-mimicked original prompt "
                        "(literal special tokens kept) for comparison.")
    return p.parse_args()


def _default_prompts(model_type: str) -> Path:
    return Path(__file__).resolve().parent.parent / "results" / model_type / "prompt_mimicked.jsonl"


def _default_output(model_type: str) -> Path:
    return Path(__file__).resolve().parent.parent / "results" / model_type / "responses.jsonl"


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
