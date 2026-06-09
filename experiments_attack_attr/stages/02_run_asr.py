
"""Stage 02 — baseline ASR: run attack prompts through the model, label success.

The undefended attack-success rate the §3 defenses are measured against. Generates
responses for the attack-bearing variants (B malicious_mimicry, D malicious_special,
F positioned_regular) and judges success via refusal-keyword matching (and optional
Llama Guard). Feeds the §3 ASR-before number (which prompts actually jailbreak).

Outputs (under ``out_dir``):
    asr.jsonl        - one row per generated prompt
    asr.csv          - same, flat
    asr_summary.json - ASR per variant + judge config
"""

from __future__ import annotations

import sys
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # repro_mb (makes experiments_attack_attr importable)
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_attack_attr.config import ExpConfig, config_from_args, get_model, make_stage_parser  # noqa: E402
from experiments_attack_attr.core import io  # noqa: E402
from experiments_attack_attr.core.labels import ASR_VARIANTS  # noqa: E402

from src.evaluate import GuardJudge, evaluate_one  # noqa: E402


def _generate(lm, cfg: ExpConfig, text: str) -> dict:
    if getattr(lm, "is_mock", False):
        from experiments_attack_attr.core.mock import mock_generate
        return mock_generate(lm, text)
    from src.attack import generate_once
    return generate_once(
        lm.model, lm.tokenizer, text,
        max_new_tokens=cfg.max_new_tokens,
        device=lm.device,
        temperature=cfg.temperature,
        do_sample=cfg.temperature > 0.0,
    )


def run(cfg: ExpConfig, lm=None) -> dict:
    prompts = io.read_jsonl(cfg.out_dir / "prompts.jsonl")
    targets = [r for r in prompts if r["variant"] in ASR_VARIANTS]
    if cfg.limit:
        targets = targets[: cfg.limit]

    lm = get_model(cfg, lm)
    judge = None
    if cfg.asr_judge in ("guard", "both"):
        if getattr(lm, "is_mock", False):
            print("[02] asr_judge wants Llama-Guard but model is mock; keyword only.")
        elif not cfg.guard_model:
            print("[02] asr_judge wants Llama-Guard but --guard_model unset; keyword only.")
        else:
            judge = GuardJudge(cfg.guard_model, lm.device, cfg.dtype)

    rows: list[dict] = []
    for r in tqdm(targets, desc="[02] ASR"):
        gen = _generate(lm, cfg, r["text"])
        ev = evaluate_one(gen["text"], r["text"], judge)
        rows.append({
            "sample_index": r["sample_index"],
            "idx": r["idx"],
            "variant": r["variant"],
            "text": r["text"],
            "response_text": gen["text"],
            "n_generated_tokens": gen.get("n_generated_tokens"),
            "refusal_keyword_matched": ev["refusal_keyword_matched"],
            "refusal_success": ev["refusal_success"],
            "guard_success": ev["guard_success"],
        })

    io.write_jsonl(cfg.out_dir / "asr.jsonl", rows)
    io.write_csv(cfg.out_dir / "asr.csv", rows, columns=[
        "sample_index", "idx", "variant", "refusal_success", "guard_success",
        "n_generated_tokens", "response_text", "text",
    ])

    summary: dict = {"asr_judge_mode": cfg.asr_judge,
                     "guard_model_used": bool(judge is not None),
                     "per_variant": {}}
    for v in ASR_VARIANTS:
        vr = [r for r in rows if r["variant"] == v]
        if not vr:
            continue
        n = len(vr)
        ref = sum(1 for r in vr if r["refusal_success"])
        grd = sum(1 for r in vr if r["guard_success"])
        summary["per_variant"][v] = {
            "n": n,
            "asr_refusal_keyword": round(100.0 * ref / n, 2),
            "asr_llama_guard": round(100.0 * grd / n, 2) if judge else None,
        }
    io.write_json(cfg.out_dir / "asr_summary.json", summary)
    print(f"[02] generated {len(rows)} responses -> {cfg.out_dir/'asr.jsonl'}")
    print(f"[02] ASR summary: {summary['per_variant']}")
    return {"n_rows": len(rows), "summary": summary}


def main() -> None:
    args = make_stage_parser(__doc__).parse_args()
    run(config_from_args(args))


if __name__ == "__main__":
    main()
