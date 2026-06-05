"""Stage 00 - build semantic-mimicry replacement.json."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG.parent
for p in (str(REPO_ROOT), str(PKG.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments_hc_4.config import ExpConfig, config_from_args, make_parser, require_model  # noqa: E402
from experiments_hc_4.core import io  # noqa: E402


def _smoke_replacement(cfg: ExpConfig) -> dict:
    return {
        "model_type": cfg.model_type,
        "auto_detected": False,
        "assistant_header": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "expected_n_tokens": 4,
        "target_token_ids": [1000, 1001, 1002],
        "target_token_strs": ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"],
        "fixed_positions": [2],
        "fixed_strs": ["assistant"],
        "replace_positions": [0, 1, 3],
        "best_triple_ids": [101, 102, 103],
        "best_triple_decoded": [" alpha", " beta", " gamma"],
        "best_similarity_l2_sum": 0.0,
        "topk": cfg.topk,
        "n_evaluated": 0,
        "n_kept_5tok": 0,
    }


def run(cfg: ExpConfig) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.skip_embedding and cfg.replacement.exists():
        print(f"[00] reusing {cfg.replacement}")
        return {"path": str(cfg.replacement), "reused": True}
    if cfg.smoke:
        repl = _smoke_replacement(cfg)
        io.write_json(cfg.replacement, repl)
        print(f"[00] smoke replacement -> {cfg.replacement}")
        return {"path": str(cfg.replacement), "smoke": True}

    require_model(cfg)
    from src.embedding import run as embedding_run

    embedding_run(Namespace(
        model_type=cfg.model_type,
        model=cfg.model,
        output=str(cfg.replacement),
        topk=cfg.topk,
        dtype=cfg.dtype,
        device=cfg.device,
    ))
    return {"path": str(cfg.replacement)}


def main() -> None:
    run(config_from_args(make_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()

