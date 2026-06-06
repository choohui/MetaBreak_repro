from __future__ import annotations

import numpy as np

from experiments_hc_4.config import ExpConfig
from experiments_hc_4.core import io
from experiments_hc_4.core.model import get_model


def _mock_replacement(lm) -> dict:
    return {
        "model_type": "llama",
        "assistant_header": lm.template.assistant_header,
        "target_token_strs": lm.template.target_token_strs,
        "target_token_ids": lm.template.target_token_ids,
        "best_triple_ids": [100489, 5809, 5810],
        "best_triple_decoded": ["ujících", "mock_start", "mock_end"],
        "neighbor_rank": 1,
        "note": "mock deterministic replacement for smoke runs",
    }


def _real_replacement(lm, cfg: ExpConfig) -> dict:
    import torch

    emb = lm.model.get_input_embeddings().weight.detach().float()
    special = set(int(x) for x in lm.template.special_token_ids)
    repl_ids = []
    repl_text = []
    per_target = []
    for tid in lm.template.target_token_ids:
        dist = torch.linalg.vector_norm(emb - emb[int(tid)], dim=1)
        if special:
            idx = torch.tensor(sorted(special), device=dist.device, dtype=torch.long)
            dist[idx] = float("inf")
        k = max(1, int(cfg.neighbor_rank))
        vals, ids = torch.topk(-dist, k=k, largest=True)
        rid = int(ids[-1].item())
        repl_ids.append(rid)
        text = lm.tokenizer.decode([rid])
        repl_text.append(text)
        per_target.append({
            "target_id": int(tid),
            "target_token": lm.tokenizer.convert_ids_to_tokens(int(tid)),
            "replacement_id": rid,
            "replacement_text": text,
            "l2_distance": float((-vals[-1]).item()),
        })
    return {
        "model_type": cfg.model_type,
        "assistant_header": lm.template.assistant_header,
        "target_token_strs": lm.template.target_token_strs,
        "target_token_ids": lm.template.target_token_ids,
        "best_triple_ids": repl_ids,
        "best_triple_decoded": repl_text,
        "neighbor_rank": cfg.neighbor_rank,
        "per_target": per_target,
    }


def run(cfg: ExpConfig, lm=None) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    lm = get_model(cfg, lm)
    repl = _mock_replacement(lm) if cfg.smoke else _real_replacement(lm, cfg)
    io.write_json(cfg.replacement_path, repl)
    print(f"[00] wrote replacement -> {cfg.replacement_path}")
    return repl

