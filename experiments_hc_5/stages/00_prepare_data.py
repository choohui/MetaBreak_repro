from __future__ import annotations

import numpy as np
import shutil

from experiments_hc_5.config import DATA_DIR, REPO_ROOT, ExpConfig
from experiments_hc_5.core import io
from experiments_hc_5.core.model import get_model


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


def _ensure_data_files(cfg: ExpConfig) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        cfg.tm1_path: [REPO_ROOT / "Q_TM-1_Llama.txt", REPO_ROOT / "prompts" / "Q_TM-1_Llama.txt"],
        cfg.q_path: [REPO_ROOT / "prompts" / "Q.txt"],
    }
    copied = []
    missing = []
    for dst, candidates in sources.items():
        if not dst.exists():
            src = next((p for p in candidates if p.exists()), None)
            if src is not None:
                shutil.copy2(src, dst)
                copied.append({"from": str(src), "to": str(dst)})
        if not dst.exists():
            missing.append(str(dst))
    for required in [cfg.positioned_words_path, cfg.benign_mimicry_path, cfg.benign_special_path]:
        if not required.exists():
            missing.append(str(required))
    if missing:
        raise FileNotFoundError("missing hc5 data inputs: " + ", ".join(missing))
    return {
        "data_dir": str(DATA_DIR),
        "copied": copied,
        "inputs": {
            "tm1": str(cfg.tm1_path),
            "q": str(cfg.q_path),
            "positioned_words": str(cfg.positioned_words_path),
            "benign_mimicry": str(cfg.benign_mimicry_path),
            "benign_special": str(cfg.benign_special_path),
        },
    }


def run(cfg: ExpConfig, lm=None) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    prep = _ensure_data_files(cfg)
    lm = get_model(cfg, lm)
    repl = _mock_replacement(lm) if cfg.smoke else _real_replacement(lm, cfg)
    io.write_json(cfg.replacement_path, repl)
    prep["replacement"] = repl
    io.write_json(cfg.out_dir / "prepare_data_summary.json", prep)
    print(f"[00] wrote replacement -> {cfg.replacement_path}")
    return prep
