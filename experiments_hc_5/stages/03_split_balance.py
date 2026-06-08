from __future__ import annotations

from experiments_hc_5.config import ExpConfig
from experiments_hc_5.core import io
from experiments_hc_5.core.balance import assert_balanced, balanced_rows_by_split


def run(cfg: ExpConfig, lm=None) -> dict:
    rows = io.read_jsonl(cfg.tokens_path)
    balanced, manifest = balanced_rows_by_split(rows, seed=cfg.seed, ratios=cfg.split_ratios)
    assert_balanced(manifest)
    io.write_jsonl(cfg.out_dir / "balanced_tokens.jsonl", balanced)
    io.write_json(cfg.out_dir / "balanced_manifest.json", manifest)
    split_manifest = {
        "split_group_key": cfg.split_group_key,
        "split_ratios": list(cfg.split_ratios),
        "groups": {},
    }
    for r in balanced:
        split_manifest["groups"][str(r["sample_index"])] = r["split"]
    io.write_json(cfg.out_dir / "split_manifest.json", split_manifest)
    print(f"[03] balanced rows={len(balanced)}")
    return manifest
