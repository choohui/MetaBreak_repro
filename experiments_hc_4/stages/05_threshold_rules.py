from __future__ import annotations

import numpy as np

from experiments_hc_4.config import ExpConfig
from experiments_hc_4.core import io
from experiments_hc_4.core.thresholds import select_threshold_rules


def run(cfg: ExpConfig, lm=None) -> dict:
    rows = io.read_jsonl(cfg.out_dir / "balanced_tokens.jsonl")
    data = np.load(cfg.out_dir / "scalar_values.npz", allow_pickle=True)
    x = data["x"].astype(float)
    names = [str(x) for x in data["feature_names"].tolist()]
    rules = select_threshold_rules(
        x,
        names,
        rows,
        cfg.fpr_targets,
        max_terms=cfg.scalar_rule_max_terms,
    )
    io.write_json(cfg.out_dir / "threshold_rules.json", rules)
    for fpr, rule in rules["selected"].items():
        ev = rule["val_eval"] if rule else None
        print(f"[05] fpr={fpr} selected={rule['kind'] if rule else None} val={ev}")
    return rules

