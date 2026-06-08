from __future__ import annotations

import numpy as np

from experiments_hc_5.config import ExpConfig
from experiments_hc_5.core import io
from experiments_hc_5.core.labels import binary_label
from experiments_hc_5.core.metrics import oriented_auc
from experiments_hc_5.core.scalar_features import (
    add_hidden_projection_features,
    add_train_normalized_features,
    build_base_matrix,
)
from experiments_hc_5.core.thresholds import select_threshold_rules


def run(cfg: ExpConfig, lm=None) -> dict:
    rows = io.read_jsonl(cfg.out_dir / "balanced_tokens.jsonl")
    hidden = np.load(cfg.features_path, allow_pickle=True)["hidden"]
    base_x, base_names = build_base_matrix(rows, cfg.top_ks)
    y = np.asarray([binary_label(r["letter"]) for r in rows], dtype=int)
    split = np.asarray([r["split"] for r in rows])
    train = split == "train"
    benign_train = train & (y == 0)
    x, names, norm_params = add_train_normalized_features(base_x, base_names, train, benign_train)
    x, names, proto_params = add_hidden_projection_features(x, names, rows, hidden, train)
    row_ids = np.asarray([int(r["row_id"]) for r in rows], dtype=int)
    np.savez_compressed(
        cfg.out_dir / "scalar_values.npz",
        x=x.astype(np.float32),
        row_ids=row_ids,
        y=y,
        split=split.astype(object),
        feature_names=np.asarray(names, dtype=object),
    )

    val = (split == "val") & (y >= 0)
    train_bin = train & (y >= 0)
    records = []
    for j, name in enumerate(names):
        train_auc, train_dir = oriented_auc(x[train_bin, j], y[train_bin])
        val_auc, val_dir = oriented_auc(x[val, j], y[val])
        records.append({
            "feature": name,
            "train_auc": round(float(train_auc), 5) if not np.isnan(train_auc) else None,
            "train_direction": train_dir,
            "val_auc": round(float(val_auc), 5) if not np.isnan(val_auc) else None,
            "val_direction": val_dir,
        })
    records.sort(key=lambda r: (r["val_auc"] is not None, r["val_auc"] or -1), reverse=True)
    io.write_csv(cfg.out_dir / "scalar_discovery.csv", records)
    rules = select_threshold_rules(
        x,
        names,
        rows,
        cfg.fpr_targets,
        max_terms=cfg.scalar_rule_max_terms,
    )
    io.write_json(cfg.out_dir / "threshold_rules.json", rules)
    summary = {
        "stage": "04_fit_detectors",
        "n_rows": len(rows),
        "n_features": len(names),
        "detector_families": {
            "baseline": [
                "exact_special_token",
                "known_replacement_token",
                "l2_neighbor_proxy",
                "sink",
                "active_value",
            ],
            "main": ["proto_attack_minus_benign"],
            "secondary": ["cos_to_attack"],
            "upper_bound": ["threshold rule combinations; no logistic regression is selected by default"],
        },
        "top_val_features": records[:30],
        "selected_threshold_rules": rules["selected"],
        "normalizer": {"enabled": True, "base_feature_count": norm_params["base_feature_count"]},
        "prototype": proto_params,
    }
    io.write_json(cfg.out_dir / "scalar_discovery_summary.json", summary)
    print(f"[04] scalar features={len(names)} rows={len(rows)} thresholds={list(rules['selected'])}")
    return summary
