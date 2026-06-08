from __future__ import annotations

import re

import numpy as np

from experiments_hc_6.config import ExpConfig
from experiments_hc_6.core import io


def _auto_layers(cfg: ExpConfig, n_layers: int) -> list[int]:
    if cfg.steer_layers.strip().lower() != "auto":
        return sorted({
            int(x.strip().lstrip("L"))
            for x in cfg.steer_layers.split(",")
            if x.strip()
        })
    summary_path = cfg.out_dir / "scalar_discovery_summary.json"
    layers: list[int] = []
    if summary_path.exists():
        summary = io.read_json(summary_path)
        for rec in summary.get("top_val_features", []):
            m = re.search(r"_L(\d+)$", str(rec.get("feature", "")))
            if m:
                layer = int(m.group(1))
                if layer not in layers:
                    layers.append(layer)
            if len(layers) >= 5:
                break
    for layer in (6, 11, 12, 15):
        if layer < n_layers and layer not in layers:
            layers.append(layer)
    return sorted(x for x in layers if 0 <= x < n_layers)


def run(cfg: ExpConfig, lm=None) -> dict:
    rows = io.read_jsonl(cfg.out_dir / "balanced_tokens.jsonl")
    hidden = np.load(cfg.features_path, allow_pickle=True)["hidden"].astype(np.float64)
    row_ids = np.asarray([int(r["row_id"]) for r in rows], dtype=int)
    h = hidden[row_ids]
    train = np.asarray([r["split"] == "train" for r in rows])
    letters = np.asarray([r["letter"] for r in rows])
    attack = train & np.isin(letters, ["B", "D"])
    benign = train & np.isin(letters, ["C", "E", "F", "G"])
    if attack.sum() == 0 or benign.sum() == 0:
        raise RuntimeError("cannot fit steering vectors without B/D and C/E/F/G train rows")

    attack_mu = h[attack].mean(axis=0)
    benign_mu = h[benign].mean(axis=0)
    raw = benign_mu - attack_mu
    norms = np.linalg.norm(raw, axis=1) + 1e-12
    directions = raw / norms[:, None]
    gaps = []
    for layer in range(h.shape[1]):
        proj_attack = h[attack, layer, :] @ directions[layer]
        proj_benign = h[benign, layer, :] @ directions[layer]
        gap = float(np.median(proj_benign) - np.median(proj_attack))
        if not np.isfinite(gap) or gap <= 0:
            gap = float(norms[layer])
        gaps.append(gap)
    gaps_arr = np.asarray(gaps, dtype=np.float64)
    selected_layers = _auto_layers(cfg, h.shape[1])
    np.savez_compressed(
        cfg.out_dir / "steering_vectors.npz",
        directions=directions.astype(np.float32),
        benign_mu=benign_mu.astype(np.float32),
        attack_mu=attack_mu.astype(np.float32),
        gaps=gaps_arr.astype(np.float32),
        selected_layers=np.asarray(selected_layers, dtype=np.int64),
    )
    layer_rows = [
        {
            "layer": int(i),
            "selected": bool(i in selected_layers),
            "gap": round(float(gaps_arr[i]), 6),
            "direction_norm_before_normalize": round(float(norms[i]), 6),
        }
        for i in range(h.shape[1])
    ]
    io.write_csv(cfg.out_dir / "steering_layers.csv", layer_rows)
    report = {
        "stage": "09_fit_steering_vectors",
        "n_train_attack": int(attack.sum()),
        "n_train_benign": int(benign.sum()),
        "n_layers": int(h.shape[1]),
        "hidden_dim": int(h.shape[2]),
        "selected_layers": selected_layers,
        "modes": cfg.steer_modes,
        "alphas": cfg.steer_alphas,
        "formula": "direction_l=(mu_benign_l-mu_attack_l)/||mu_benign_l-mu_attack_l||; A excluded",
    }
    io.write_json(cfg.out_dir / "steering_fit.json", report)
    print(f"[09] selected_layers={selected_layers}")
    return report
