from __future__ import annotations

import numpy as np

from .labels import binary_label


def _arr(row: dict, key: str) -> np.ndarray:
    return np.asarray(row.get(key, []), dtype=np.float64)


def _add_matrix(cols: list[np.ndarray], names: list[str], prefix: str, mat: np.ndarray) -> None:
    cols.append(mat)
    names.extend([f"{prefix}_L{i}" for i in range(mat.shape[1])])


def _rank_pct(mat: np.ndarray, rows: list[dict]) -> np.ndarray:
    out = np.zeros_like(mat)
    groups: dict[tuple[int, int], list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault((int(r["sample_index"]), int(r["pos_offset"])), []).append(i)
    for idxs in groups.values():
        if len(idxs) <= 1:
            out[idxs, :] = 1.0
            continue
        for l in range(mat.shape[1]):
            vals = mat[idxs, l]
            order = np.argsort(-vals, kind="mergesort")
            ranks = np.empty(len(idxs), dtype=np.float64)
            ranks[order] = np.arange(len(idxs), dtype=np.float64)
            pct = 1.0 - ranks / max(1, len(idxs) - 1)
            for j, idx in enumerate(idxs):
                out[idx, l] = pct[j]
    return out


def _topk_flags(rank: np.ndarray, rows: list[dict], top_ks: list[int]) -> tuple[np.ndarray, list[str]]:
    sizes: dict[tuple[int, int], int] = {}
    for r in rows:
        key = (int(r["sample_index"]), int(r["pos_offset"]))
        sizes[key] = sizes.get(key, 0) + 1
    mats = []
    names = []
    for k in top_ks:
        flags = np.zeros_like(rank)
        for i, r in enumerate(rows):
            m = sizes[(int(r["sample_index"]), int(r["pos_offset"]))]
            min_pct = 1.0 - (min(k, m) - 1) / max(1, m - 1)
            flags[i, :] = (rank[i, :] >= min_pct).astype(float)
        mats.append(flags)
        names.extend([f"sink_top{k}_L{l}" for l in range(rank.shape[1])])
    return (np.concatenate(mats, axis=1), names) if mats else (np.zeros((len(rows), 0)), [])


def _summaries(prefix: str, mat: np.ndarray, cols: list[np.ndarray], names: list[str]) -> None:
    n = mat.shape[1]
    bands = {
        "early": (0, min(5, n)),
        "middle": (min(5, n), min(21, n)),
        "late": (min(21, n), n),
    }
    for band, (lo, hi) in bands.items():
        if hi > lo:
            cols.append(mat[:, lo:hi].mean(axis=1, keepdims=True))
            names.append(f"{prefix}_{band}_mean")
    cols.append(mat.max(axis=1, keepdims=True))
    names.append(f"{prefix}_max")
    cols.append((mat[:, -1:] - mat[:, :1]))
    names.append(f"{prefix}_last_minus_first")
    if n > 2:
        x = np.arange(n, dtype=np.float64)
        xc = x - x.mean()
        denom = (xc ** 2).sum() + 1e-12
        slope = ((mat - mat.mean(axis=1, keepdims=True)) @ xc / denom).reshape(-1, 1)
        cols.append(slope)
        names.append(f"{prefix}_layer_slope")


def build_base_matrix(rows: list[dict], top_ks: list[int]) -> tuple[np.ndarray, list[str]]:
    sink = np.vstack([_arr(r, "sink") for r in rows])
    value = np.vstack([_arr(r, "value_norm") for r in rows])
    output = np.vstack([_arr(r, "output_norm") for r in rows])
    active_value = np.vstack([_arr(r, "active_value") for r in rows])
    active_output = np.vstack([_arr(r, "active_output") for r in rows])
    hidden_norm = np.vstack([_arr(r, "hidden_norm") for r in rows])
    rank = _rank_pct(sink, rows)

    cols: list[np.ndarray] = []
    names: list[str] = []
    for prefix, mat in {
        "sink": sink,
        "sink_rank_pct": rank,
        "value_norm": value,
        "output_norm": output,
        "active_value": active_value,
        "active_output": active_output,
        "hidden_norm": hidden_norm,
    }.items():
        _add_matrix(cols, names, prefix, mat)

    topk, topk_names = _topk_flags(rank, rows, top_ks)
    if topk.shape[1]:
        cols.append(topk)
        names.extend(topk_names)

    for prefix, mat in {
        "sink": sink,
        "active_value": active_value,
        "active_output": active_output,
        "hidden_norm": hidden_norm,
    }.items():
        _summaries(prefix, mat, cols, names)

    x = np.concatenate(cols, axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), names


def add_train_normalized_features(x: np.ndarray, names: list[str], train_mask: np.ndarray,
                                  benign_train_mask: np.ndarray) -> tuple[np.ndarray, list[str], dict]:
    benign = x[benign_train_mask]
    med = np.median(benign, axis=0)
    mad = np.median(np.abs(benign - med), axis=0) + 1e-8
    rz = (x - med) / mad
    low_tail = np.zeros_like(x)
    high_tail = np.zeros_like(x)
    for j in range(x.shape[1]):
        vals = np.sort(benign[:, j])
        low_tail[:, j] = np.searchsorted(vals, x[:, j], side="right") / max(1, len(vals))
        high_tail[:, j] = 1.0 - np.searchsorted(vals, x[:, j], side="left") / max(1, len(vals))
    extra = np.concatenate([rz, low_tail, high_tail], axis=1)
    extra_names = [f"rz__{n}" for n in names] + [f"benign_lowtail__{n}" for n in names] + [
        f"benign_hightail__{n}" for n in names
    ]
    params = {"median": med.tolist(), "mad": mad.tolist(), "base_feature_count": len(names)}
    return np.concatenate([x, extra], axis=1), names + extra_names, params


def add_hidden_projection_features(x: np.ndarray, names: list[str], rows: list[dict], hidden: np.ndarray,
                                   train_mask: np.ndarray) -> tuple[np.ndarray, list[str], dict]:
    if hidden.size == 0:
        return x, names, {"enabled": False}
    row_ids = [int(r["row_id"]) for r in rows]
    h = hidden[row_ids].astype(np.float64)  # [N, L, D]
    y = np.asarray([binary_label(r["letter"]) for r in rows], dtype=int)
    tr = train_mask & (y >= 0)
    atk = tr & (y == 1)
    ben = tr & (y == 0)
    ref = train_mask & (np.asarray([r["letter"] == "A" for r in rows]))
    if atk.sum() == 0 or ben.sum() == 0:
        return x, names, {"enabled": False}
    feats = []
    feat_names = []
    for layer in range(h.shape[1]):
        ha = h[:, layer, :]
        mu_a = h[atk, layer, :].mean(axis=0)
        mu_b = h[ben, layer, :].mean(axis=0)
        direction = mu_a - mu_b
        dnorm = np.linalg.norm(direction) + 1e-12
        proj = (ha @ direction / dnorm).reshape(-1, 1)
        dist_b = np.linalg.norm(ha - mu_b, axis=1, keepdims=True)
        var_b = h[ben, layer, :].var(axis=0) + 1e-4
        maha_b = np.sqrt(((ha - mu_b) ** 2 / var_b).mean(axis=1, keepdims=True))
        feats.extend([proj, dist_b, maha_b])
        feat_names.extend([
            f"proto_attack_minus_benign_L{layer}",
            f"dist_benign_centroid_L{layer}",
            f"diag_mahal_benign_L{layer}",
        ])
        if ref.sum() > 0:
            mu_ref = h[ref, layer, :].mean(axis=0)
            dist_ref = np.linalg.norm(ha - mu_ref, axis=1, keepdims=True)
            feats.append(dist_ref)
            feat_names.append(f"dist_A_centroid_L{layer}")
    return np.concatenate([x] + feats, axis=1), names + feat_names, {"enabled": True}

