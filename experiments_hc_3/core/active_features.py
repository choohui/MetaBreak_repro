"""Active SinkProbe feature construction.

The current hc_2 artifacts store per-token, mean-over-head sink scores rather
than full per-head attention maps. hc_3 therefore builds a "labeled-token"
SinkProbe: rank/order features are computed among the analyzed tokens within
the same prompt and `pos_offset`. If a future extractor stores all-token or
per-head sinks, this module is the one place to extend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .labels import NEGATIVE_CATS, POSITIVE_CATS

SCALAR_BASES = ("sink", "value_norm", "output_norm", "hidden_norm")
ACTIVE_BASES = ("active_value", "active_output")


@dataclass
class FeatureSet:
    rows: list[dict]
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list[str]


def defense_labels(rows: list[dict]) -> np.ndarray:
    y = []
    for r in rows:
        cat = r["category"]
        if cat in POSITIVE_CATS:
            y.append(1)
        elif cat in NEGATIVE_CATS:
            y.append(0)
        else:
            y.append(-1)
    return np.array(y, dtype=int)


def _safe_array(row: dict, key: str) -> np.ndarray:
    return np.asarray(row.get(key, []), dtype=np.float64)


def _n_attn_layers(rows: list[dict]) -> int:
    if not rows:
        return 0
    return len(rows[0].get("sink", []))


def _n_hidden_layers(rows: list[dict]) -> int:
    if not rows:
        return 0
    return len(rows[0].get("hidden_norm", []))


def _rank_percentiles(rows: list[dict], key: str, n_layers: int) -> np.ndarray:
    """Descending rank percentile within each prompt/pos_offset/layer.

    1.0 means highest score in that prompt group; 0.0 means lowest. Reference A
    rows stay in the ranking pool because template sinks are a useful baseline,
    but binary training later excludes A with y=-1.
    """
    out = np.zeros((len(rows), n_layers), dtype=np.float64)
    by_group: dict[tuple[int, int], list[int]] = {}
    for i, r in enumerate(rows):
        by_group.setdefault((int(r["sample_index"]), int(r["pos_offset"])), []).append(i)
    for idxs in by_group.values():
        m = len(idxs)
        if m <= 1:
            out[idxs, :] = 1.0
            continue
        for l in range(n_layers):
            vals = np.array([_safe_array(rows[i], key)[l] for i in idxs], dtype=np.float64)
            order = np.argsort(-vals, kind="mergesort")
            ranks = np.empty(m, dtype=np.float64)
            ranks[order] = np.arange(m, dtype=np.float64)
            pct = 1.0 - ranks / (m - 1)
            for j, i in enumerate(idxs):
                out[i, l] = pct[j]
    return out


def _topk_flags(rank_pct: np.ndarray, rows: list[dict], top_ks: list[int]) -> tuple[np.ndarray, list[str]]:
    by_group_size: dict[tuple[int, int], int] = {}
    for r in rows:
        key = (int(r["sample_index"]), int(r["pos_offset"]))
        by_group_size[key] = by_group_size.get(key, 0) + 1

    cols = []
    names = []
    # Convert top-k to a rank percentile threshold per row group.
    for k in top_ks:
        flags = np.zeros_like(rank_pct)
        for i, r in enumerate(rows):
            m = by_group_size[(int(r["sample_index"]), int(r["pos_offset"]))]
            if m <= 1:
                flags[i, :] = 1.0
                continue
            min_pct = 1.0 - (min(k, m) - 1) / (m - 1)
            flags[i, :] = (rank_pct[i, :] >= min_pct).astype(float)
        cols.append(flags)
        names.extend([f"sink_top{k}_L{l}" for l in range(rank_pct.shape[1])])
    if not cols:
        return np.zeros((len(rows), 0)), []
    return np.concatenate(cols, axis=1), names


def build_feature_set(rows: list[dict], pos_offset: int, top_ks: list[int]) -> FeatureSet:
    rows = [r for r in rows if int(r["pos_offset"]) == int(pos_offset)]
    n_attn = _n_attn_layers(rows)
    n_hidden = _n_hidden_layers(rows)
    if not rows or n_attn == 0:
        return FeatureSet(rows, np.zeros((0, 0)), np.zeros(0, dtype=int),
                          np.zeros(0, dtype=int), [])

    sink = np.vstack([_safe_array(r, "sink") for r in rows])
    value = np.vstack([_safe_array(r, "value_norm") for r in rows])
    output = np.vstack([_safe_array(r, "output_norm") for r in rows])
    hidden = np.vstack([_safe_array(r, "hidden_norm") for r in rows])
    if hidden.shape[1] != n_hidden:
        hidden = hidden[:, :n_hidden]

    active_value = sink * value
    active_output = sink * output
    rank_sink = _rank_percentiles(rows, "sink", n_attn)

    matrices: list[np.ndarray] = []
    names: list[str] = []

    def add_matrix(prefix: str, mat: np.ndarray) -> None:
        matrices.append(mat)
        names.extend([f"{prefix}_L{l}" for l in range(mat.shape[1])])

    add_matrix("sink", sink)
    add_matrix("sink_rank_pct", rank_sink)
    add_matrix("value_norm", value)
    add_matrix("output_norm", output)
    add_matrix("active_value", active_value)
    add_matrix("active_output", active_output)
    if n_hidden:
        add_matrix("hidden_norm", hidden)

    topk, topk_names = _topk_flags(rank_sink, rows, top_ks)
    if topk.shape[1]:
        matrices.append(topk)
        names.extend(topk_names)

    # Compact trajectory summaries. Layer bands are clipped for small smoke runs.
    bands = {
        "early": (0, min(5, n_attn)),
        "middle": (min(5, n_attn), min(21, n_attn)),
        "late": (min(21, n_attn), n_attn),
    }
    for base_name, mat in {
        "sink": sink,
        "active_value": active_value,
        "active_output": active_output,
    }.items():
        for band, (lo, hi) in bands.items():
            if hi > lo:
                matrices.append(mat[:, lo:hi].mean(axis=1, keepdims=True))
                names.append(f"{base_name}_{band}_mean")
        matrices.append(mat.max(axis=1, keepdims=True))
        names.append(f"{base_name}_max")
        if n_attn > 1:
            matrices.append((mat[:, -1:] - mat[:, :1]))
            names.append(f"{base_name}_last_minus_first")

    x = np.concatenate(matrices, axis=1).astype(np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = defense_labels(rows)
    groups = np.array([int(r["sample_index"]) for r in rows], dtype=int)
    return FeatureSet(rows, x, y, groups, names)

