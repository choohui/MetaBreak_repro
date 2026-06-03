"""Shared analysis helpers for the threshold stages (05 and 06).

Loads the stage-03 artifacts, builds the 5 measurement-signal matrices for an
arbitrary row subset, and evaluates single-threshold detection per signal/layer
with both a per-type breakdown and an ASR-restricted view.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core import io, metrics
from core.features import (
    ALL_SIGNALS, COS_SIGNAL, SCALAR_SIGNALS,
    cos_to_ref_matrix, ref_centroids_from, signal_matrix,
)
from core.labels import CAT_A, CAT_TO_LETTER, NEGATIVE_CATS, POSITIVE_CATS


def load_artifacts(out_dir: Path):
    rows = io.read_jsonl(out_dir / "tokens.jsonl")
    hidden = np.load(out_dir / "features.npz")["hidden"]
    asr_path = out_dir / "asr.jsonl"
    success: set[int] = set()
    if asr_path.exists():
        for r in io.read_jsonl(asr_path):
            if r.get("refusal_success") or r.get("guard_success"):
                success.add(int(r["sample_index"]))
    return rows, hidden, success


def reference_centroids(rows_sub: list[dict], hidden: np.ndarray) -> np.ndarray | None:
    """A (system-special) centroid per layer from the given rows, or None."""
    if not hidden.size:
        return None
    ridx = np.array([r["row_id"] for r in rows_sub])
    a_mask = np.array([r["category"] == CAT_A for r in rows_sub])
    return ref_centroids_from(hidden[ridx], a_mask)


def build_signals(rows_sub: list[dict], hidden: np.ndarray,
                  a_centroids: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """signal name -> [n, n_layers] matrix (incl. derived cos_to_ref).

    ``a_centroids`` lets a caller supply the A centroid computed from a larger
    pool (e.g. stage 06's reduced subset may lack A rows of its own).
    """
    sig: dict[str, np.ndarray] = {name: signal_matrix(rows_sub, name)
                                  for name in SCALAR_SIGNALS}
    if hidden.size:
        ridx = np.array([r["row_id"] for r in rows_sub])
        Hsub = hidden[ridx]
        centroids = a_centroids
        if centroids is None:
            a_mask = np.array([r["category"] == CAT_A for r in rows_sub])
            centroids = ref_centroids_from(Hsub, a_mask)
        sig[COS_SIGNAL] = cos_to_ref_matrix(Hsub, centroids)
    return sig


def binary_labels(rows_sub: list[dict], success: set[int] | None = None) -> np.ndarray:
    """1 = attack (B,D), 0 = benign (C,E,F,G), -1 = reference/excluded.

    If ``success`` is given (ASR view), an attack row counts as positive only
    when its prompt actually succeeded; failed-attack rows are dropped (-1).
    """
    y = []
    for r in rows_sub:
        c = r["category"]
        if c in POSITIVE_CATS:
            if success is not None and int(r["sample_index"]) not in success:
                y.append(-1)
            else:
                y.append(1)
        elif c in NEGATIVE_CATS:
            y.append(0)
        else:
            y.append(-1)
    return np.array(y)


def evaluate_signals(signals: dict[str, np.ndarray], y: np.ndarray) -> dict:
    """Per signal: per-layer binary_metrics + the best layer by AUC."""
    mask = y >= 0
    out: dict = {}
    for name in ALL_SIGNALS:
        if name not in signals:
            continue
        mat = signals[name]
        layers = []
        best_layer, best_auc = None, -1.0
        for l in range(mat.shape[1]):
            m = metrics.binary_metrics(mat[mask, l], y[mask])
            m["layer"] = l
            layers.append(m)
            auc = m.get("auc")
            if auc is not None and auc == auc and auc > best_auc:
                best_auc, best_layer = auc, l
        out[name] = {"per_layer": layers, "best_layer": best_layer, "best_auc": best_auc}
    return out


def per_type_breakdown(signals: dict[str, np.ndarray], rows_sub: list[dict],
                       evals: dict) -> list[dict]:
    """At each signal's best layer + Youden threshold, the flagged-rate of every
    category (TPR for B/D, FPR for C/E/F/G)."""
    letters = [r["letter"] for r in rows_sub]
    table: list[dict] = []
    for name, info in evals.items():
        l = info["best_layer"]
        if l is None:
            continue
        best = info["per_layer"][l]
        t = best.get("youden_threshold")
        if t is None:
            continue
        higher = best.get("direction") != "lower_is_attack"
        scores = signals[name][:, l]
        # binary_metrics negates the score when lower_is_attack, so its
        # youden_threshold is on the oriented score; reproduce that orientation.
        s = scores if higher else -scores
        pred = s >= t
        for letter in ["A", "B", "C", "D", "E", "F", "G"]:
            idx = [i for i, lt in enumerate(letters) if lt == letter]
            if not idx:
                continue
            flagged = float(np.mean(pred[idx]))
            role = "positive" if letter in ("B", "D") else (
                "reference" if letter == "A" else "negative")
            table.append({
                "signal": name, "layer": l, "letter": letter, "role": role,
                "n": len(idx), "flagged_rate": round(flagged, 5),
            })
    return table


def best_summary(evals: dict) -> list[dict]:
    return [{"signal": name, "best_layer": info["best_layer"],
             "best_auc": info["best_auc"]} for name, info in evals.items()]
