"""Per-token measurement-signal extraction (Main.md §2.2).

From one :class:`~core.capture.ForwardCapture` we read, for an analyzed token
position, the per-layer scalar signals:

    hidden_norm  : L2 norm of the hidden state        (len L+1)
    sink         : attention sink score (mean / heads) (len L)
    value_norm   : ||v_proj output||                  (len L)
    output_norm  : ||o_proj input||                   (len L)

The fifth signal, ``cos_to_ref`` (cosine of the hidden state to the A/system
centroid), needs the global A centroid and is computed downstream (stages
04/05) from the saved hidden cube. :func:`hidden_vector` returns that cube slice.
"""

from __future__ import annotations

import numpy as np
import torch

from .capture import ForwardCapture


class CaptureSignals:
    """Pre-computes the per-(layer, position) signal tensors for one capture."""

    def __init__(self, cap: ForwardCapture, sinks_mean: torch.Tensor):
        # hidden_norms[l] : [seq]  (l = 0..L, layer 0 = embedding)
        self.hidden_norms = [torch.linalg.vector_norm(h, dim=-1) for h in cap.hidden_states]
        self.sinks_mean = sinks_mean          # [L, seq]
        self.value_norms = cap.value_norms     # [L, seq]
        self.output_norms = cap.output_norms   # [L, seq]
        self.n_hidden_layers = len(cap.hidden_states)  # L+1
        self.n_attn_layers = sinks_mean.shape[0]       # L

    def signals_at(self, pos: int) -> dict:
        return {
            "hidden_norm": [round(float(hn[pos]), 5) for hn in self.hidden_norms],
            "sink": [round(float(self.sinks_mean[l, pos]), 8)
                     for l in range(self.n_attn_layers)],
            "value_norm": [round(float(self.value_norms[l, pos]), 5)
                           for l in range(self.n_attn_layers)],
            "output_norm": [round(float(self.output_norms[l, pos]), 5)
                            for l in range(self.n_attn_layers)],
        }


def hidden_vector(cap: ForwardCapture, pos: int) -> np.ndarray:
    """Stack the hidden states at ``pos`` over all layers -> [L+1, dim] float16."""
    return np.stack(
        [cap.hidden_states[l][pos].numpy().astype(np.float16)
         for l in range(len(cap.hidden_states))],
        axis=0,
    )


# --------------------------------------------------------------------------- #
# Analysis-side helpers (used by stages 05/06 on saved artifacts)
# --------------------------------------------------------------------------- #

# The four scalar measurement signals stored per-token in ``tokens.jsonl``.
SCALAR_SIGNALS = ["hidden_norm", "sink", "value_norm", "output_norm"]
# Plus the derived signal computed from the hidden cube + A centroid.
COS_SIGNAL = "cos_to_ref"
ALL_SIGNALS = SCALAR_SIGNALS + [COS_SIGNAL]


def signal_matrix(rows: list[dict], name: str) -> np.ndarray:
    """Stack a per-layer scalar signal across rows -> [n, n_layers]."""
    return np.array([r[name] for r in rows], dtype=np.float64)


def ref_centroids_from(hidden_sub: np.ndarray, a_mask: np.ndarray) -> np.ndarray:
    """Per-layer A (system-special) centroid from the masked rows -> [L+1, dim]."""
    if a_mask.sum() == 0:
        return np.zeros((hidden_sub.shape[1], hidden_sub.shape[2]), dtype=np.float64)
    return hidden_sub[a_mask].astype(np.float64).mean(axis=0)


def cos_to_ref_matrix(hidden_sub: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Cosine of each row's hidden state to the A centroid, per layer -> [n, L+1]."""
    n, n_layers, _ = hidden_sub.shape
    out = np.full((n, n_layers), np.nan)
    H = hidden_sub.astype(np.float64)
    for l in range(n_layers):
        c = centroids[l]
        nc = np.linalg.norm(c)
        if nc == 0:
            continue
        Hl = H[:, l, :]
        num = Hl @ c
        den = np.linalg.norm(Hl, axis=1) * nc
        nz = den > 0
        out[nz, l] = num[nz] / den[nz]
    return out
