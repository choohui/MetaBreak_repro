"""Scalarizers — reduce an internal representation to ONE scalar per token, per
layer, WITHOUT a logistic-regression classifier (hc_4_claude paradigm).

A scalarizer maps the per-token internal state to a real-valued matrix
``[n_tokens, n_layers]``. A downstream threshold (see :mod:`core.thresholds`)
turns that scalar into a flag. Everything that needs a fitted direction /
centroid / covariance fits it on the **TRAIN rows only** (``train_mask``); the
same fitted geometry then scores every row, so the held-out numbers are honest.

Two families, kept clearly separated (the user's decision):

  * **clean**  — pure measurement or one-class OOD distances, no fitted 2-class
    boundary:  ``hidden_norm, value_norm, output_norm, sink, cos_to_ref,
    cos_to_attack, mahalanobis_benign, pca_resid, energy_lse, active_value,
    active_output``.  This set defines the headline claim.
  * **borderline** — a single fitted linear direction projected to 1-D, which
    edges toward a linear classifier:  ``diff_means, lda_1d, pca_sep_proj``.
    Reported in a separate block so they never silently enter the clean claim.

Layer spaces differ by signal: hidden-based scalars span L+1 layers (embedding +
each block); attention/value/output scalars span L layers. Each scalarizer just
returns its native ``[n, n_layers]`` and reports per-layer metrics in that space.

Memory note: ``compute`` fits per layer and discards the (dim x dim) factor
before the next layer, so peak memory is one covariance, not all L+1.
"""

from __future__ import annotations

import numpy as np

from .features import signal_matrix
from .labels import CAT_A
from .cascade import binary_labels

# Optional per-prompt normalisation wrappers (the anti-distribution-shift weapon
# against the hc_2 held-out threshold collapse).
NORMALIZE_MODES = ("none", "zscore", "rank", "robust")

# Shrinkage for covariance-based scalarizers (Ledoit-Wolf-style toward a scaled
# identity); keeps Cholesky / inverses finite when n_benign < dim or in smoke.
_SHRINK = 0.10
_JITTER = 1e-4


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _row_hidden(rows: list[dict], hidden: np.ndarray) -> np.ndarray | None:
    """Gather the hidden cube slice for ``rows`` -> [n, L+1, dim] float32."""
    if hidden is None or not getattr(hidden, "size", 0):
        return None
    ridx = np.array([r["row_id"] for r in rows], dtype=int)
    return hidden[ridx].astype(np.float32)


def _shrunk_cov(Xc: np.ndarray) -> np.ndarray:
    """Shrinkage covariance of centered rows ``Xc`` [m, d] -> [d, d]."""
    m, d = Xc.shape
    if m < 2:
        return np.eye(d)
    cov = (Xc.T @ Xc) / (m - 1)
    mu_trace = np.trace(cov) / d
    cov = (1.0 - _SHRINK) * cov + _SHRINK * mu_trace * np.eye(d)
    cov += _JITTER * np.eye(d)
    return cov


def _whiten_factor(cov: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor L with L L^T = cov (diag fallback)."""
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return np.diag(np.sqrt(np.clip(np.diag(cov), _JITTER, None)))


def _solve_lower(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve L z = B for z (B is [d, k]); plain numpy, no scipy."""
    try:
        return np.linalg.solve(L, B)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(L, B, rcond=None)[0]


def normalize_matrix(mat: np.ndarray, rows: list[dict], mode: str) -> np.ndarray:
    """Per-prompt normalisation of every column within each sample_index group.

    ``zscore`` -> (x-mean)/std ; ``rank`` -> ascending rank percentile in [0,1] ;
    ``robust`` -> (x-median)/IQR. ``none`` returns the matrix unchanged. A token's
    scalar becomes relative to its own prompt, removing global train->test scale
    drift (the mechanism that collapsed the hc_2 single-threshold cascade)."""
    if mode == "none" or mat.size == 0:
        return mat
    out = np.array(mat, dtype=np.float64, copy=True)
    groups: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(int(r["sample_index"]), []).append(i)
    for idx in groups.values():
        sub = out[idx]                       # [m, n_layers]
        if mode == "zscore":
            mu = np.nanmean(sub, axis=0)
            sd = np.nanstd(sub, axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            out[idx] = (sub - mu) / sd
        elif mode == "robust":
            med = np.nanmedian(sub, axis=0)
            q1 = np.nanpercentile(sub, 25, axis=0)
            q3 = np.nanpercentile(sub, 75, axis=0)
            iqr = np.where((q3 - q1) > 0, (q3 - q1), 1.0)
            out[idx] = (sub - med) / iqr
        elif mode == "rank":
            m = sub.shape[0]
            if m <= 1:
                out[idx] = np.full_like(sub, 0.5)
            else:
                order = np.argsort(np.argsort(sub, axis=0), axis=0)
                out[idx] = order / (m - 1.0)
    return out


# --------------------------------------------------------------------------- #
# scalarizer definitions
# --------------------------------------------------------------------------- #
class Scalarizer:
    key: str = ""
    needs_hidden: bool = False
    borderline: bool = False
    higher_note: str = "auto"          # binary_metrics auto-orients; documentation only

    def compute(self, rows, Hsub, train_mask, y, aux) -> np.ndarray:
        raise NotImplementedError


class _RawSignal(Scalarizer):
    def __init__(self, key):
        self.key = key

    def compute(self, rows, Hsub, train_mask, y, aux):
        return signal_matrix(rows, self.key)


class _ActiveProduct(Scalarizer):
    """sink x norm — hc_3's 'active' feature, no fit."""
    def __init__(self, key, norm_name):
        self.key = key
        self._norm = norm_name

    def compute(self, rows, Hsub, train_mask, y, aux):
        return signal_matrix(rows, "sink") * signal_matrix(rows, self._norm)


class _CosToCentroid(Scalarizer):
    """Cosine of the hidden state to a fitted centroid, per layer."""
    needs_hidden = True

    def __init__(self, key, which):
        self.key = key
        self._which = which           # "A" or "attack"

    def compute(self, rows, Hsub, train_mask, y, aux):
        n, n_layers, d = Hsub.shape
        if self._which == "A":
            sel = np.array([r["category"] == CAT_A for r in rows]) & train_mask
        else:
            sel = (y == 1) & train_mask
        out = np.full((n, n_layers), np.nan)
        if sel.sum() == 0:
            return out
        cents = Hsub[sel].mean(axis=0)            # [n_layers, d]
        aux.setdefault(self.key, cents.astype(np.float32))
        for l in range(n_layers):
            c = cents[l]
            nc = np.linalg.norm(c)
            if nc == 0:
                continue
            Hl = Hsub[:, l, :]
            den = np.linalg.norm(Hl, axis=1) * nc
            nz = den > 0
            out[nz, l] = (Hl[nz] @ c) / den[nz]
        return out


class _MahalanobisBenign(Scalarizer):
    """Distance to the TRAIN benign Gaussian (shrinkage-regularised), per layer."""
    needs_hidden = True
    key = "mahalanobis_benign"

    def compute(self, rows, Hsub, train_mask, y, aux):
        n, n_layers, d = Hsub.shape
        benign = (y == 0) & train_mask
        out = np.full((n, n_layers), np.nan)
        if benign.sum() < 2:
            return out
        for l in range(n_layers):
            Xb = Hsub[benign, l, :].astype(np.float64)
            mu = Xb.mean(axis=0)
            L = _whiten_factor(_shrunk_cov(Xb - mu))
            z = _solve_lower(L, (Hsub[:, l, :].astype(np.float64) - mu).T)  # [d, n]
            out[:, l] = np.sqrt(np.clip((z * z).sum(axis=0), 0, None))
        return out


class _EnergyLSE(Scalarizer):
    """logsumexp of whitened-coordinate magnitudes under the TRAIN benign
    Gaussian — an OOD energy dominated by the single largest deviation, so it
    ranks tokens differently from the L2 Mahalanobis distance."""
    needs_hidden = True
    key = "energy_lse"

    def compute(self, rows, Hsub, train_mask, y, aux):
        n, n_layers, d = Hsub.shape
        benign = (y == 0) & train_mask
        out = np.full((n, n_layers), np.nan)
        if benign.sum() < 2:
            return out
        for l in range(n_layers):
            Xb = Hsub[benign, l, :].astype(np.float64)
            mu = Xb.mean(axis=0)
            L = _whiten_factor(_shrunk_cov(Xb - mu))
            z = np.abs(_solve_lower(L, (Hsub[:, l, :].astype(np.float64) - mu).T))  # [d, n]
            zmax = z.max(axis=0)
            out[:, l] = zmax + np.log(np.exp(z - zmax).sum(axis=0))  # logsumexp over dims
        return out


class _PCAResid(Scalarizer):
    """Reconstruction residual off the TRAIN benign top-k PCA subspace."""
    needs_hidden = True
    key = "pca_resid"

    def __init__(self, k: int = 8):
        self._k = k

    def compute(self, rows, Hsub, train_mask, y, aux):
        n, n_layers, d = Hsub.shape
        benign = (y == 0) & train_mask
        out = np.full((n, n_layers), np.nan)
        if benign.sum() < 2:
            return out
        for l in range(n_layers):
            Xb = Hsub[benign, l, :].astype(np.float64)
            mu = Xb.mean(axis=0)
            Xc = Xb - mu
            k = int(min(self._k, Xc.shape[0] - 1, d))
            if k < 1:
                continue
            # PCA via SVD of the centered benign matrix (no dxd covariance).
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            V = Vt[:k]                              # [k, d]
            H = Hsub[:, l, :].astype(np.float64) - mu
            proj = H @ V.T                          # [n, k]
            recon = proj @ V                        # [n, d]
            out[:, l] = np.linalg.norm(H - recon, axis=1)
        return out


class _DiffMeans(Scalarizer):
    """Signed projection onto the unit mean-difference direction (attack-benign)."""
    needs_hidden = True
    borderline = True
    key = "diff_means"

    def compute(self, rows, Hsub, train_mask, y, aux):
        n, n_layers, d = Hsub.shape
        atk = (y == 1) & train_mask
        ben = (y == 0) & train_mask
        out = np.full((n, n_layers), np.nan)
        if atk.sum() == 0 or ben.sum() == 0:
            return out
        dirs = np.zeros((n_layers, d), dtype=np.float32)
        for l in range(n_layers):
            mu_a = Hsub[atk, l, :].astype(np.float64).mean(axis=0)
            mu_b = Hsub[ben, l, :].astype(np.float64).mean(axis=0)
            w = mu_a - mu_b
            nw = np.linalg.norm(w)
            if nw == 0:
                continue
            w = w / nw
            dirs[l] = w.astype(np.float32)
            mid = 0.5 * (mu_a + mu_b)
            out[:, l] = (Hsub[:, l, :].astype(np.float64) - mid) @ w
        aux.setdefault(self.key, dirs)
        return out


class _LDA1D(Scalarizer):
    """Fisher-LDA 1-D projection: w = Sw^{-1}(mu_a - mu_b), the most
    classifier-like scalarizer (hence borderline)."""
    needs_hidden = True
    borderline = True
    key = "lda_1d"

    def compute(self, rows, Hsub, train_mask, y, aux):
        n, n_layers, d = Hsub.shape
        atk = (y == 1) & train_mask
        ben = (y == 0) & train_mask
        out = np.full((n, n_layers), np.nan)
        if atk.sum() < 2 or ben.sum() < 2:
            return out
        dirs = np.zeros((n_layers, d), dtype=np.float32)
        for l in range(n_layers):
            Xa = Hsub[atk, l, :].astype(np.float64)
            Xb = Hsub[ben, l, :].astype(np.float64)
            mu_a, mu_b = Xa.mean(axis=0), Xb.mean(axis=0)
            Sw = _shrunk_cov(Xa - mu_a) * (len(Xa) - 1) + _shrunk_cov(Xb - mu_b) * (len(Xb) - 1)
            Sw = Sw / max(1, (len(Xa) + len(Xb) - 2))
            try:
                w = np.linalg.solve(Sw, mu_a - mu_b)
            except np.linalg.LinAlgError:
                w = (mu_a - mu_b)
            nw = np.linalg.norm(w)
            if nw == 0:
                continue
            w = w / nw
            dirs[l] = w.astype(np.float32)
            mid = 0.5 * (mu_a + mu_b)
            out[:, l] = (Hsub[:, l, :].astype(np.float64) - mid) @ w
        aux.setdefault(self.key, dirs)
        return out


class _PCASepProj(Scalarizer):
    """Projection onto the single TRAIN-PCA component that best separates the
    classes by |AUC| on train (component selection is train-only)."""
    needs_hidden = True
    borderline = True
    key = "pca_sep_proj"

    def __init__(self, k: int = 10):
        self._k = k

    def compute(self, rows, Hsub, train_mask, y, aux):
        from .metrics import roc_auc
        n, n_layers, d = Hsub.shape
        tr = train_mask & (y >= 0)
        out = np.full((n, n_layers), np.nan)
        if tr.sum() < 4:
            return out
        ytr = y[tr]
        if ytr.min() == ytr.max():
            return out
        dirs = np.zeros((n_layers, d), dtype=np.float32)
        for l in range(n_layers):
            Xtr = Hsub[tr, l, :].astype(np.float64)
            mu = Xtr.mean(axis=0)
            Xc = Xtr - mu
            k = int(min(self._k, Xc.shape[0] - 1, d))
            if k < 1:
                continue
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            V = Vt[:k]                               # [k, d]
            train_proj = Xc @ V.T                    # [n_tr, k]
            best_j, best_auc = 0, -1.0
            for j in range(k):
                a = roc_auc(train_proj[:, j], ytr)
                a = abs(a - 0.5)
                if a > best_auc:
                    best_auc, best_j = a, j
            w = V[best_j]
            dirs[l] = w.astype(np.float32)
            out[:, l] = (Hsub[:, l, :].astype(np.float64) - mu) @ w
        aux.setdefault(self.key, dirs)
        return out


# --------------------------------------------------------------------------- #
# registry + sets
# --------------------------------------------------------------------------- #
_SCALARIZER_LIST: list[Scalarizer] = [
    _RawSignal("hidden_norm"),
    _RawSignal("value_norm"),
    _RawSignal("output_norm"),
    _RawSignal("sink"),
    _ActiveProduct("active_value", "value_norm"),
    _ActiveProduct("active_output", "output_norm"),
    _CosToCentroid("cos_to_ref", "A"),
    _CosToCentroid("cos_to_attack", "attack"),
    _MahalanobisBenign(),
    _EnergyLSE(),
    _PCAResid(),
    _DiffMeans(),
    _LDA1D(),
    _PCASepProj(),
]
REGISTRY: dict[str, Scalarizer] = {s.key: s for s in _SCALARIZER_LIST}

CLEAN_SET = ["hidden_norm", "value_norm", "output_norm", "sink",
             "cos_to_ref", "cos_to_attack", "mahalanobis_benign",
             "pca_resid", "energy_lse", "active_value", "active_output"]
BORDERLINE_SET = ["diff_means", "lda_1d", "pca_sep_proj"]
SCALARIZER_SETS = {"clean": CLEAN_SET, "borderline": BORDERLINE_SET,
                   "all": CLEAN_SET + BORDERLINE_SET}


def is_borderline(key: str) -> bool:
    return bool(REGISTRY[key].borderline) if key in REGISTRY else False


def needs_hidden(key: str) -> bool:
    return bool(REGISTRY[key].needs_hidden) if key in REGISTRY else False


def resolve_keys(cfg, have_hidden: bool) -> list[str]:
    """The scalarizer keys to run, per ``cfg`` (explicit list or named set),
    dropping hidden-based ones when no hidden cube is available."""
    if getattr(cfg, "scalarizers", None):
        keys = [k for k in cfg.scalarizers if k in REGISTRY]
    else:
        keys = list(SCALARIZER_SETS.get(cfg.scalarizer_set, CLEAN_SET))
    if not have_hidden:
        keys = [k for k in keys if not REGISTRY[k].needs_hidden]
    return keys


def compute_scalars(cfg, rows: list[dict], hidden: np.ndarray,
                    train_mask: np.ndarray, keys=None, success=None):
    """Fit every scalarizer's geometry on the TRAIN rows and score ALL rows.

    Returns ``(mats, aux, y)`` where ``mats[key]`` is a normalised
    ``[n, n_layers]`` matrix, ``aux[key]`` holds light fitted directions/centroids
    (NOT covariances — those are recomputed, never stored), and ``y`` is the
    per-row defense label (1 attack / 0 benign / -1 reference). ``train_mask`` is
    a boolean array aligned to ``rows``."""
    have_hidden = bool(getattr(hidden, "size", 0))
    keys = keys or resolve_keys(cfg, have_hidden)
    y = binary_labels(rows, success)
    train_mask = np.asarray(train_mask, dtype=bool)
    Hsub = _row_hidden(rows, hidden) if have_hidden else None
    aux: dict[str, np.ndarray] = {}
    mats: dict[str, np.ndarray] = {}
    for k in keys:
        sc = REGISTRY[k]
        if sc.needs_hidden and Hsub is None:
            continue
        m = sc.compute(rows, Hsub, train_mask, y, aux)
        m = normalize_matrix(m, rows, cfg.normalize)
        mats[k] = m
    return mats, aux, y
