"""Per-prompt deployable guards built from the experiments_hwichan signals.

The token-level studies (`defense_thresholds.py`, `cascade_defense.py`) ask "are
attack *tokens* separable from benign *tokens*?". A deployed guard does not get to
label individual tokens — it sees a whole user message and must decide **block or
allow**. This module bridges that gap:

  1. ``extract_features`` runs one ``forward_capture`` per prompt and returns, for
     every *content-region* token (template prefix/suffix excluded), the per-layer
     scalar signals ``sink`` / ``hidden_norm`` / ``value_norm`` / ``output_norm`` and
     (optionally) ``cos_to_D``.

  2. A prompt is scored by **max-over-content-tokens** of an oriented feature at one
     layer — the natural per-prompt reduction of a per-token detector (block if *any*
     token looks attack-like). This is also what produces a *prompt-level* false-
     positive rate, which is strictly worse than the token-level FPR the sibling
     reports quote (a benign prompt is flagged if any one of its tokens trips), so the
     thresholds must be re-calibrated here rather than reused.

  3. ``InternalDetector`` / ``CascadeDetector`` wrap a calibrated (feature, layer,
     orientation, threshold) into a ``blocks(pf) -> bool`` decision. ``L2GuardDetector``
     wraps the input-side guard from ``experiments_yeonseok`` so all methods sit on the
     same ASR axis.

Calibration (``fit_feature`` / ``calibrate_*``) learns the best layer, orientation and
threshold from a *held-out calibration split* (benign + attack prompts) — the eval
split is never used to pick the operating point.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # repro_mb/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_hwichan.common import (  # noqa: E402
    CAT_SYSTEM,
    LoadedModel,
    forward_capture,
    label_token_categories,
    sink_scores,
    template_prefix_suffix_lengths,
)
from experiments_hwichan.defense_thresholds import roc_auc  # noqa: E402

SCALAR_FEATURES = ["sink", "hidden_norm", "value_norm", "output_norm"]
ALL_FEATURES = SCALAR_FEATURES + ["cos_to_D"]
# Features whose computation needs the attention tensor (slow / memory-heavy).
ATTENTION_FEATURES = {"sink"}


# --------------------------------------------------------------------------- #
# Per-prompt feature extraction
# --------------------------------------------------------------------------- #


@dataclass
class PromptFeatures:
    """Per-content-token feature matrices for a single prompt.

    ``feats[name]`` is ``[n_content_tokens, n_layers]`` where the layer axis follows
    the same convention as ``extract_representations.py``: ``hidden_norm`` / ``cos_to_D``
    have ``L+1`` layers (embedding + each decoder layer), the in-layer scalars
    (``sink`` / ``value_norm`` / ``output_norm``) have ``L``.
    """

    text: str
    seq_len: int
    content_lo: int
    content_hi: int
    feats: dict[str, np.ndarray]

    def n_content(self) -> int:
        return self.content_hi - self.content_lo

    def has(self, feature: str) -> bool:
        return feature in self.feats and self.feats[feature].shape[0] > 0


def _content_bounds(seq_len: int, prefix_len: int, suffix_len: int) -> tuple[int, int]:
    lo, hi = prefix_len, seq_len - suffix_len
    if hi <= lo:  # degenerate (very short prompt) — fall back to whole sequence
        return 0, seq_len
    return lo, hi


def collect_d_centroid(lm: LoadedModel, texts: list[str]) -> np.ndarray:
    """Average hidden state of genuine system-special (D) tokens, per layer.

    Used as the reference direction for the ``cos_to_D`` feature ("does this content
    token look, internally, like a real chat-template special token?"). Computed over
    a small benign calibration set so it never sees the eval prompts.
    Returns ``[L+1, dim]`` float32.
    """
    tpl = lm.template
    prefix_len, suffix_len = template_prefix_suffix_lengths(lm)
    acc: Optional[torch.Tensor] = None
    count = 0
    for text in texts:
        cap = forward_capture(lm, text)
        labels = label_token_categories(
            cap.input_ids, tpl, prefix_len, suffix_len, "ordinary"
        )
        d_pos = [p for p, c in labels.items() if c == CAT_SYSTEM]
        if not d_pos:
            continue
        H = torch.stack(cap.hidden_states, dim=0)  # [L+1, seq, dim]
        sel = H[:, d_pos, :].sum(dim=1)  # [L+1, dim]
        acc = sel if acc is None else acc + sel
        count += len(d_pos)
    if acc is None or count == 0:
        raise RuntimeError("No system-special (D) tokens found to build cos_to_D centroid.")
    return (acc / count).numpy().astype(np.float32)


def extract_features(
    lm: LoadedModel,
    text: str,
    *,
    prefix_len: int,
    suffix_len: int,
    d_centroid: Optional[np.ndarray] = None,
    features: tuple[str, ...] = tuple(ALL_FEATURES),
) -> PromptFeatures:
    """Forward one prompt and return per-content-token feature matrices."""
    cap = forward_capture(lm, text)
    seq = len(cap.input_ids)
    lo, hi = _content_bounds(seq, prefix_len, suffix_len)
    idx = list(range(lo, hi))
    feats: dict[str, np.ndarray] = {}

    H = torch.stack(cap.hidden_states, dim=0)  # [L+1, seq, dim]
    if "hidden_norm" in features:
        feats["hidden_norm"] = (
            torch.linalg.vector_norm(H[:, idx, :], dim=-1).T.numpy()
        )  # [n, L+1]
    if "value_norm" in features:
        feats["value_norm"] = cap.value_norms[:, idx].T.numpy()  # [n, L]
    if "output_norm" in features:
        feats["output_norm"] = cap.output_norms[:, idx].T.numpy()  # [n, L]
    if "sink" in features:
        sinks = sink_scores(cap)["mean_over_heads"]  # [L, seq]
        feats["sink"] = sinks[:, idx].T.numpy()  # [n, L]
    if "cos_to_D" in features and d_centroid is not None:
        dc = torch.from_numpy(np.ascontiguousarray(d_centroid)).float()  # [L+1, dim]
        Hc = H[:, idx, :].float()  # [L+1, n, dim]
        num = (Hc * dc.unsqueeze(1)).sum(dim=-1)  # [L+1, n]
        den = torch.linalg.vector_norm(Hc, dim=-1) * torch.linalg.vector_norm(
            dc, dim=-1
        ).unsqueeze(1)
        cos = torch.where(den > 0, num / den.clamp(min=1e-9), torch.zeros_like(num))
        feats["cos_to_D"] = cos.T.numpy()  # [n, L+1]

    return PromptFeatures(text, seq, lo, hi, feats)


# --------------------------------------------------------------------------- #
# Prompt-level scoring + threshold calibration
# --------------------------------------------------------------------------- #


def _prompt_score(pf: PromptFeatures, feature: str, layer: int, sign: int) -> float:
    """Oriented max-over-content-tokens score (higher == more attack-like).

    Returns ``-inf`` when the prompt has no usable token for this feature, so it can
    never be blocked by accident.
    """
    if not pf.has(feature):
        return float("-inf")
    col = pf.feats[feature][:, layer]
    col = col[~np.isnan(col)]
    if col.size == 0:
        return float("-inf")
    return float(np.max(sign * col))


def _scores(feat_list: list[PromptFeatures], feature: str, layer: int, sign: int) -> np.ndarray:
    return np.array(
        [_prompt_score(pf, feature, layer, sign) for pf in feat_list], dtype=np.float64
    )


def threshold_at_fpr(benign_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose benign block-rate is <= ``target_fpr`` (max recall)."""
    b = np.sort(benign_scores[np.isfinite(benign_scores)])
    if b.size == 0:
        return float("inf")
    for tau in b:
        if float((b >= tau).mean()) <= target_fpr:
            return float(tau)
    return float(b[-1] + abs(b[-1]) * 1e-6 + 1e-6)  # block nothing


def youden_threshold(attack_scores: np.ndarray, benign_scores: np.ndarray) -> float:
    a = attack_scores[np.isfinite(attack_scores)]
    b = benign_scores[np.isfinite(benign_scores)]
    cand = np.unique(np.concatenate([a, b])) if a.size and b.size else a
    best_t, best_j = float("inf"), -1.0
    for t in cand:
        tpr = float((a >= t).mean()) if a.size else 0.0
        fpr = float((b >= t).mean()) if b.size else 0.0
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, float(t)
    return best_t


@dataclass
class FeatureFit:
    """Best (layer, orientation) for one feature, with calibration score arrays."""

    feature: str
    layer: int
    sign: int
    auc: float
    attack_scores: np.ndarray  # oriented, on the calibration attack split
    benign_scores: np.ndarray  # oriented, on the calibration benign split


def fit_feature(
    attack_pf: list[PromptFeatures],
    benign_pf: list[PromptFeatures],
    feature: str,
) -> Optional[FeatureFit]:
    """Pick the layer+orientation that best separates attack from benign prompts."""
    have = [pf for pf in attack_pf + benign_pf if pf.has(feature)]
    if not have:
        return None
    n_layers = have[0].feats[feature].shape[1]
    y = np.concatenate([np.ones(len(attack_pf)), np.zeros(len(benign_pf))])
    best: Optional[FeatureFit] = None
    for layer in range(n_layers):
        for sign in (1, -1):
            a = _scores(attack_pf, feature, layer, sign)
            b = _scores(benign_pf, feature, layer, sign)
            scores = np.concatenate([a, b])
            valid = np.isfinite(scores)
            if valid.sum() < 2 or len(np.unique(y[valid])) < 2:
                continue
            auc = roc_auc(scores[valid], y[valid])
            if best is None or auc > best.auc:
                best = FeatureFit(feature, layer, sign, round(float(auc), 5), a, b)
    return best


# --------------------------------------------------------------------------- #
# Deployable detectors
# --------------------------------------------------------------------------- #


@dataclass
class InternalDetector:
    """Block iff max-over-content-tokens oriented feature >= threshold."""

    feature: str
    layer: int
    sign: int
    tau: float
    operating_point: str  # e.g. "fpr0.05" / "youden"
    auc: float

    @property
    def name(self) -> str:
        return f"internal:{self.feature}@L{self.layer}|{self.operating_point}"

    def score(self, pf: PromptFeatures) -> float:
        return _prompt_score(pf, self.feature, self.layer, self.sign)

    def blocks(self, pf: PromptFeatures) -> bool:
        return self.score(pf) >= self.tau


@dataclass
class CascadeDetector:
    """Sink-gate funnel: pass the cheap sink gate, then a hidden-state threshold.

    Stage 1 keeps a ``recall_target`` fraction of attacks (gate at the attack-score
    quantile) and drops easy benigns for free; stage 2 thresholds a stronger feature
    only on survivors, calibrated so the *whole-pool* benign block-rate stays under the
    FPR target.
    """

    s1_feature: str
    s1_layer: int
    s1_sign: int
    gate_tau: float
    recall_target: float
    s2_feature: str
    s2_layer: int
    s2_sign: int
    tau2: float
    operating_point: str
    s1_auc: float
    s2_auc: float

    @property
    def name(self) -> str:
        return (
            f"cascade:{self.s1_feature}->{self.s2_feature}@L{self.s2_layer}"
            f"|r{self.recall_target}|{self.operating_point}"
        )

    def passes_gate(self, pf: PromptFeatures) -> bool:
        return _prompt_score(pf, self.s1_feature, self.s1_layer, self.s1_sign) >= self.gate_tau

    def blocks(self, pf: PromptFeatures) -> bool:
        if not self.passes_gate(pf):
            return False
        return _prompt_score(pf, self.s2_feature, self.s2_layer, self.s2_sign) >= self.tau2


@dataclass
class L2GuardDetector:
    """Input-side L2 mimicry guard (experiments_yeonseok) on the same ASR axis."""

    guard: Any
    label: str = "input_l2_guard"

    @property
    def name(self) -> str:
        return self.label

    def blocks_text(self, text: str) -> bool:
        return bool(self.guard.inspect_text(text)["blocked"])


# --------------------------------------------------------------------------- #
# Building detectors at several operating points from a feature fit
# --------------------------------------------------------------------------- #


def build_internal_detectors(
    fit: FeatureFit,
    fpr_targets: tuple[float, ...],
    include_youden: bool = True,
) -> list[InternalDetector]:
    dets: list[InternalDetector] = []
    for target in fpr_targets:
        tau = threshold_at_fpr(fit.benign_scores, target)
        dets.append(
            InternalDetector(fit.feature, fit.layer, fit.sign, tau, f"fpr{target}", fit.auc)
        )
    if include_youden:
        tau = youden_threshold(fit.attack_scores, fit.benign_scores)
        dets.append(
            InternalDetector(fit.feature, fit.layer, fit.sign, tau, "youden", fit.auc)
        )
    return dets


def build_cascade_detectors(
    sink_fit: FeatureFit,
    stage2_fit: FeatureFit,
    attack_pf: list[PromptFeatures],
    benign_pf: list[PromptFeatures],
    recall_targets: tuple[float, ...],
    fpr_targets: tuple[float, ...],
) -> list[CascadeDetector]:
    """Calibrate sink-gate + stage-2 thresholds on the calibration split."""
    dets: list[CascadeDetector] = []
    s1_attack = _scores(attack_pf, sink_fit.feature, sink_fit.layer, sink_fit.sign)
    s1_benign = _scores(benign_pf, sink_fit.feature, sink_fit.layer, sink_fit.sign)
    s2_benign = _scores(benign_pf, stage2_fit.feature, stage2_fit.layer, stage2_fit.sign)

    for recall in recall_targets:
        finite = s1_attack[np.isfinite(s1_attack)]
        if finite.size == 0:
            continue
        gate_tau = float(np.quantile(finite, 1.0 - recall))
        passed_benign = s1_benign >= gate_tau  # benign prompts that survive stage 1
        n_benign = len(benign_pf)
        for target in fpr_targets:
            # threshold stage-2 so the *whole-pool* benign block-rate <= target.
            cand = s2_benign[passed_benign & np.isfinite(s2_benign)]
            tau2 = float("inf")
            if cand.size:
                for t in np.sort(cand):
                    blocked = passed_benign & np.isfinite(s2_benign) & (s2_benign >= t)
                    if n_benign and blocked.mean() <= target:
                        tau2 = float(t)
                        break
            dets.append(
                CascadeDetector(
                    sink_fit.feature, sink_fit.layer, sink_fit.sign, gate_tau, recall,
                    stage2_fit.feature, stage2_fit.layer, stage2_fit.sign, tau2,
                    f"fpr{target}", sink_fit.auc, stage2_fit.auc,
                )
            )
    return dets
