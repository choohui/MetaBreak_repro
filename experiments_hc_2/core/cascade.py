"""Two-stage cascade-detector primitives (Main.md §3 / §4).

The defense hc_2 implements and evaluates:

  * **Stage 1 — sink gate (cheap):** per prompt, keep only the top ``keep_pct``%
    of tokens by max-over-layers sink score; drop the low-sink body. This is the
    §3 "narrow the token range first" idea turned into a runnable gate.
  * **Stage 2 — signal threshold:** on the gate survivors only, flag a token when
    one measurement signal at a chosen layer crosses a threshold fixed at a
    documented operating point (a target benign FPR).

Everything here is pure (numpy + the saved per-token signals) so the cascade is
evaluated model-free from stage-03 artifacts. Decoupled from the ``stages``
package: it depends only on ``core``.
"""

from __future__ import annotations

import math

import numpy as np

from . import metrics
from .labels import NEGATIVE_CATS, POSITIVE_CATS

# Attack-slot categories whose flagging blocks an attack prompt.
_ATTACK_LETTERS = ("B", "D")


# --------------------------------------------------------------------------- #
# Stage 1 — sink gate
# --------------------------------------------------------------------------- #
def per_prompt_by_sink(rows: list[dict]) -> dict:
    """Group rows by prompt; each list sorted by max-over-layers sink (desc)."""
    by_sample: dict[int, list[dict]] = {}
    for r in rows:
        by_sample.setdefault(r["sample_index"], []).append(r)
    for rs in by_sample.values():
        rs.sort(key=lambda r: max(r["sink"]), reverse=True)
    return by_sample


def sink_gate(rows: list[dict], keep_pct: float) -> list[dict]:
    """1st-stage gate: per prompt keep the top ``keep_pct``% of tokens by
    max-over-layers sink (at least one). ``keep_pct >= 100`` keeps everything."""
    if keep_pct >= 100:
        return list(rows)
    kept: list[dict] = []
    for rs in per_prompt_by_sink(rows).values():
        k = max(1, math.ceil(len(rs) * keep_pct / 100.0))
        kept.extend(rs[:k])
    return kept


# --------------------------------------------------------------------------- #
# Labels + Stage 2 threshold
# --------------------------------------------------------------------------- #
def binary_labels(rows: list[dict], success: set[int] | None = None) -> np.ndarray:
    """1 = attack (B,D), 0 = benign (C,E,F,G), -1 = reference/excluded.

    With ``success`` (ASR view) an attack row is positive only if its prompt
    actually succeeded; other attack rows are dropped (-1)."""
    y = []
    for r in rows:
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


def threshold_at_fpr(col: np.ndarray, y: np.ndarray, fpr_target: float) -> dict:
    """Fit the 2nd-stage threshold on ``col`` at a target benign FPR.

    Returns the oriented threshold + direction + AUC (None threshold if the data
    is degenerate). ``binary_metrics`` auto-orients, so callers must orient the
    score the same way before comparing (see :func:`predict`)."""
    mask = y >= 0
    m = metrics.binary_metrics(col[mask], y[mask], fpr_targets=(fpr_target,))
    key = f"{int(fpr_target * 100)}pct"
    thr = (m.get("threshold_at_fpr") or {}).get(key)
    if thr is None:                      # no threshold meets the FPR budget -> Youden
        thr = m.get("youden_threshold")
    return {"threshold": thr, "direction": m.get("direction"),
            "auc": m.get("auc"), "fpr_target": fpr_target}


def predict(col: np.ndarray, threshold: float | None, direction: str | None) -> np.ndarray:
    """Boolean attack predictions for a signal column at a fixed threshold."""
    if threshold is None:
        return np.zeros(len(col), dtype=bool)
    s = col if direction != "lower_is_attack" else -col
    return s >= threshold


# --------------------------------------------------------------------------- #
# Aggregation — per-type rates + prompt-level block / ASR
# --------------------------------------------------------------------------- #
def per_type_rates(rows: list[dict], pred: np.ndarray) -> dict:
    """Per-letter flagged rate: TPR for B/D, FPR for C/E/F/G (and A reference)."""
    cnt: dict[str, int] = {}
    flag: dict[str, int] = {}
    for r, p in zip(rows, pred):
        L = r["letter"]
        cnt[L] = cnt.get(L, 0) + 1
        flag[L] = flag.get(L, 0) + int(bool(p))
    out = {}
    for L in sorted(cnt):
        role = "attack" if L in _ATTACK_LETTERS else ("reference" if L == "A" else "benign")
        out[L] = {"role": role, "n": cnt[L], "flagged": flag[L],
                  "rate": round(flag[L] / cnt[L], 5) if cnt[L] else None}
    return out


def prompt_block_and_asr(rows: list[dict], pred: np.ndarray, success: set[int]) -> dict:
    """Prompt-level block decision + ASR before/after.

    An attack prompt is *blocked* if any of its attack-slot (B/D) tokens is
    flagged. ASR-before = succeeded / attack-prompts; ASR-after additionally
    requires the prompt to have evaded the block — so it can only drop."""
    blocked: dict[int, bool] = {}
    attack_samples: set[int] = set()
    for r, p in zip(rows, pred):
        if r["letter"] in _ATTACK_LETTERS:
            s = int(r["sample_index"])
            attack_samples.add(s)
            if bool(p):
                blocked[s] = True
    n_attack = len(attack_samples)
    n_blocked = sum(1 for s in attack_samples if blocked.get(s))
    succeeded = attack_samples & set(success)
    n_succ = len(succeeded)
    still = sum(1 for s in succeeded if not blocked.get(s))
    return {
        "n_attack_prompts": n_attack,
        "n_blocked_prompts": n_blocked,
        "block_rate_prompt": round(n_blocked / n_attack, 5) if n_attack else None,
        "n_succeeded": n_succ,
        "asr_before": round(n_succ / n_attack, 5) if n_attack else None,
        "asr_after": round(still / n_attack, 5) if n_attack else None,
        # honest efficacy: of the attacks that WOULD have succeeded, how many the
        # defense actually blocked (ASR-after<=before is otherwise trivially true).
        "block_rate_among_successful":
            round((n_succ - still) / n_succ, 5) if n_succ else None,
    }
