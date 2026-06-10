"""OURS — token-level diff-means detection + ±1 drop (sanitize).

This is the choan.md method. Calibration fits the unit ``(μ_attack − μ_benign)``
direction (the ``diff_means`` scalarizer from experiments_hc_4_claude) on the
TRAIN hidden states of attack-slot tokens (mimicked / literal assistant-header
positions) vs benign content tokens, picks the most separating layer, and sets a
low-FPR threshold. At inference it projects every user-content token onto that
direction, flags those above threshold, drops each flagged token together with
its ±1 neighbours (the ``drop_token_pm1`` action from experiments_hc_4), and
re-generates from the sanitised ids — surgical removal instead of whole-prompt
refusal, so benign utility is preserved.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from core import capture, stats
from core.defense_base import GuardResult
from core.model import LoadedModel
from core.template import attack_slot_positions, template_prefix_suffix_lengths

_BENIGN_TOKENS_PER_PROMPT = 32


class OursDefense:
    name = "ours"

    def __init__(self, target_fpr: float = 0.01, **_kw):
        self.target_fpr = float(target_fpr)
        self.layer: int | None = None
        self.direction: np.ndarray | None = None     # unit vector [dim]
        self.threshold: float = 0.0

    # ------------------------------------------------------------------ #
    def _content_range(self, lm: LoadedModel, seq_len: int) -> tuple[int, int]:
        pre, suf = template_prefix_suffix_lengths(lm.tokenizer)
        return pre, max(pre, seq_len - suf)

    def prepare(self, lm: LoadedModel, calib: dict) -> dict:
        pos_vecs: list[np.ndarray] = []   # [L+1, dim] per attack-slot token
        neg_vecs: list[np.ndarray] = []
        for rec in tqdm(calib["attack_train"], desc="[ours] attack calib", unit="prompt",
                        leave=False, dynamic_ncols=True):
            ids, hid = capture.capture_hidden(lm, rec["text"])
            for p in sorted(attack_slot_positions(ids, lm.template)):
                if 0 <= p < hid.shape[1]:
                    pos_vecs.append(hid[:, p, :])
        for rec in tqdm(calib["benign_train"], desc="[ours] benign calib", unit="prompt",
                        leave=False, dynamic_ncols=True):
            ids, hid = capture.capture_hidden(lm, rec["text"])
            lo, hi = self._content_range(lm, hid.shape[1])
            idx = list(range(lo, hi))
            if len(idx) > _BENIGN_TOKENS_PER_PROMPT:
                idx = np.linspace(lo, hi - 1, _BENIGN_TOKENS_PER_PROMPT).astype(int).tolist()
            for p in idx:
                neg_vecs.append(hid[:, p, :])

        if not pos_vecs or not neg_vecs:
            raise RuntimeError("ours.prepare: need >=1 attack-slot and >=1 benign token "
                               f"(got {len(pos_vecs)} pos, {len(neg_vecs)} neg)")

        Hpos = np.stack(pos_vecs, axis=0).astype(np.float64)   # [Np, L+1, dim]
        Hneg = np.stack(neg_vecs, axis=0).astype(np.float64)   # [Nn, L+1, dim]
        n_layers = Hpos.shape[1]
        # per-layer unit diff-means direction + projection scores
        proj = np.full((Hpos.shape[0] + Hneg.shape[0], n_layers), np.nan)
        dirs = np.zeros((n_layers, Hpos.shape[2]), dtype=np.float64)
        H = np.concatenate([Hpos, Hneg], axis=0)
        y = np.concatenate([np.ones(Hpos.shape[0]), np.zeros(Hneg.shape[0])])
        for l in tqdm(range(n_layers), desc="[ours] fit layers", unit="layer",
                      leave=False, dynamic_ncols=True):
            w = Hpos[:, l, :].mean(0) - Hneg[:, l, :].mean(0)
            nw = np.linalg.norm(w)
            if nw == 0:
                continue
            w = w / nw
            dirs[l] = w
            proj[:, l] = H[:, l, :] @ w
        layer, auc = stats.best_layer(proj, y)
        if auc < 0.5:        # flip orientation so higher score = attack
            dirs[layer] = -dirs[layer]
            proj[:, layer] = -proj[:, layer]
            auc = 1.0 - auc
        self.layer = int(layer)
        self.direction = dirs[layer].astype(np.float32)
        neg_proj = proj[y == 0, layer]
        self.threshold = stats.threshold_fpr(neg_proj, self.target_fpr)
        return {"defense": self.name, "layer": self.layer, "train_auc": round(float(auc), 4),
                "threshold": round(float(self.threshold), 5),
                "n_pos": int(Hpos.shape[0]), "n_neg": int(Hneg.shape[0]),
                "target_fpr": self.target_fpr}

    # ------------------------------------------------------------------ #
    def guard(self, lm: LoadedModel, prompt_text: str) -> GuardResult:
        assert self.direction is not None and self.layer is not None, "call prepare() first"
        ids, hid = capture.capture_hidden(lm, prompt_text)
        seq = hid.shape[1]
        lo, hi = self._content_range(lm, seq)
        w = self.direction.astype(np.float64)
        scores = hid[self.layer, :, :] @ w        # [seq]
        flagged = {p for p in range(lo, hi) if scores[p] >= self.threshold}
        rm: set[int] = set()
        for p in flagged:
            rm.update({p - 1, p, p + 1})
        rm = {p for p in rm if lo <= p < hi}      # never touch the chat wrapper
        sanitized = [int(t) for i, t in enumerate(ids) if i not in rm]
        max_score = float(scores[lo:hi].max()) if hi > lo else float("nan")
        return GuardResult(action="sanitize", flagged=bool(flagged),
                           score=max_score, sanitized_ids=sanitized,
                           n_flagged=len(flagged),
                           meta={"n_dropped": len(rm)})
