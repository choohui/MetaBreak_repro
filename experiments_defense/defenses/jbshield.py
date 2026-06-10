"""JBShield-D — toxic + jailbreak concept detection (refuse).

Reimplementation (from the USENIX'25 paper, NISPLab/JBShield) of the *detection*
half. JBShield's Linear-Representation view says a jailbreak succeeds when the
prompt activates BOTH a *toxic* concept (shared with plain harmful prompts) AND a
*jailbreak* concept (what flips refusal to compliance). We approximate each
concept by a mean-difference direction on pooled (last-token) hidden states:

  * toxic direction      = μ(harmful)   − μ(harmless)      [harmful = header-stripped attack question; harmless = benign]
  * jailbreak direction  = μ(jailbreak) − μ(plain-harmful) [jailbreak = full mimicry prompt]

A prompt is refused iff its projections onto BOTH directions exceed their
TRAIN-fitted thresholds. (JBShield-M mitigation steering is left as future work.)
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from core import capture, stats
from core.defense_base import REFUSAL_TEXT, GuardResult
from core.model import LoadedModel


def _pooled(lm: LoadedModel, texts: list[str], desc: str) -> np.ndarray:
    """[n, L+1, dim] last-token hidden per prompt."""
    rows = [
        capture.last_token(capture.capture_hidden(lm, t)[1])
        for t in tqdm(texts, desc=desc, unit="prompt", leave=False, dynamic_ncols=True)
    ]
    return np.stack(rows, axis=0).astype(np.float64)


def _fit_concept(Xpos: np.ndarray, Xneg: np.ndarray, target_fpr: float, desc: str) -> dict:
    """Per-layer unit mean-difference direction; pick best layer + low-FPR threshold."""
    n_layers = Xpos.shape[1]
    X = np.concatenate([Xpos, Xneg], axis=0)
    y = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))])
    proj = np.full((len(X), n_layers), np.nan)
    dirs = np.zeros((n_layers, Xpos.shape[2]))
    for l in tqdm(range(n_layers), desc=desc, unit="layer", leave=False, dynamic_ncols=True):
        w = Xpos[:, l, :].mean(0) - Xneg[:, l, :].mean(0)
        nw = np.linalg.norm(w)
        if nw == 0:
            continue
        w = w / nw
        dirs[l] = w
        proj[:, l] = X[:, l, :] @ w
    layer, auc = stats.best_layer(proj, y)
    if auc < 0.5:
        dirs[layer] = -dirs[layer]
        proj[:, layer] = -proj[:, layer]
        auc = 1.0 - auc
    thr = stats.threshold_fpr(proj[y == 0, layer], target_fpr)
    return {"layer": int(layer), "dir": dirs[layer].astype(np.float32),
            "threshold": float(thr), "auc": round(float(auc), 4)}


class JBShieldDefense:
    name = "jbshield"

    def __init__(self, target_fpr: float = 0.05, **_kw):
        self.target_fpr = float(target_fpr)
        self.toxic: dict | None = None
        self.jail: dict | None = None

    def prepare(self, lm: LoadedModel, calib: dict) -> dict:
        harmful = [r["bare_question"] for r in calib["attack_train"]]
        jailbreak = [r["text"] for r in calib["attack_train"]]
        harmless = [r["text"] for r in calib["benign_train"]]

        Xharm = _pooled(lm, harmful, "[jbshield] harmful calib")
        Xharmless = _pooled(lm, harmless, "[jbshield] harmless calib")
        Xjail = _pooled(lm, jailbreak, "[jbshield] jailbreak calib")

        self.toxic = _fit_concept(Xharm, Xharmless, self.target_fpr, "[jbshield] toxic layers")
        self.jail = _fit_concept(Xjail, Xharm, self.target_fpr, "[jbshield] jail layers")
        return {"defense": self.name,
                "toxic": {k: self.toxic[k] for k in ("layer", "threshold", "auc")},
                "jailbreak": {k: self.jail[k] for k in ("layer", "threshold", "auc")},
                "target_fpr": self.target_fpr}

    def _proj(self, lm: LoadedModel, prompt_text: str) -> tuple[float, float]:
        vec = capture.last_token(capture.capture_hidden(lm, prompt_text)[1]).astype(np.float64)
        t = float(vec[self.toxic["layer"]] @ self.toxic["dir"])
        j = float(vec[self.jail["layer"]] @ self.jail["dir"])
        return t, j

    def guard(self, lm: LoadedModel, prompt_text: str) -> GuardResult:
        assert self.toxic is not None and self.jail is not None, "call prepare() first"
        t, j = self._proj(lm, prompt_text)
        unsafe = (t >= self.toxic["threshold"]) and (j >= self.jail["threshold"])
        if unsafe:
            return GuardResult(action="refuse", flagged=True, score=min(t, j),
                               refusal_text=REFUSAL_TEXT, n_flagged=1,
                               meta={"toxic": t, "jailbreak": j})
        return GuardResult(action="pass", flagged=False, score=min(t, j),
                           meta={"toxic": t, "jailbreak": j})
