"""GUARD-SLM — per-layer last-token-activation SVM (refuse).

Reimplementation (from arXiv:2603.28817, solidlabnetwork/GUARD-SLM) of the
token-activation defence: extract the final-token hidden activation at each
layer, train a linear SVM (malicious vs benign) per layer on the TRAIN split,
keep the best-separating layer, and at inference refuse any prompt the SVM labels
malicious. GUARD-SLM's GPT-4o success judge is replaced by our local
refusal-keyword judge elsewhere in the harness.
"""

from __future__ import annotations

import numpy as np

from core import capture, stats
from core.defense_base import REFUSAL_TEXT, GuardResult
from core.model import LoadedModel


def _pooled(lm: LoadedModel, texts: list[str]) -> np.ndarray:
    return np.stack([capture.last_token(capture.capture_hidden(lm, t)[1]) for t in texts],
                    axis=0).astype(np.float64)


class GuardSLMDefense:
    name = "guard_slm"

    def __init__(self, C: float = 1.0, **_kw):
        self.C = float(C)
        self.layer: int | None = None
        self.svm = None
        self.scaler = None

    def prepare(self, lm: LoadedModel, calib: dict) -> dict:
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        mal = [r["text"] for r in calib["attack_train"]]
        ben = [r["text"] for r in calib["benign_train"]]
        Xmal, Xben = _pooled(lm, mal), _pooled(lm, ben)
        X = np.concatenate([Xmal, Xben], axis=0)
        y = np.concatenate([np.ones(len(Xmal)), np.zeros(len(Xben))]).astype(int)
        n_layers = X.shape[1]

        best = {"layer": 0, "auc": -1.0, "svm": None, "scaler": None}
        for l in range(n_layers):
            Xl = X[:, l, :]
            scaler = StandardScaler().fit(Xl)
            Xs = scaler.transform(Xl)
            svm = LinearSVC(C=self.C, max_iter=5000)
            try:
                svm.fit(Xs, y)
            except ValueError:
                continue
            auc = stats.roc_auc(svm.decision_function(Xs), y)
            if auc > best["auc"]:
                best = {"layer": l, "auc": auc, "svm": svm, "scaler": scaler}
        if best["svm"] is None:
            raise RuntimeError("guard_slm.prepare: no layer could be fit")
        self.layer, self.svm, self.scaler = best["layer"], best["svm"], best["scaler"]
        return {"defense": self.name, "layer": int(self.layer),
                "train_auc": round(float(best["auc"]), 4),
                "n_mal": int(len(Xmal)), "n_ben": int(len(Xben)), "C": self.C}

    def guard(self, lm: LoadedModel, prompt_text: str) -> GuardResult:
        assert self.svm is not None, "call prepare() first"
        vec = capture.last_token(capture.capture_hidden(lm, prompt_text)[1])[self.layer]
        Xs = self.scaler.transform(vec.reshape(1, -1).astype(np.float64))
        score = float(self.svm.decision_function(Xs)[0])
        if int(self.svm.predict(Xs)[0]) == 1:
            return GuardResult(action="refuse", flagged=True, score=score,
                               refusal_text=REFUSAL_TEXT, n_flagged=1)
        return GuardResult(action="pass", flagged=False, score=score)
