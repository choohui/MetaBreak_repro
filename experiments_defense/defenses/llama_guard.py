"""Llama-Guard — prompt-level input moderation (refuse).

Reuses ``src.evaluate.GuardJudge`` to load Meta's Llama-Guard-3-8B, but applies it
as an *input* filter: the user prompt alone is classified, and an ``unsafe``
verdict refuses the request before any victim generation. This is the
representative prompt-level (reject-the-whole-prompt) baseline; its weakness on
the GSM8k+header set is the point of the comparison.

The guard model is independent of the victim model, so ``guard_model`` is passed
separately. Under ``--smoke`` (or a mock victim) a deterministic stub stands in.
"""

from __future__ import annotations

import sys
from pathlib import Path

from core.defense_base import REFUSAL_TEXT, GuardResult
from core.model import LoadedModel

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_STUB_MARKERS = ("<|eot_id|>", "<|im_end|>", "<end_of_turn>", "Sure,")


class LlamaGuardDefense:
    name = "llama_guard"

    def __init__(self, guard_model: str | None = None, dtype: str = "bfloat16",
                 device: str | None = None, smoke: bool = False, **_kw):
        self.guard_model = guard_model
        self.dtype = dtype
        self.device = device
        self.smoke = smoke
        self.judge = None  # src.evaluate.GuardJudge

    def prepare(self, lm: LoadedModel, calib: dict) -> dict:
        use_stub = self.smoke or lm.is_mock or not self.guard_model
        if use_stub:
            self.judge = None
            return {"defense": self.name, "mode": "stub" if (self.smoke or lm.is_mock)
                    else "no_guard_model"}
        import torch
        from src.evaluate import GuardJudge
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.judge = GuardJudge(self.guard_model, device, self.dtype)
        return {"defense": self.name, "mode": "llama_guard_3", "guard_model": self.guard_model}

    # ------------------------------------------------------------------ #
    def _classify(self, prompt_text: str) -> bool:
        """True == unsafe (refuse)."""
        if self.judge is None:
            return any(m in prompt_text for m in _STUB_MARKERS)
        import torch
        chat = [{"role": "user", "content": prompt_text}]
        input_ids = self.judge.tokenizer.apply_chat_template(
            chat, return_tensors="pt").to(self.judge.device)
        in_len = int(input_ids.shape[1])
        with torch.no_grad():
            out = self.judge.model.generate(
                input_ids, max_new_tokens=20, do_sample=False,
                pad_token_id=self.judge.tokenizer.eos_token_id)
        gen = self.judge.tokenizer.decode(out[0, in_len:], skip_special_tokens=True).strip()
        first = gen.splitlines()[0].strip().lower() if gen else ""
        return first.startswith("unsafe")

    def guard(self, lm: LoadedModel, prompt_text: str) -> GuardResult:
        unsafe = self._classify(prompt_text)
        if unsafe:
            return GuardResult(action="refuse", flagged=True, score=1.0,
                               refusal_text=REFUSAL_TEXT, n_flagged=1)
        return GuardResult(action="pass", flagged=False, score=0.0)
