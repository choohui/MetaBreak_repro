"""Unified defense interface + per-prompt evaluation driver.

Every defense — token-level ``ours`` (sanitize) and the three prompt-level
baselines (refuse) — exposes the same two methods, so one loop drives all four.
This is also the single seam the smoke test exercises with the mock model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from . import model as M

# Canned refusal returned when a prompt-level defense blocks the input.
REFUSAL_TEXT = ("I can't help with this request because it appears to attempt to "
                "override the chat-role structure or bypass safety protections.")


@dataclass
class GuardResult:
    action: str                          # "pass" | "refuse" | "sanitize"
    flagged: bool = False
    score: float | None = None
    sanitized_ids: list[int] | None = None   # set when action == "sanitize"
    refusal_text: str | None = None          # set when action == "refuse"
    n_flagged: int = 0                       # tokens flagged (token-level defenses)
    meta: dict[str, Any] = field(default_factory=dict)


class Defense(Protocol):
    name: str

    def prepare(self, lm: M.LoadedModel, calib: dict) -> dict:
        """Fit / load any per-model state from the TRAIN calibration data.
        Returns a small JSON-serialisable artifact summary."""
        ...

    def guard(self, lm: M.LoadedModel, prompt_text: str) -> GuardResult:
        """Decide pass / refuse / sanitize for one user prompt."""
        ...


def eval_one(lm: M.LoadedModel, defense: Defense, prompt_text: str,
             max_new_tokens: int = 256) -> dict:
    """Run the defense on one prompt and produce the defended response."""
    r = defense.guard(lm, prompt_text)
    if r.action == "refuse":
        response = r.refusal_text or REFUSAL_TEXT
    elif r.action == "sanitize" and r.sanitized_ids is not None:
        response = M.generate_from_ids(lm, r.sanitized_ids, max_new_tokens)
    else:
        response = M.generate(lm, prompt_text, max_new_tokens)
    return {
        "action": r.action,
        "flagged": bool(r.flagged),
        "n_flagged": int(r.n_flagged),
        "score": r.score,
        "response": response,
    }
