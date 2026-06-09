"""The four defenses compared in choan.md §4.

  ours        — token-level diff-means detection + ±1 drop (sanitize)
  llama_guard — Llama-Guard-3 input moderation (refuse)
  jbshield    — JBShield-D toxic+jailbreak concept detection (refuse)
  guard_slm   — GUARD-SLM per-layer last-token SVM (refuse)

Each exposes ``name``, ``prepare(lm, calib) -> dict`` and
``guard(lm, prompt_text) -> GuardResult`` (see core.defense_base.Defense).
"""

from .ours import OursDefense
from .llama_guard import LlamaGuardDefense
from .jbshield import JBShieldDefense
from .guard_slm import GuardSLMDefense

REGISTRY = {
    "ours": OursDefense,
    "llama_guard": LlamaGuardDefense,
    "jbshield": JBShieldDefense,
    "guard_slm": GuardSLMDefense,
}


def build_defense(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown defense {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
