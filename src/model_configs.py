"""Per-family chat-template config for MetaBreak TM-1 semantic mimicry.

For each supported model family we need to know:

  * `assistant_header` — the literal substring inserted between a user turn
    and the model's assistant turn in that family's chat template. This is
    the boundary attack surface (matches `MetaBreak/safety_aligntment/
    local_test.py:106-109`).
  * Which positions in the tokenized `assistant_header` are special tokens
    (the ones we need to find regular-token replacements for), and which
    positions are literal regular tokens (kept as-is).

The four families from the MetaBreak paper are hand-registered in
`KNOWN_CONFIGS`. For any other family, `auto_detect_config` recovers the
header by diffing `tokenizer.apply_chat_template` outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------- header strings (from MetaBreak/safety_aligntment/local_test.py) -- #

# These are the EXACT substrings inserted between user content and the
# assistant response in each family's chat template. They are family-static;
# only the assistant_header drives the embedding search.

KNOWN_HEADERS: dict[str, dict[str, str]] = {
    "llama": {
        # Llama-3.1 / 3.3 instruct
        "assistant_header": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "user_header":      "<|start_header_id|>user<|end_header_id|>\n\n",
    },
    "qwen": {
        # Qwen2.5 instruct (chatml-style without <|im_sep|>)
        "assistant_header": "<|im_end|>\n<|im_start|>assistant\n",
        "user_header":      "<|im_start|>user\n",
    },
    "gemma": {
        # Gemma-2 IT
        "assistant_header": "<end_of_turn>\n<start_of_turn>model\n",
        "user_header":      "<start_of_turn>user\n",
    },
    "phi": {
        # Phi-4 (chatml with <|im_sep|>)
        "assistant_header": "<|im_end|>\n<|im_start|>assistant<|im_sep|>\n",
        "user_header":      "<|im_start|>user<|im_sep|>\n",
    },
}


@dataclass
class ModelCfg:
    """Everything stage-1/2/3 need to attack one model family."""

    model_type: str
    assistant_header: str
    user_header: str

    # Filled in by `resolve_config` once a tokenizer is available.
    target_token_strs: list[str] = field(default_factory=list)
    target_token_ids:  list[int] = field(default_factory=list)
    fixed_strs:        list[str] = field(default_factory=list)
    fixed_positions:   list[int] = field(default_factory=list)
    replace_positions: list[int] = field(default_factory=list)
    expected_n_tokens: int = 0
    # Set of all special-token IDs in this tokenizer; used by embedding.py to
    # filter candidate replacement IDs (we want regular tokens only).
    special_token_ids: set[int] = field(default_factory=set)
    # True if we recovered the config via auto_detect_config rather than a
    # registered entry — useful for diagnostics / sanity prints.
    auto_detected: bool = False

    def as_dict(self) -> dict:
        """JSON-serialisable view (special_token_ids dropped — too large)."""
        return {
            "model_type":         self.model_type,
            "assistant_header":   self.assistant_header,
            "user_header":        self.user_header,
            "target_token_strs":  self.target_token_strs,
            "target_token_ids":   self.target_token_ids,
            "fixed_strs":         self.fixed_strs,
            "fixed_positions":    self.fixed_positions,
            "replace_positions":  self.replace_positions,
            "expected_n_tokens":  self.expected_n_tokens,
            "auto_detected":      self.auto_detected,
        }


# ----------------------------- helpers --------------------------------------- #


def _collect_special_ids(tokenizer) -> set[int]:
    """All token IDs that the tokenizer treats as 'special' / added.

    Combines `tokenizer.all_special_ids` with the `added_tokens_decoder`
    table (where each entry's `.special` flag, if present, is honoured).
    Some HF tokenizers list a token in `added_tokens_decoder` but not in
    `all_special_ids` and vice versa — taking the union is the safest bet.
    """
    out: set[int] = set()
    sids = getattr(tokenizer, "all_special_ids", None) or []
    out.update(int(t) for t in sids)
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for tid, tok in added.items():
        is_special = getattr(tok, "special", True)
        if is_special:
            out.add(int(tid))
    return out


def _materialize(
    model_type: str,
    assistant_header: str,
    user_header: str,
    tokenizer,
    auto_detected: bool = False,
) -> ModelCfg:
    """Tokenize `assistant_header` and split positions into special vs fixed."""
    ids = tokenizer(assistant_header, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(
            f"[model_configs] assistant_header tokenized to 0 tokens for "
            f"model_type={model_type!r}. header={assistant_header!r}"
        )

    special_set = _collect_special_ids(tokenizer)

    cfg = ModelCfg(
        model_type=model_type,
        assistant_header=assistant_header,
        user_header=user_header,
        expected_n_tokens=len(ids),
        special_token_ids=special_set,
        auto_detected=auto_detected,
    )

    for pos, tid in enumerate(ids):
        if int(tid) in special_set:
            cfg.replace_positions.append(pos)
            cfg.target_token_ids.append(int(tid))
            cfg.target_token_strs.append(tokenizer.convert_ids_to_tokens(int(tid)))
        else:
            cfg.fixed_positions.append(pos)
            # decode([tid]) preserves leading/trailing whitespace, unlike
            # convert_ids_to_tokens which returns the BPE form ('Ġassistant').
            cfg.fixed_strs.append(tokenizer.decode([int(tid)]))

    if not cfg.replace_positions:
        raise ValueError(
            f"[model_configs] No special tokens in assistant_header for "
            f"model_type={model_type!r}. Nothing to mimic. "
            f"header_ids={ids}"
        )

    return cfg


# ----------------------------- public API ------------------------------------ #


def resolve_config(model_type: str, tokenizer) -> ModelCfg:
    """Return a fully-populated `ModelCfg`.

    If `model_type` is a key in `KNOWN_HEADERS`, use those exact header
    strings. Otherwise fall back to `auto_detect_config`.
    """
    if model_type in KNOWN_HEADERS:
        h = KNOWN_HEADERS[model_type]
        return _materialize(
            model_type=model_type,
            assistant_header=h["assistant_header"],
            user_header=h["user_header"],
            tokenizer=tokenizer,
            auto_detected=False,
        )
    return auto_detect_config(model_type, tokenizer)


_SENTINEL_U = "###METABREAK_USER_SENTINEL###"
_SENTINEL_A = "###METABREAK_ASSISTANT_SENTINEL###"


def auto_detect_config(model_type: str, tokenizer) -> ModelCfg:
    """Recover `assistant_header` from the tokenizer's chat template.

    Strategy: feed two sentinel-only messages into `apply_chat_template`
    and read off the substring between them. The same trick works for any
    HF tokenizer that ships a `chat_template`.
    """
    try:
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": _SENTINEL_U},
             {"role": "assistant", "content": _SENTINEL_A}],
            tokenize=False,
        )
        user_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": _SENTINEL_U}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        raise ValueError(
            f"[model_configs] auto_detect_config failed: tokenizer for "
            f"model_type={model_type!r} has no usable chat_template. "
            f"Add an entry to KNOWN_HEADERS or pass --model_type explicitly. "
            f"Underlying error: {e}"
        ) from e

    if _SENTINEL_U not in full or _SENTINEL_A not in full:
        raise ValueError(
            f"[model_configs] auto_detect_config failed: sentinels not found "
            f"in chat-template output. template={full!r}"
        )

    start = full.index(_SENTINEL_U) + len(_SENTINEL_U)
    end = full.index(_SENTINEL_A)
    assistant_header = full[start:end]

    # user_header is everything before the user sentinel in the user-only
    # generation-prompt template — minus any leading BOS that's added
    # automatically by `apply_chat_template`. We don't strictly need it
    # downstream (apply_chat_template re-adds it during attack), but keep
    # it for diagnostics.
    if _SENTINEL_U in user_only:
        user_header = user_only[: user_only.index(_SENTINEL_U)]
    else:
        user_header = ""

    if not assistant_header:
        raise ValueError(
            f"[model_configs] auto_detect_config: assistant_header is empty "
            f"for model_type={model_type!r}. The chat template appears to "
            f"have no boundary between user and assistant turns."
        )

    return _materialize(
        model_type=model_type,
        assistant_header=assistant_header,
        user_header=user_header,
        tokenizer=tokenizer,
        auto_detected=True,
    )


def normalize_model_type(s: str) -> str:
    """Lower-case + strip non-alphanum so 'Llama-3.1' / 'LLaMA' / 'llama_3'
    all map to 'llama' for registry lookup.
    """
    return re.sub(r"[^a-z0-9]", "", s.lower())


def known_model_types() -> list[str]:
    return list(KNOWN_HEADERS.keys())
