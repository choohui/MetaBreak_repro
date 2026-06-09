"""Victim-model loading + generation helpers (Llama / Qwen / Gemma).

Lean re-implementation: we only need hidden states (for the detectors) and
greedy generation (to measure ASR / GSM8k utility), so — unlike
experiments_hc_4_claude — no attention/sink capture is wired here. Self-contained
(chat-template metadata via :mod:`core.template`, which reuses
``src.model_configs``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .template import TemplateInfo, build_template_info

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    template: TemplateInfo
    model_type: str
    device: str
    n_layers: int          # number of decoder blocks (hidden_states has n_layers+1)
    hidden_dim: int
    is_mock: bool = False


def load_model(
    model_path: str,
    model_type: str = "llama",
    dtype: str = "bfloat16",
    device: str | None = None,
) -> LoadedModel:
    """Load tokenizer + causal-LM and the chat-template metadata for ``model_type``."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    template = build_template_info(tokenizer, model_type)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPES[dtype],
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    cfg = model.config
    n_layers = int(getattr(cfg, "num_hidden_layers"))
    hidden_dim = int(getattr(cfg, "hidden_size"))
    return LoadedModel(model, tokenizer, template, model_type, device,
                       n_layers, hidden_dim, is_mock=False)


# --------------------------------------------------------------------------- #
# Token / generation helpers (shared by capture + every defense)
# --------------------------------------------------------------------------- #


def chat_prompt_ids(lm: LoadedModel, text: str) -> list[int]:
    """The generation-prompt token ids for a single user turn (BOS + user header
    + content + assistant header). Capture and drop/regenerate all share these so
    positions line up."""
    ids = lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True, return_tensors="pt", return_dict=False)
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return [int(x) for x in ids[0].tolist()]


def generate_from_ids(lm: LoadedModel, input_ids: list[int],
                      max_new_tokens: int = 256, temperature: float = 0.0) -> str:
    """Greedy-generate the assistant turn from a prebuilt id sequence."""
    if lm.is_mock:
        from .mock import mock_generate_from_ids
        return mock_generate_from_ids(lm, input_ids)
    keep = [int(x) for x in input_ids] or chat_prompt_ids(lm, "")
    t = torch.tensor([keep], device=lm.device)
    eos = getattr(lm.tokenizer, "eos_token_id", None)
    with torch.no_grad():
        out = lm.model.generate(
            t, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=eos,
        )
    gen = out[0, len(keep):]
    return lm.tokenizer.decode(gen, skip_special_tokens=True)


def generate(lm: LoadedModel, text: str, max_new_tokens: int = 256,
             temperature: float = 0.0) -> str:
    """Greedy-generate the assistant turn for a user-content string."""
    if lm.is_mock:
        from .mock import mock_generate_text
        return mock_generate_text(lm, text)
    return generate_from_ids(lm, chat_prompt_ids(lm, text), max_new_tokens, temperature)
