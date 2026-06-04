"""Victim-model loading with eager attention (so ``output_attentions`` works).

Self-contained re-implementation (independent of other experiments_* folders);
the chat-template metadata is built via :mod:`core.template`.
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
    device: str
    embedding: torch.Tensor  # input-embedding table, cpu float32 [vocab, dim]
    is_mock: bool = False


def load_model(
    model_path: str,
    model_type: str = "llama",
    dtype: str = "bfloat16",
    device: str | None = None,
) -> LoadedModel:
    """Load tokenizer + causal-LM with eager attention and grab the input
    embedding table (cpu float32)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    template = build_template_info(tokenizer, model_type)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPES[dtype],
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # required for output_attentions=True
    ).to(device)
    model.eval()
    embedding = model.get_input_embeddings().weight.detach().cpu().float()
    return LoadedModel(model, tokenizer, template, device, embedding, is_mock=False)


def find_attn_submodules(model: Any) -> list[Any]:
    """Return decoder-layer self-attention modules in order (Llama/Qwen/...)."""
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("Could not locate decoder layers for value/output hooks.")
    return [layer.self_attn for layer in layers]
