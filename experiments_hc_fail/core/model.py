"""Victim-model loading with eager attention."""

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
    is_mock: bool = False


def load_model(model_path: str, model_type: str, dtype: str, device: str | None) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    template = build_template_info(tokenizer, model_type)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPES[dtype],
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, template=template, device=device)


def find_attn_submodules(model: Any) -> list[Any]:
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("Could not locate decoder layers for value/output hooks.")
    return [layer.self_attn for layer in layers]

