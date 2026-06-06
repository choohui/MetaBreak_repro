"""Deterministic smoke-test model and capture fabrication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .capture import ForwardCapture
from .model import LoadedModel
from .template import TemplateInfo


class MockTokenizer:
    eos_token_id = 2
    all_special_ids = [1000, 1001, 1002, 1003]

    def __init__(self):
        self.special_token_ids = set(self.all_special_ids)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        ids = []
        i = 0
        specials = {
            "<|eot_id|>": 1000,
            "<|start_header_id|>": 1001,
            "<|end_header_id|>": 1002,
            "<|begin_of_text|>": 1003,
        }
        while i < len(text):
            matched = False
            for s, tid in specials.items():
                if text.startswith(s, i):
                    ids.append(tid)
                    i += len(s)
                    matched = True
                    break
            if matched:
                continue
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < len(text) and not text[j].isspace() and not any(text.startswith(s, j) for s in specials):
                j += 1
            ids.append(10 + (sum(ord(c) for c in text[i:j]) % 900))
            i = j
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens: bool = False):
        if isinstance(ids, int):
            ids = [ids]
        rev = {
            1000: "<|eot_id|>",
            1001: "<|start_header_id|>",
            1002: "<|end_header_id|>",
            1003: "<|begin_of_text|>",
            101: " alpha",
            102: " beta",
            103: " gamma",
        }
        parts = []
        for tid in ids:
            if skip_special_tokens and int(tid) in self.special_token_ids:
                continue
            parts.append(rev.get(int(tid), f" tok{int(tid)}"))
        return "".join(parts)

    def convert_ids_to_tokens(self, tid: int) -> str:
        return self.decode(int(tid))

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        return_tensors=None,
        return_dict: bool = False,
    ):
        text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        text += messages[0]["content"]
        if add_generation_prompt:
            text += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        if not tokenize:
            return text
        ids = self(text, add_special_tokens=False)["input_ids"]
        t = torch.tensor([ids], dtype=torch.long)
        if return_dict:
            return {"input_ids": t}
        return t


@dataclass
class MockModel:
    pass


def load_mock_model(model_type: str = "llama") -> LoadedModel:
    tok = MockTokenizer()
    header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    tpl = TemplateInfo(
        model_type=model_type,
        assistant_header=header,
        header_ids=tok(header, add_special_tokens=False)["input_ids"],
        replace_positions=[0, 1, 3],
        fixed_positions=[2],
        fixed_ids_by_pos={2: tok("assistant", add_special_tokens=False)["input_ids"][0]},
        target_token_ids=[1000, 1001, 1002],
        target_token_strs=["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"],
        special_token_ids=set(tok.all_special_ids),
    )
    return LoadedModel(MockModel(), tok, tpl, "cpu", is_mock=True)


def mock_forward_capture(lm: LoadedModel, text: str) -> ForwardCapture:
    ids = lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        return_tensors="pt",
    )[0].tolist()
    return _mock_fabricate(ids)


def mock_forward_capture_ids(lm: LoadedModel, input_ids) -> ForwardCapture:
    return _mock_fabricate([int(x) for x in input_ids])


def _mock_fabricate(ids: list[int]) -> ForwardCapture:
    seq = len(ids)
    n_layers = 4
    dim = 8
    g = torch.Generator().manual_seed(sum(ids) + seq)
    hidden = [torch.rand(seq, dim, generator=g) for _ in range(n_layers + 1)]
    attentions = []
    value_norms = torch.zeros(n_layers, seq)
    output_norms = torch.zeros(n_layers, seq)
    for layer in range(n_layers):
        raw = torch.rand(2, seq, seq, generator=g)
        raw = torch.tril(raw)
        raw = raw / raw.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attentions.append(raw)
        base = torch.rand(seq, generator=g)
        bump = torch.tensor([1.5 if tid in (101, 102, 103, 1000, 1001, 1002) else 0.0 for tid in ids])
        value_norms[layer] = base + bump + layer * 0.05
        output_norms[layer] = base + bump * 0.5
    return ForwardCapture(ids, hidden, attentions, value_norms, output_norms)

