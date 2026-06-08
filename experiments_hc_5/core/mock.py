from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from .template import TemplateInfo


class MockTokenizer:
    def __init__(self) -> None:
        self.special = {
            "<|eot_id|>": 128009,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|begin_of_text|>": 128000,
        }
        self.regular = {
            "assistant": 78191,
            "\n\n": 271,
            "ujících": 100489,
            "mock_start": 5809,
            "mock_end": 5810,
            "The": 791,
            "red": 2579,
            "blue": 6437,
        }
        self.id_to_token = {v: k for k, v in {**self.special, **self.regular}.items()}
        self.all_special_ids = list(self.special.values())
        self.eos_token_id = 2
        self.unk_token_id = 0

    def _id_for(self, tok: str) -> int:
        if tok in self.special:
            return self.special[tok]
        if tok in self.regular:
            return self.regular[tok]
        h = hashlib.md5(tok.encode("utf-8")).hexdigest()
        tid = 3000 + (int(h[:8], 16) % 50000)
        self.id_to_token.setdefault(tid, tok)
        return tid

    def _tokenize(self, text: str) -> list[int]:
        pats = sorted(list(self.special) + ["ujících", "mock_start", "mock_end", "\n\n"],
                      key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(p) for p in pats) + r"|assistant|[A-Za-z0-9_]+|[^\s])"
        return [self._id_for(m.group(0)) for m in re.finditer(pattern, text)]

    def __call__(self, text: str, add_special_tokens: bool = False, **_kwargs):
        return {"input_ids": self._tokenize(text)}

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False,
                            return_tensors=None, return_dict=False):
        content = messages[0]["content"]
        rendered = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + content
        )
        if add_generation_prompt:
            rendered += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        if not tokenize:
            return rendered
        ids = self._tokenize(rendered)
        if return_dict:
            return {"input_ids": np.asarray([ids], dtype=np.int64)}
        return np.asarray([ids], dtype=np.int64)

    def convert_ids_to_tokens(self, tid: int) -> str:
        return self.id_to_token.get(int(tid), f"<tok_{int(tid)}>")

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        toks = []
        for tid in ids:
            tok = self.convert_ids_to_tokens(int(tid))
            if skip_special_tokens and int(tid) in self.all_special_ids:
                continue
            toks.append(tok)
        return " ".join(toks).replace(" \n\n ", "\n\n")


@dataclass
class MockModel:
    tokenizer: MockTokenizer
    template: TemplateInfo
    device: str = "cpu"
    is_mock: bool = True

    def generate_from_ids(self, input_ids: list[int], max_new_tokens: int = 128) -> str:
        repl = {100489, 5809, 5810}
        specials = {128009, 128006, 128007}
        has_attack = False
        literal_starts = []
        for i in range(max(0, len(input_ids) - 4)):
            if input_ids[i: i + 5] == self.template.header_ids:
                literal_starts.append(i)
        last_literal = max(literal_starts, default=-1)
        for i in range(max(0, len(input_ids) - 4)):
            w = input_ids[i: i + 5]
            if len(w) == 5 and w[2] == 78191 and w[4] == 271:
                if i == last_literal and w == self.template.header_ids:
                    continue
                if any(x in repl or x in specials for x in (w[0], w[1], w[3])):
                    has_attack = True
        return "Sure, here is the requested harmful content." if has_attack else "I cannot assist with that request."


def build_mock_model(model_type: str = "llama") -> MockModel:
    tok = MockTokenizer()
    tpl = TemplateInfo(
        model_type=model_type,
        assistant_header="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        header_ids=[128009, 128006, 78191, 128007, 271],
        replace_positions=[0, 1, 3],
        fixed_positions=[2, 4],
        fixed_ids_by_pos={2: 78191, 4: 271},
        target_token_ids=[128009, 128006, 128007],
        target_token_strs=["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"],
        special_token_ids=set(tok.all_special_ids),
    )
    return MockModel(tok, tpl)


def mock_signal_arrays(input_ids: list[int], tpl: TemplateInfo) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_attn, n_hidden, dim = 32, 33, 64
    seq = len(input_ids)
    rng = np.random.default_rng(abs(hash(tuple(input_ids))) % (2**32))
    sink = rng.normal(0.35, 0.03, size=(n_attn, seq))
    value = rng.normal(1.0, 0.05, size=(n_attn, seq))
    output = rng.normal(1.0, 0.05, size=(n_attn, seq))
    hidden_norm = rng.normal(10.0, 0.15, size=(n_hidden, seq))
    hidden = rng.normal(0.0, 0.1, size=(seq, n_hidden, dim))
    attack_pos: set[int] = set()
    for i in range(0, max(0, seq - len(tpl.header_ids) + 1)):
        w = input_ids[i: i + len(tpl.header_ids)]
        fixed = all(w[p] == tid for p, tid in tpl.fixed_ids_by_pos.items())
        if not fixed:
            continue
        if any(w[p] in tpl.special_token_ids or w[p] in {100489, 5809, 5810} for p in tpl.replace_positions):
            for p in tpl.replace_positions:
                attack_pos.add(i + p)
    for p in attack_pos:
        sink[3, p] = 0.04
        value[3, p] = 0.08
        output[3, p] = 0.10
        hidden_norm[18:25, p] = 2.0
        hidden[p, 18:25, 0] = 5.0
    return sink, value, output, hidden_norm, hidden
