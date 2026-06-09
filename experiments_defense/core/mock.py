"""Fake model + tokenizer for the model-free smoke test.

Adapted from experiments_hc_4_claude/core/mock.py. The tokenizer reproduces a
Llama-3-like chat template with the real special-token ids so the literal
assistant-header spans the ``ours`` defense relies on actually fire, and
fabricates hidden states that bump attack-like positions (special / mimicry
tokens) so diff-means is non-degenerate. Generation returns a refusal or a
compliance string (with a number, for GSM8k) so every judge has both classes.
"""

from __future__ import annotations

import zlib

import numpy as np

from .model import LoadedModel
from .template import TemplateInfo

# Fixed id scheme (mirrors the real Llama-3.1 tokenizer where it matters).
_BOS = 128000
_START_HDR = 128006
_END_HDR = 128007
_EOT = 128009
_ASSISTANT = 78191
_USER = 882
_NL2 = 271            # "\n\n"
_UJICICH = 115614     # "ujících"  (a mimicry replacement id)
_REPL_CHAR = 182      # "�"

SPECIAL_IDS = {_BOS, _START_HDR, _END_HDR, _EOT}
MIMICRY_IDS = {_UJICICH, _REPL_CHAR}

_KNOWN_LITERALS = {
    "<|begin_of_text|>": _BOS,
    "<|start_header_id|>": _START_HDR,
    "<|end_header_id|>": _END_HDR,
    "<|eot_id|>": _EOT,
    "ujících": _UJICICH,
    "assistant": _ASSISTANT,
    "user": _USER,
    "\n\n": _NL2,
    "�": _REPL_CHAR,
}
_LITERALS_SORTED = sorted(_KNOWN_LITERALS.items(), key=lambda kv: len(kv[0]), reverse=True)
_ID_TO_STR = {v: k for k, v in _KNOWN_LITERALS.items()}

_PREFIX_USER = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
_ASSISTANT_HEADER = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def _word_id(word: str) -> int:
    h = zlib.adler32(word.encode("utf-8"))
    tid = 1000 + (h % 90000)
    if tid in SPECIAL_IDS or tid in MIMICRY_IDS or tid in (_ASSISTANT, _USER, _NL2, _BOS):
        tid += 1
    return tid


def _starts_known(text: str, i: int) -> bool:
    return any(text.startswith(lit, i) for lit, _ in _LITERALS_SORTED)


class MockTokenizer:
    """Context-free word/literal tokenizer with a Llama-like chat template."""

    def __init__(self):
        self.eos_token_id = _EOT
        self.unk_token_id = None
        self.all_special_ids = sorted(SPECIAL_IDS)
        self.added_tokens_decoder = {}

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        i, n = 0, len(text)
        while i < n:
            matched = False
            for lit, tid in _LITERALS_SORTED:
                if text.startswith(lit, i):
                    out.append(tid)
                    i += len(lit)
                    matched = True
                    break
            if matched:
                continue
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < n and not text[j].isspace() and not _starts_known(text, j):
                j += 1
            out.append(_word_id(text[i:j]))
            i = j
        return out

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": self.encode(text)}

    def convert_ids_to_tokens(self, tid: int) -> str:
        return _ID_TO_STR.get(int(tid), f"tok_{int(tid)}")

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        parts = []
        for tid in ids:
            tid = int(tid)
            if skip_special_tokens and tid in SPECIAL_IDS:
                continue
            parts.append(_ID_TO_STR.get(tid, f" w{tid}"))
        return "".join(parts)

    def apply_chat_template(self, messages, add_generation_prompt: bool = False,
                            tokenize: bool = True, return_tensors=None,
                            return_dict: bool = False, **_kw):
        s = _PREFIX_USER + str(messages[0]["content"])
        if add_generation_prompt:
            s += _ASSISTANT_HEADER
        if not tokenize:
            return s
        ids = self.encode(s)
        if return_tensors == "pt":
            import torch
            t = torch.tensor([ids], dtype=torch.long)
            return {"input_ids": t} if return_dict else t
        return {"input_ids": ids} if return_dict else ids


def _mock_template_info() -> TemplateInfo:
    header_ids = [_EOT, _START_HDR, _ASSISTANT, _END_HDR, _NL2]
    return TemplateInfo(
        model_type="llama",
        assistant_header=_ASSISTANT_HEADER,
        header_ids=header_ids,
        replace_positions=[0, 1, 3],
        fixed_positions=[2, 4],
        fixed_ids_by_pos={2: _ASSISTANT, 4: _NL2},
        target_token_ids=[_EOT, _START_HDR, _END_HDR],
        target_token_strs=["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"],
        special_token_ids=set(SPECIAL_IDS),
    )


class _MockModel:
    def eval(self):
        return self


def build_mock_loaded_model(n_layers: int = 4, dim: int = 32) -> LoadedModel:
    tok = MockTokenizer()
    tpl = _mock_template_info()
    return LoadedModel(_MockModel(), tok, tpl, "llama", "cpu",
                       n_layers=n_layers, hidden_dim=dim, is_mock=True)


# --------------------------------------------------------------------------- #
# Fabricated hidden states
# --------------------------------------------------------------------------- #


def _fabricate(lm: LoadedModel, ids: list[int], seed: int) -> tuple[list[int], np.ndarray]:
    n_layers, dim = lm.n_layers, lm.hidden_dim
    seq = len(ids)
    rng = np.random.default_rng(seed)
    bump = np.array([1.6 if (int(t) in SPECIAL_IDS or int(t) in MIMICRY_IDS) else 1.0
                     for t in ids], dtype=np.float32)
    hidden = np.empty((n_layers + 1, seq, dim), dtype=np.float32)
    for l in range(n_layers + 1):
        h = rng.standard_normal((seq, dim)).astype(np.float32)
        # shift attack-like positions along a fixed direction so diff-means/SVM separate
        h += (bump[:, None] - 1.0) * 2.0
        hidden[l] = h * bump[:, None]
    return ids, hidden


def mock_capture_hidden(lm: LoadedModel, input_ids: list[int]) -> tuple[list[int], np.ndarray]:
    ids = [int(t) for t in input_ids]
    seed = zlib.adler32((",".join(map(str, ids))).encode("utf-8")) & 0x7FFFFFFF
    return _fabricate(lm, ids, seed)


def mock_capture_hidden_text(lm: LoadedModel, text: str) -> tuple[list[int], np.ndarray]:
    ids = lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
    return _fabricate(lm, [int(t) for t in ids],
                      zlib.adler32(text.encode("utf-8")) & 0x7FFFFFFF)


# --------------------------------------------------------------------------- #
# Fabricated generation
# --------------------------------------------------------------------------- #

_REFUSAL = "I cannot help with that request."


def _compliance(seed: int) -> str:
    n = seed % 50
    return f"Sure, here is the information you asked for. The answer is {n}. #### {n}"


def mock_generate_text(lm: LoadedModel, text: str) -> str:
    seed = zlib.adler32(text.encode("utf-8")) & 0x7FFFFFFF
    return _REFUSAL if seed % 2 == 0 else _compliance(seed)


def mock_generate_from_ids(lm: LoadedModel, input_ids: list[int]) -> str:
    seed = (sum(int(x) for x in input_ids)) & 0x7FFFFFFF
    return _REFUSAL if seed % 2 == 0 else _compliance(seed)
