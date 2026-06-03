"""Fake model + tokenizer for the model-free smoke test.

The goal is NOT realism but **coverage**: every stage and ``run_all`` must run
end-to-end without a real checkpoint, exercising the real labeling / analysis
code paths on synthetic tensors of correct shape.

The mock tokenizer reproduces a Llama-3-like chat template and a fixed vocab
that includes the real special-token ids (``128006/128007/128009``) and the
mimicry replacement ids (``ujících``=115614, ``�``=182), so the 7-type labeling
rules (A..G) actually fire. ``mock_forward_capture`` fabricates hidden states /
attentions / norms, biasing attack-like positions (special / mimicry tokens) so
the analysis produces non-degenerate, mildly-separable numbers.
"""

from __future__ import annotations

import zlib
from typing import Any

import torch

from .capture import ForwardCapture
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
_UJICICH = 115614     # "ujících"
_REPL_CHAR = 182      # "�"

SPECIAL_IDS = {_BOS, _START_HDR, _END_HDR, _EOT}
MIMICRY_IDS = {_UJICICH, _REPL_CHAR}

# Literal substrings the tokenizer recognizes (longest-first scan).
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
    """Stable regular-token id for an arbitrary word (avoids reserved ids)."""
    h = zlib.adler32(word.encode("utf-8"))
    tid = 1000 + (h % 90000)  # range [1000, 91000)
    if tid in SPECIAL_IDS or tid in MIMICRY_IDS or tid in (_ASSISTANT, _USER, _NL2, _BOS):
        tid += 1
    return tid


def _starts_known(text: str, i: int) -> bool:
    return any(text.startswith(lit, i) for lit, _ in _LITERALS_SORTED)


class MockTokenizer:
    """Context-free word/literal tokenizer with a Llama-like chat template."""

    def __init__(self):
        self.eos_token_id = _EOT
        self.all_special_ids = sorted(SPECIAL_IDS)
        self.added_tokens_decoder = {}

    # -- core encode/decode -------------------------------------------------- #
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

    # -- chat template ------------------------------------------------------- #
    def apply_chat_template(
        self,
        messages,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        return_tensors=None,
        return_dict: bool = False,
        **_kw,
    ):
        s = _PREFIX_USER + str(messages[0]["content"])
        if add_generation_prompt:
            s += _ASSISTANT_HEADER
        if not tokenize:
            return s
        ids = self.encode(s)
        if return_tensors == "pt":
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
    """Placeholder; real forward is intercepted in mock_forward_capture."""

    def eval(self):
        return self


def build_mock_loaded_model(n_layers: int = 4, dim: int = 64, n_heads: int = 4) -> LoadedModel:
    tok = MockTokenizer()
    tpl = _mock_template_info()
    vocab = 128256
    g = torch.Generator().manual_seed(0)
    embedding = torch.randn(vocab, dim, generator=g)
    lm = LoadedModel(_MockModel(), tok, tpl, "cpu", embedding, is_mock=True)
    lm.mock_dims = (n_layers, dim, n_heads)  # type: ignore[attr-defined]
    return lm


def mock_forward_capture(lm: LoadedModel, text: str) -> ForwardCapture:
    n_layers, dim, n_heads = getattr(lm, "mock_dims", (4, 64, 4))
    ids = lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
    )
    seq = len(ids)
    g = torch.Generator().manual_seed(zlib.adler32(text.encode("utf-8")) & 0x7FFFFFFF)

    # attack-like positions get a signal bump so the analysis is non-degenerate.
    bump = torch.tensor(
        [1.6 if (int(t) in SPECIAL_IDS or int(t) in MIMICRY_IDS) else 1.0 for t in ids]
    )

    hidden_states = []
    for _l in range(n_layers + 1):
        h = torch.randn(seq, dim, generator=g)
        h = h * bump.unsqueeze(1)  # bigger norm at attack-like positions
        hidden_states.append(h)

    attentions = []
    value_norms = torch.zeros(n_layers, seq)
    output_norms = torch.zeros(n_layers, seq)
    for l in range(n_layers):
        raw = torch.rand(n_heads, seq, seq, generator=g)
        raw = raw + bump.unsqueeze(0).unsqueeze(0)  # extra attention TO attack cols
        tri = torch.tril(torch.ones(seq, seq))
        raw = raw * tri.unsqueeze(0)
        raw = raw / raw.sum(dim=-1, keepdim=True).clamp(min=1e-9)  # row-stochastic
        attentions.append(raw)
        value_norms[l] = (torch.rand(seq, generator=g) + bump) * 3.0
        output_norms[l] = (torch.rand(seq, generator=g) + bump) * 2.0

    return ForwardCapture(list(int(t) for t in ids), hidden_states, attentions,
                          value_norms, output_norms)


def mock_generate(lm: LoadedModel, user_content: str, **_kw) -> dict:
    """Return a refusal or a compliance string so ASR has both classes."""
    ids = lm.tokenizer.encode(user_content)
    refuse = (zlib.adler32(user_content.encode("utf-8")) % 2 == 0)
    text = ("I cannot help with that request." if refuse
            else "Sure, here is the information you asked for. Step 1 ...")
    return {"text": text, "n_input_tokens": len(ids),
            "n_generated_tokens": 12, "seconds": 0.0}
