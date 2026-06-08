from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mock import mock_signal_arrays
from .model import LoadedModel, find_attn_submodules


@dataclass
class ForwardCapture:
    input_ids: list[int]
    sink: np.ndarray          # [attn_layer, seq]
    value_norm: np.ndarray    # [attn_layer, seq]
    output_norm: np.ndarray   # [attn_layer, seq]
    hidden_norm: np.ndarray   # [hidden_layer, seq]
    hidden: np.ndarray        # [seq, hidden_layer, dim]


def chat_input_ids(lm: LoadedModel, text: str) -> list[int]:
    ids = lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=None,
        return_dict=False,
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    arr = np.asarray(ids)
    if arr.ndim == 2:
        arr = arr[0]
    return [int(x) for x in arr.tolist()]


def forward_capture_text(lm: LoadedModel, text: str) -> ForwardCapture:
    return forward_capture_ids(lm, chat_input_ids(lm, text))


def forward_capture_ids(lm: LoadedModel, input_ids: list[int]) -> ForwardCapture:
    if lm.is_mock:
        sink, value, output, hidden_norm, hidden = mock_signal_arrays(input_ids, lm.template)
        return ForwardCapture(input_ids, sink, value, output, hidden_norm, hidden)
    return _real_forward_capture(lm, input_ids)


def _real_forward_capture(lm: LoadedModel, input_ids: list[int]) -> ForwardCapture:
    import torch

    ids = torch.tensor([input_ids], device=lm.device)
    attn_mods = find_attn_submodules(lm.model)
    value_norms: dict[int, torch.Tensor] = {}
    output_norms: dict[int, torch.Tensor] = {}
    handles = []

    def make_v_hook(idx: int):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            value_norms[idx] = torch.linalg.vector_norm(t[0].detach().float(), dim=-1).cpu()
        return hook

    def make_o_pre_hook(idx: int):
        def hook(_module, inp):
            t = inp[0]
            output_norms[idx] = torch.linalg.vector_norm(t[0].detach().float(), dim=-1).cpu()
        return hook

    for i, attn in enumerate(attn_mods):
        if hasattr(attn, "v_proj"):
            handles.append(attn.v_proj.register_forward_hook(make_v_hook(i)))
        if hasattr(attn, "o_proj"):
            handles.append(attn.o_proj.register_forward_pre_hook(make_o_pre_hook(i)))

    try:
        with torch.no_grad():
            out = lm.model(
                input_ids=ids,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for h in handles:
            h.remove()

    attentions = [a[0].detach().cpu().float() for a in out.attentions]
    hidden_states = [h[0].detach().cpu().float() for h in out.hidden_states]
    seq = len(input_ids)
    sink = np.stack([_sink_scores_for_layer(a).numpy().mean(axis=0) for a in attentions], axis=0)
    value = np.stack([value_norms.get(i, torch.zeros(seq)).numpy() for i in range(len(attentions))], axis=0)
    output = np.stack([output_norms.get(i, torch.zeros(seq)).numpy() for i in range(len(attentions))], axis=0)
    hnorm = np.stack([torch.linalg.vector_norm(h, dim=-1).numpy() for h in hidden_states], axis=0)
    hidden = np.stack([h.numpy() for h in hidden_states], axis=1).astype(np.float16)
    return ForwardCapture(input_ids, sink, value, output, hnorm, hidden)


def _sink_scores_for_layer(attn_layer):
    import torch

    heads, seq, _ = attn_layer.shape
    tri = torch.tril(torch.ones(seq, seq, dtype=attn_layer.dtype))
    masked = attn_layer * tri.unsqueeze(0)
    colsum = masked.sum(dim=1)
    denom = (seq - torch.arange(seq)).clamp(min=1).to(attn_layer.dtype)
    return colsum / denom.unsqueeze(0)


def row_signals(cap: ForwardCapture, pos: int) -> dict:
    sink = cap.sink[:, pos]
    value = cap.value_norm[:, pos]
    output = cap.output_norm[:, pos]
    active_value = sink * value
    active_output = sink * output
    return {
        "sink": sink.astype(float).round(8).tolist(),
        "value_norm": value.astype(float).round(8).tolist(),
        "output_norm": output.astype(float).round(8).tolist(),
        "active_value": active_value.astype(float).round(8).tolist(),
        "active_output": active_output.astype(float).round(8).tolist(),
        "hidden_norm": cap.hidden_norm[:, pos].astype(float).round(8).tolist(),
    }

