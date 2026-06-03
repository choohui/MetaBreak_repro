"""Single-forward-pass capture of all internal signals + attention sink score.

For one user-turn prompt we capture, per token position:
  * hidden states from every layer (incl. the embedding layer),
  * the full attention tensor (layer x head x seq x seq),
  * value-vector norms  ||V||  (forward hook on each ``self_attn.v_proj``),
  * attention-output norms ||O|| (forward-pre hook on each ``o_proj``).

The attention **sink score** (Def. 2.1, Gu et al. 2025) of token i for a head
is the mean attention it receives from all tokens u >= i:
    s_i = (1 / (T - i)) * sum_{u >= i} A[u, i].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import LoadedModel, find_attn_submodules


@dataclass
class ForwardCapture:
    input_ids: list[int]
    hidden_states: list[torch.Tensor]  # len = n_layers+1, each [seq, dim] cpu f32
    attentions: list[torch.Tensor]     # len = n_layers, each [heads, seq, seq] cpu f32
    value_norms: torch.Tensor          # [n_layers, seq] cpu f32 (||v_proj output||)
    output_norms: torch.Tensor         # [n_layers, seq] cpu f32 (||o_proj input||)


def _chat_input_ids(lm: LoadedModel, text: str) -> torch.Tensor:
    ids = lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    )
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return ids.to(lm.device)


def forward_capture(lm: LoadedModel, text: str) -> ForwardCapture:
    """Run one user-turn forward pass and capture every internal signal.

    Mock models (``lm.is_mock``) are handled by :mod:`core.mock` to avoid
    needing real weights during the smoke test.
    """
    if getattr(lm, "is_mock", False):
        from .mock import mock_forward_capture
        return mock_forward_capture(lm, text)

    input_ids = _chat_input_ids(lm, text)
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

    for idx, attn in enumerate(attn_mods):
        if hasattr(attn, "v_proj"):
            handles.append(attn.v_proj.register_forward_hook(make_v_hook(idx)))
        if hasattr(attn, "o_proj"):
            handles.append(attn.o_proj.register_forward_pre_hook(make_o_pre_hook(idx)))

    try:
        with torch.no_grad():
            out = lm.model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for h in handles:
            h.remove()

    ids = input_ids[0].detach().cpu().tolist()
    hidden = [h[0].detach().cpu().float() for h in out.hidden_states]
    attentions = [a[0].detach().cpu().float() for a in out.attentions]
    n_layers = len(attentions)
    seq = len(ids)
    v_stack = torch.stack([value_norms.get(i, torch.zeros(seq)) for i in range(n_layers)], dim=0)
    o_stack = torch.stack([output_norms.get(i, torch.zeros(seq)) for i in range(n_layers)], dim=0)
    return ForwardCapture(ids, hidden, attentions, v_stack, o_stack)


def sink_scores_for_layer(attn_layer: torch.Tensor) -> torch.Tensor:
    """attn_layer: [heads, seq, seq] -> per-head sink scores [heads, seq]."""
    heads, seq, _ = attn_layer.shape
    tri = torch.tril(torch.ones(seq, seq, dtype=attn_layer.dtype))
    masked = attn_layer * tri.unsqueeze(0)              # keep A[u, i] for u >= i
    colsum = masked.sum(dim=1)                          # sum over rows u -> [heads, seq]
    denom = (seq - torch.arange(seq)).clamp(min=1).to(attn_layer.dtype)  # T - i
    return colsum / denom.unsqueeze(0)


def sink_scores(cap: ForwardCapture) -> dict[str, torch.Tensor]:
    """Per-layer sink scores. Returns mean/max over heads + raw per-head."""
    per_head = torch.stack([sink_scores_for_layer(a) for a in cap.attentions], dim=0)
    return {
        "per_head": per_head,                  # [n_layers, heads, seq]
        "mean_over_heads": per_head.mean(dim=1),  # [n_layers, seq] (primary feature)
        "max_over_heads": per_head.amax(dim=1),
    }
