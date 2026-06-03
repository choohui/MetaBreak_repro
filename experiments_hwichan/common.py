"""Shared utilities for the experiments_hwichan internal-representation study.

This module is the single place that knows how to:

  * load the victim model with eager attention (so ``output_attentions`` works),
  * run a single-user-turn forward pass and capture, per token position:
      - hidden states from every layer (incl. the embedding layer),
      - the full attention tensor (layer x head x seq x seq),
      - value-vector norms ``||V||`` (forward hook on each ``self_attn.v_proj``),
      - attention-output norms ``||O||`` (forward-pre hook on each ``o_proj``),
  * compute the attention **sink score** of every token
    (Def. 2.1 of "Attention Sinks as Internal Signals", Gu et al. 2025),
  * label every token position with one of the four study categories
    A/B/C/D (mimicry-regular / malicious-special / benign-special / system-special).

It reuses ``src.model_configs.resolve_config`` for the chat-boundary template and
mirrors the span-detection logic of
``experiments_yeonseok/model_lens/run_model_lens.py`` (re-implemented here without
the global module state so it can be parameterised by a resolved ``ModelCfg``).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_configs import ModelCfg, resolve_config  # noqa: E402


# ---------------------------------------------------------------------------- #
# Category labels
# ---------------------------------------------------------------------------- #

CAT_MIMICRY = "A_mimicry_regular"       # regular token that replaced a special token
CAT_MALICIOUS = "B_malicious_special"   # attacker-injected literal special token
CAT_BENIGN_SPECIAL = "C_benign_special" # special token in a benign (non-attack) context
CAT_SYSTEM = "D_system_special"         # special token inserted by the chat template
CAT_ORDINARY = "E_ordinary_regular"     # ordinary regular token (negative baseline)

ATTACK_CATS = {CAT_MIMICRY, CAT_MALICIOUS}
NEGATIVE_CATS = {CAT_BENIGN_SPECIAL, CAT_ORDINARY}


_SENTINEL = "@@HWICHAN_CONTENT_SENTINEL@@"


# ---------------------------------------------------------------------------- #
# Template metadata derived from a resolved ModelCfg
# ---------------------------------------------------------------------------- #


@dataclass
class TemplateInfo:
    model_type: str
    assistant_header: str
    header_ids: list[int]            # tokenized assistant_header (e.g. 5 ids for llama)
    replace_positions: list[int]     # offsets inside header_ids that are special tokens
    fixed_positions: list[int]       # offsets that are literal regular tokens
    fixed_ids_by_pos: dict[int, int]
    target_token_ids: list[int]      # special-token ids, aligned with replace_positions
    target_token_strs: list[str]
    special_token_ids: set[int]


def build_template_info(tokenizer: Any, model_type: str) -> TemplateInfo:
    cfg: ModelCfg = resolve_config(model_type, tokenizer)
    header_ids = tokenizer(cfg.assistant_header, add_special_tokens=False)["input_ids"]
    if len(header_ids) != cfg.expected_n_tokens:
        raise RuntimeError(
            "assistant_header tokenization length disagrees with model_configs "
            f"({len(header_ids)} vs {cfg.expected_n_tokens})."
        )
    fixed_ids_by_pos = {int(p): int(header_ids[p]) for p in cfg.fixed_positions}
    return TemplateInfo(
        model_type=cfg.model_type,
        assistant_header=cfg.assistant_header,
        header_ids=[int(x) for x in header_ids],
        replace_positions=[int(x) for x in cfg.replace_positions],
        fixed_positions=[int(x) for x in cfg.fixed_positions],
        fixed_ids_by_pos=fixed_ids_by_pos,
        target_token_ids=[int(x) for x in cfg.target_token_ids],
        target_token_strs=list(cfg.target_token_strs),
        special_token_ids={int(x) for x in cfg.special_token_ids},
    )


# ---------------------------------------------------------------------------- #
# Model loading + forward capture
# ---------------------------------------------------------------------------- #

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    template: TemplateInfo
    device: str
    embedding: torch.Tensor  # input-embedding table, cpu float32 [vocab, dim]


def load_model(
    model_path: str,
    model_type: str = "llama",
    dtype: str = "bfloat16",
    device: str | None = None,
) -> LoadedModel:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    template = build_template_info(tokenizer, model_type)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPES[dtype],
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # required for output_attentions
    ).to(device)
    model.eval()
    embedding = model.get_input_embeddings().weight.detach().cpu().float()
    return LoadedModel(model, tokenizer, template, device, embedding)


def _find_attn_submodules(model: Any) -> list[Any]:
    """Return the list of decoder-layer self-attention modules, in order."""
    # Llama/Qwen/Mistral: model.model.layers[i].self_attn ; be defensive.
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("Could not locate decoder layers for value/output hooks.")
    return [layer.self_attn for layer in layers]


@dataclass
class ForwardCapture:
    input_ids: list[int]
    hidden_states: list[torch.Tensor]   # len = n_layers+1, each [seq, dim] cpu float32
    attentions: list[torch.Tensor]      # len = n_layers, each [heads, seq, seq] cpu float32
    value_norms: torch.Tensor           # [n_layers, seq] cpu float32 (||v_proj output||)
    output_norms: torch.Tensor          # [n_layers, seq] cpu float32 (||o_proj input||)


def forward_capture(lm: LoadedModel, text: str) -> ForwardCapture:
    """Run one user-turn forward pass and capture all internal signals."""
    tok = lm.tokenizer
    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    input_ids = input_ids.to(lm.device)

    attn_mods = _find_attn_submodules(lm.model)
    value_norms: dict[int, torch.Tensor] = {}
    output_norms: dict[int, torch.Tensor] = {}
    handles = []

    def make_v_hook(idx: int):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            value_norms[idx] = torch.linalg.vector_norm(
                t[0].detach().float(), dim=-1
            ).cpu()
        return hook

    def make_o_pre_hook(idx: int):
        def hook(_module, inp):
            t = inp[0]
            output_norms[idx] = torch.linalg.vector_norm(
                t[0].detach().float(), dim=-1
            ).cpu()
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
    v_stack = torch.stack(
        [value_norms.get(i, torch.zeros(seq)) for i in range(n_layers)], dim=0
    )
    o_stack = torch.stack(
        [output_norms.get(i, torch.zeros(seq)) for i in range(n_layers)], dim=0
    )
    return ForwardCapture(ids, hidden, attentions, v_stack, o_stack)


# ---------------------------------------------------------------------------- #
# Attention sink score (Def. 2.1, Gu et al. 2025 / "Attention Sinks as Internal
# Signals"). For head (l,h), the sink score of token i is the mean attention it
# receives from all tokens u >= i:  s_i = (1/(T-i)) * sum_{u>=i} A[u, i].
# ---------------------------------------------------------------------------- #


def sink_scores_for_layer(attn_layer: torch.Tensor) -> torch.Tensor:
    """attn_layer: [heads, seq, seq] -> per-head sink scores [heads, seq]."""
    heads, seq, _ = attn_layer.shape
    # lower-triangular incl diagonal: keep A[u, i] for u >= i
    tri = torch.tril(torch.ones(seq, seq, dtype=attn_layer.dtype))
    masked = attn_layer * tri.unsqueeze(0)          # zero out u < i
    colsum = masked.sum(dim=1)                       # sum over u (rows) -> [heads, seq]
    denom = (seq - torch.arange(seq)).clamp(min=1).to(attn_layer.dtype)  # T - i
    return colsum / denom.unsqueeze(0)


def sink_scores(cap: ForwardCapture) -> dict[str, torch.Tensor]:
    """Compute per-layer sink scores.

    Returns dict with:
      * ``per_head``: [n_layers, heads, seq]
      * ``mean_over_heads``: [n_layers, seq]   (primary scalar feature per layer)
      * ``max_over_heads``: [n_layers, seq]
    """
    per_head = torch.stack([sink_scores_for_layer(a) for a in cap.attentions], dim=0)
    return {
        "per_head": per_head,
        "mean_over_heads": per_head.mean(dim=1),
        "max_over_heads": per_head.amax(dim=1),
    }


# ---------------------------------------------------------------------------- #
# Span detection (mirrors model_lens, parameterised by TemplateInfo)
# ---------------------------------------------------------------------------- #


@dataclass
class Span:
    start: int
    ids: list[int]
    kind: str


def find_literal_assistant_spans(ids: list[int], tpl: TemplateInfo) -> list[Span]:
    w = len(tpl.header_ids)
    spans = []
    for pos in range(0, max(0, len(ids) - w + 1)):
        if ids[pos : pos + w] == tpl.header_ids:
            spans.append(Span(pos, ids[pos : pos + w], "literal_assistant_header"))
    return spans


def find_regular_assistant_spans(ids: list[int], tpl: TemplateInfo) -> list[Span]:
    w = len(tpl.header_ids)
    spans = []
    for pos in range(0, max(0, len(ids) - w + 1)):
        window = ids[pos : pos + w]
        if any(window[fp] != fid for fp, fid in tpl.fixed_ids_by_pos.items()):
            continue
        if any(int(window[rp]) in tpl.special_token_ids for rp in tpl.replace_positions):
            continue
        # require that at least one replace position is NOT the literal fixed/header id,
        # i.e. it really is a mimicry of a special slot (avoids matching ordinary text
        # that merely contains 'assistant' + '\n\n').
        spans.append(Span(pos, window, "regular_assistant_header"))
    return spans


def template_prefix_suffix_lengths(lm: LoadedModel) -> tuple[int, int]:
    """Token lengths of the fixed template prefix (BOS+user header) and suffix
    (final generation-prompt assistant header) that wrap the user content."""
    tok = lm.tokenizer
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": _SENTINEL}],
        add_generation_prompt=True,
        tokenize=False,
    )
    if _SENTINEL not in rendered:
        raise RuntimeError("Sentinel lost while rendering chat template.")
    prefix_str, suffix_str = rendered.split(_SENTINEL, 1)
    prefix_ids = tok(prefix_str, add_special_tokens=False)["input_ids"]
    suffix_ids = tok(suffix_str, add_special_tokens=False)["input_ids"]
    return len(prefix_ids), len(suffix_ids)


# ---------------------------------------------------------------------------- #
# Token-level category labelling
# ---------------------------------------------------------------------------- #


def label_token_categories(
    ids: list[int],
    tpl: TemplateInfo,
    prefix_len: int,
    suffix_len: int,
    variant: str,
) -> dict[int, str]:
    """Return {position: category} for the labelled positions of interest.

    Positions not returned are treated as ordinary regular tokens (E) unless
    explicitly added by the caller.

    Region rule (robust, content-agnostic):
      * positions in the template prefix  -> system specials (D) if special id
      * positions in the template suffix   -> system specials (D) if special id
      * positions inside user content      -> A/B/C depending on variant & spans
    """
    n = len(ids)
    content_lo = prefix_len
    content_hi = n - suffix_len
    labels: dict[int, str] = {}

    # D: every special-token id that lives in the fixed template prefix/suffix.
    for p, tid in enumerate(ids):
        in_template = p < content_lo or p >= content_hi
        if in_template and int(tid) in tpl.special_token_ids:
            labels[p] = CAT_SYSTEM

    if variant == "mimicked":
        # A: replace-position offsets inside mimicked regular-assistant-header spans
        #    that fall within the content region.
        for span in find_regular_assistant_spans(ids, tpl):
            if not (content_lo <= span.start and span.start + len(span.ids) <= content_hi):
                continue
            for off in tpl.replace_positions:
                labels[span.start + off] = CAT_MIMICRY

    elif variant == "malicious":
        # B: special-token positions of literal-assistant-header spans inside content.
        #    (The final genuine generation-prompt header is in the suffix region and
        #     was already labelled D above, so it is naturally excluded.)
        for span in find_literal_assistant_spans(ids, tpl):
            if not (content_lo <= span.start and span.start + len(span.ids) <= content_hi):
                continue
            for off in tpl.replace_positions:
                labels[span.start + off] = CAT_MALICIOUS

    elif variant == "benign_special":
        # C: any special-token id inside the content region that is NOT part of a
        #    full literal assistant-header span (those would be attacks).
        attack_positions: set[int] = set()
        for span in find_literal_assistant_spans(ids, tpl):
            if content_lo <= span.start and span.start + len(span.ids) <= content_hi:
                for off in tpl.replace_positions:
                    attack_positions.add(span.start + off)
        for p in range(content_lo, content_hi):
            if int(ids[p]) in tpl.special_token_ids and p not in attack_positions:
                labels[p] = CAT_BENIGN_SPECIAL

    elif variant == "ordinary":
        pass  # only D specials labelled; ordinary tokens added by caller if needed

    return labels


def sample_ordinary_positions(
    ids: list[int],
    tpl: TemplateInfo,
    prefix_len: int,
    suffix_len: int,
    max_positions: int | None = 6,
) -> list[int]:
    """Ordinary (non-special) content positions to seed the E baseline.

    ``max_positions=None`` (or a negative value) returns *every* ordinary content
    position — used by the Tier-2 honest study to score the full sequence rather
    than a sparse 4-6 token sample.
    """
    content_lo, content_hi = prefix_len, len(ids) - suffix_len
    cand = [
        p
        for p in range(content_lo, content_hi)
        if int(ids[p]) not in tpl.special_token_ids
    ]
    if not cand:
        return []
    if max_positions is None or max_positions < 0 or len(cand) <= max_positions:
        return cand
    step = len(cand) / max_positions
    return [cand[int(i * step)] for i in range(max_positions)]


# ---------------------------------------------------------------------------- #
# IO helpers
# ---------------------------------------------------------------------------- #


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_replacement(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
