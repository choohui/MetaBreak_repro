"""Inspect internal hidden/attention behavior of MetaBreak mimicry spans."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.model_configs import resolve_config  # noqa: E402


TARGET_IDS: dict[str, int] = {}
ROLE_OFFSETS: dict[str, int] = {}
LITERAL_HEADER_IDS: list[int] = []
FIXED_IDS_BY_POS: dict[int, int] = {}
REPLACE_POSITIONS: list[int] = []
SPECIAL_TOKEN_IDS: set[int] = set()


@dataclass
class Span:
    start: int
    ids: list[int]
    kind: str


def role_name_from_token_str(token_str: str, pos: int) -> str:
    known = {
        "<|eot_id|>": "eot",
        "<|start_header_id|>": "start_header",
        "<|end_header_id|>": "end_header",
        "<|im_start|>": "im_start",
        "<|im_end|>": "im_end",
        "<|im_sep|>": "im_sep",
        "<start_of_turn>": "start_of_turn",
        "<end_of_turn>": "end_of_turn",
    }
    if token_str in known:
        return known[token_str]
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", token_str).strip("_").lower()
    return cleaned or f"target_{pos}"


def configure_template(tokenizer: Any, model_type: str) -> dict[str, Any]:
    """Load the chat-boundary template from src/model_configs.py.

    The lens code then detects spans by the resolved literal assistant header
    shape instead of carrying a second hard-coded copy of the Llama IDs.
    """
    global TARGET_IDS, ROLE_OFFSETS, LITERAL_HEADER_IDS
    global FIXED_IDS_BY_POS, REPLACE_POSITIONS, SPECIAL_TOKEN_IDS

    cfg = resolve_config(model_type, tokenizer)
    literal_ids = tokenizer(cfg.assistant_header, add_special_tokens=False)["input_ids"]
    if len(literal_ids) != cfg.expected_n_tokens:
        raise RuntimeError(
            "Resolved assistant_header length changed between model_configs and "
            "local tokenization."
        )

    target_ids: dict[str, int] = {}
    role_offsets: dict[str, int] = {}
    for pos, token_id, token_str in zip(
        cfg.replace_positions,
        cfg.target_token_ids,
        cfg.target_token_strs,
    ):
        role = role_name_from_token_str(token_str, pos)
        target_ids[role] = int(token_id)
        role_offsets[role] = int(pos)

    TARGET_IDS = target_ids
    ROLE_OFFSETS = role_offsets
    LITERAL_HEADER_IDS = [int(x) for x in literal_ids]
    FIXED_IDS_BY_POS = {int(pos): int(literal_ids[pos]) for pos in cfg.fixed_positions}
    REPLACE_POSITIONS = [int(pos) for pos in cfg.replace_positions]
    SPECIAL_TOKEN_IDS = {int(x) for x in cfg.special_token_ids}

    return {
        "model_type": cfg.model_type,
        "assistant_header": cfg.assistant_header,
        "literal_header_ids": LITERAL_HEADER_IDS,
        "target_ids": TARGET_IDS,
        "target_token_strs": cfg.target_token_strs,
        "fixed_ids_by_pos": FIXED_IDS_BY_POS,
        "replace_positions": REPLACE_POSITIONS,
        "role_offsets": ROLE_OFFSETS,
        "auto_detected": cfg.auto_detected,
    }


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_inputs(tokenizer: Any, user_content: str, device: str) -> torch.Tensor:
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    )
    if not isinstance(out, torch.Tensor):
        out = out["input_ids"]
    return out.to(device)


def find_literal_assistant_spans(input_ids: list[int]) -> list[Span]:
    width = len(LITERAL_HEADER_IDS)
    spans = []
    for pos in range(0, max(0, len(input_ids) - width + 1)):
        if input_ids[pos : pos + width] == LITERAL_HEADER_IDS:
            spans.append(Span(pos, input_ids[pos : pos + width], "literal_assistant_header"))
    return spans


def find_regular_assistant_spans(input_ids: list[int]) -> list[Span]:
    width = len(LITERAL_HEADER_IDS)
    spans = []
    for pos in range(0, max(0, len(input_ids) - width + 1)):
        span = input_ids[pos : pos + width]
        if any(span[fixed_pos] != fixed_id for fixed_pos, fixed_id in FIXED_IDS_BY_POS.items()):
            continue
        if any(int(span[replace_pos]) in SPECIAL_TOKEN_IDS for replace_pos in REPLACE_POSITIONS):
            continue
        spans.append(Span(pos, span, "regular_assistant_header"))
    return spans


def keep_injected_literal_spans(spans: list[Span], n_expected: int) -> list[Span]:
    # The final generation prompt is a real assistant header appended by the
    # chat template. The attack-controlled literal spans occur before it.
    return spans[:n_expected]


def tensor_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float(), b.float(), dim=0).item())


def tensor_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a.float() - b.float()).item())


def attention_mass_to_span(
    attn_layer: torch.Tensor,
    span: Span,
    *,
    query_window: int,
) -> float | None:
    # attn_layer: [heads, seq, seq]
    seq_len = int(attn_layer.shape[-1])
    q_start = span.start + len(span.ids)
    q_end = min(seq_len, q_start + query_window)
    if q_start >= q_end:
        return None
    span_positions = torch.arange(span.start, span.start + len(span.ids), device=attn_layer.device)
    query_positions = torch.arange(q_start, q_end, device=attn_layer.device)
    mass = attn_layer[:, query_positions][:, :, span_positions].sum(dim=-1)
    return float(mass.mean().float().item())


def forward_lens(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    device: str,
) -> dict[str, Any]:
    input_ids = build_inputs(tokenizer, text, device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    ids = input_ids[0].detach().cpu().tolist()
    hidden = [h[0].detach().cpu() for h in out.hidden_states]
    attentions = [a[0].detach().cpu() for a in out.attentions]
    return {
        "input_ids": ids,
        "hidden_states": hidden,
        "attentions": attentions,
    }


def safe_mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def analyze_pair(
    *,
    idx: int,
    original: str,
    mimicked: str,
    model: Any,
    tokenizer: Any,
    embedding: torch.Tensor,
    device: str,
    query_window: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mim = forward_lens(model, tokenizer, mimicked, device=device)
    base = forward_lens(model, tokenizer, original, device=device)

    mim_spans = find_regular_assistant_spans(mim["input_ids"])
    base_spans_all = find_literal_assistant_spans(base["input_ids"])
    base_spans = keep_injected_literal_spans(base_spans_all, len(mim_spans))
    n_aligned = min(len(mim_spans), len(base_spans))

    layer_rows: list[dict[str, Any]] = []
    for layer_idx, (mim_h, base_h) in enumerate(zip(mim["hidden_states"], base["hidden_states"])):
        role_cos: dict[str, list[float]] = {role: [] for role in ROLE_OFFSETS}
        role_l2: dict[str, list[float]] = {role: [] for role in ROLE_OFFSETS}
        role_special_margin: dict[str, list[float]] = {role: [] for role in ROLE_OFFSETS}
        role_hidden_norm: dict[str, list[float]] = {role: [] for role in ROLE_OFFSETS}

        for span_idx in range(n_aligned):
            mim_span = mim_spans[span_idx]
            base_span = base_spans[span_idx]
            for role, offset in ROLE_OFFSETS.items():
                mim_pos = mim_span.start + offset
                base_pos = base_span.start + offset
                mim_vec = mim_h[mim_pos]
                base_vec = base_h[base_pos]
                target_emb = embedding[TARGET_IDS[role]]
                own_emb = embedding[mim_span.ids[offset]]
                role_cos[role].append(tensor_cos(mim_vec, base_vec))
                role_l2[role].append(tensor_l2(mim_vec, base_vec))
                role_special_margin[role].append(
                    tensor_cos(mim_vec, target_emb) - tensor_cos(mim_vec, own_emb)
                )
                role_hidden_norm[role].append(float(torch.linalg.vector_norm(mim_vec.float()).item()))

        row: dict[str, Any] = {
            "sample_idx": idx,
            "layer": layer_idx,
            "n_mimicked_spans": len(mim_spans),
            "n_literal_spans_total": len(base_spans_all),
            "n_aligned_spans": n_aligned,
        }
        for role in ROLE_OFFSETS:
            row[f"{role}_cos_to_literal_hidden_mean"] = safe_mean(role_cos[role])
            row[f"{role}_l2_to_literal_hidden_mean"] = safe_mean(role_l2[role])
            row[f"{role}_special_vs_own_embedding_cos_margin_mean"] = safe_mean(role_special_margin[role])
            row[f"{role}_hidden_norm_mean"] = safe_mean(role_hidden_norm[role])

        if layer_idx > 0:
            attn_idx = layer_idx - 1
            mim_attn = mim["attentions"][attn_idx]
            base_attn = base["attentions"][attn_idx]
            mim_masses = [
                attention_mass_to_span(mim_attn, span, query_window=query_window)
                for span in mim_spans
            ]
            base_masses = [
                attention_mass_to_span(base_attn, span, query_window=query_window)
                for span in base_spans
            ]
            row["mimicked_post_span_attention_mass_mean"] = safe_mean(mim_masses)
            row["literal_post_span_attention_mass_mean"] = safe_mean(base_masses)
            if row["mimicked_post_span_attention_mass_mean"] is not None and row["literal_post_span_attention_mass_mean"] is not None:
                row["post_span_attention_mass_delta_mim_minus_literal"] = round(
                    row["mimicked_post_span_attention_mass_mean"]
                    - row["literal_post_span_attention_mass_mean"],
                    6,
                )
            else:
                row["post_span_attention_mass_delta_mim_minus_literal"] = None
        else:
            row["mimicked_post_span_attention_mass_mean"] = None
            row["literal_post_span_attention_mass_mean"] = None
            row["post_span_attention_mass_delta_mim_minus_literal"] = None

        layer_rows.append(row)

    sample_summary = {
        "idx": idx,
        "mimicked_input_len": len(mim["input_ids"]),
        "baseline_input_len": len(base["input_ids"]),
        "mimicked_spans": [
            {
                "start": span.start,
                "ids": span.ids,
                "decoded": tokenizer.decode(span.ids),
            }
            for span in mim_spans
        ],
        "literal_spans_used": [
            {
                "start": span.start,
                "ids": span.ids,
                "decoded": tokenizer.decode(span.ids),
            }
            for span in base_spans
        ],
        "literal_spans_total": len(base_spans_all),
        "n_aligned_spans": n_aligned,
    }
    return layer_rows, sample_summary


def analyze_benign(
    *,
    row: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: str,
    query_window: int,
) -> dict[str, Any]:
    lens = forward_lens(model, tokenizer, row["text"], device=device)
    regular_spans = find_regular_assistant_spans(lens["input_ids"])
    literal_spans = find_literal_assistant_spans(lens["input_ids"])
    per_layer = []
    for layer_idx, attn in enumerate(lens["attentions"], start=1):
        masses = [
            attention_mass_to_span(attn, span, query_window=query_window)
            for span in regular_spans
        ]
        per_layer.append(
            {
                "layer": layer_idx,
                "regular_post_span_attention_mass_mean": safe_mean(masses),
            }
        )
    return {
        "id": row.get("id"),
        "category": row.get("category"),
        "input_len": len(lens["input_ids"]),
        "n_regular_assistant_spans": len(regular_spans),
        "n_literal_assistant_spans": len(literal_spans),
        "regular_spans": [
            {
                "start": span.start,
                "ids": span.ids,
                "decoded": tokenizer.decode(span.ids),
            }
            for span in regular_spans
        ],
        "attention_by_layer": per_layer,
    }


def aggregate_layer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    aggregate = []
    metric_keys = [
        key
        for key in rows[0]
        if key not in {"sample_idx", "layer"}
        and isinstance(rows[0].get(key), (int, float, type(None)))
    ]
    for layer in sorted(by_layer):
        group = by_layer[layer]
        out = {"layer": layer, "n_samples": len(group)}
        for key in metric_keys:
            vals = [g.get(key) for g in group]
            if all(v is None for v in vals):
                out[key] = None
            else:
                numeric = [float(v) for v in vals if v is not None]
                out[key] = round(sum(numeric) / len(numeric), 6) if numeric else None
        aggregate.append(out)
    return aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_type", default="llama")
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--benign_prompts", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--n_benign", type=int, default=5)
    p.add_argument("--query_window", type=int, default=3)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    template_metadata = configure_template(tokenizer, args.model_type)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    embedding = model.get_input_embeddings().weight.detach().cpu().float()

    attack_rows = read_jsonl(Path(args.prompts), limit=args.n)
    all_layer_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for row in attack_rows:
        layer_rows, sample_summary = analyze_pair(
            idx=int(row["idx"]),
            original=row["original"],
            mimicked=row["mimicked"],
            model=model,
            tokenizer=tokenizer,
            embedding=embedding,
            device=device,
            query_window=args.query_window,
        )
        all_layer_rows.extend(layer_rows)
        sample_rows.append(sample_summary)

    aggregate = aggregate_layer_rows(all_layer_rows)
    write_jsonl(out_dir / "sample_spans.jsonl", sample_rows)
    write_jsonl(out_dir / "layer_metrics_per_sample.jsonl", all_layer_rows)
    write_json(out_dir / "layer_metrics_summary.json", aggregate)
    write_csv(out_dir / "layer_metrics_summary.csv", aggregate)

    benign_summaries: list[dict[str, Any]] = []
    if args.benign_prompts:
        benign_rows = read_jsonl(Path(args.benign_prompts), limit=args.n_benign)
        for row in benign_rows:
            benign_summaries.append(
                analyze_benign(
                    row=row,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    query_window=args.query_window,
                )
            )
        write_jsonl(out_dir / "benign_lens.jsonl", benign_summaries)

    summary = {
        "model": args.model,
        "model_type": args.model_type,
        "template": template_metadata,
        "n_attack_samples": len(attack_rows),
        "n_benign_samples": len(benign_summaries),
        "query_window": args.query_window,
        "n_layers_including_embedding": len(aggregate),
        "files": {
            "sample_spans": str(out_dir / "sample_spans.jsonl"),
            "layer_metrics_per_sample": str(out_dir / "layer_metrics_per_sample.jsonl"),
            "layer_metrics_summary_json": str(out_dir / "layer_metrics_summary.json"),
            "layer_metrics_summary_csv": str(out_dir / "layer_metrics_summary.csv"),
            "benign_lens": str(out_dir / "benign_lens.jsonl") if benign_summaries else None,
        },
        "last_layer": aggregate[-1] if aggregate else None,
        "mid_layer": aggregate[len(aggregate) // 2] if aggregate else None,
        "first_layer": aggregate[0] if aggregate else None,
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
