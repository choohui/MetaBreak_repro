"""Token-level L2 guard for MetaBreak semantic mimicry prompts.

The MetaBreak TM-1 attack replaces Llama chat-template special tokens with
regular tokens that are close in the input-embedding table. This module detects
that attack after tokenization by scanning for the assistant-header shape whose
regular tokens are embedding-near the target special tokens.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_IDS = {
    "eot": 128009,
    "start_header": 128006,
    "end_header": 128007,
}
ASSISTANT_ID = 78191
DOUBLE_NEWLINE_ID = 271
LLAMA3_SPECIAL_START = 128000

DEFAULT_REFUSAL = (
    "I can't assist with requests that try to override the chat role structure "
    "or bypass safety protections."
)


@dataclass(frozen=True)
class DistanceThresholds:
    eot: float
    start_header: float
    end_header: float

    def as_dict(self) -> dict[str, float]:
        return {
            "eot": self.eot,
            "start_header": self.start_header,
            "end_header": self.end_header,
        }


class L2MimicryGuard:
    """Detect regular-token impersonations of Llama chat special tokens."""

    def __init__(
        self,
        tokenizer: Any,
        embedding_weights: torch.Tensor,
        *,
        neighbor_rank: int = 256,
        threshold_margin: float = 0.0,
        structural_min_spans: int = 2,
        special_start: int = LLAMA3_SPECIAL_START,
    ) -> None:
        if neighbor_rank < 1:
            raise ValueError("neighbor_rank must be >= 1")

        self.tokenizer = tokenizer
        self.embedding_weights = embedding_weights.detach().to(torch.float32).cpu()
        self.neighbor_rank = int(neighbor_rank)
        self.threshold_margin = float(threshold_margin)
        self.structural_min_spans = int(structural_min_spans)
        self.special_start = int(special_start)

        self._validate_tokenizer()
        self.thresholds, self.nearest_neighbors = self._calibrate_thresholds()

    @classmethod
    def from_model(
        cls,
        tokenizer: Any,
        model: Any,
        *,
        neighbor_rank: int = 256,
        threshold_margin: float = 0.0,
        structural_min_spans: int = 2,
        special_start: int = LLAMA3_SPECIAL_START,
    ) -> "L2MimicryGuard":
        return cls(
            tokenizer,
            model.get_input_embeddings().weight,
            neighbor_rank=neighbor_rank,
            threshold_margin=threshold_margin,
            structural_min_spans=structural_min_spans,
            special_start=special_start,
        )

    def _validate_tokenizer(self) -> None:
        expected = {
            TARGET_IDS["eot"]: "<|eot_id|>",
            TARGET_IDS["start_header"]: "<|start_header_id|>",
            TARGET_IDS["end_header"]: "<|end_header_id|>",
            ASSISTANT_ID: "assistant",
        }
        for token_id, want in expected.items():
            if token_id == ASSISTANT_ID:
                got = self.tokenizer.decode([token_id])
            else:
                got = self.tokenizer.convert_ids_to_tokens(token_id)
            if got != want:
                raise RuntimeError(
                    f"Unexpected tokenizer mapping for id={token_id}: {got!r} "
                    f"(expected {want!r}). This guard expects a Llama-3.x "
                    "instruct tokenizer."
                )
        got_newline = self.tokenizer.decode([DOUBLE_NEWLINE_ID])
        if got_newline != "\n\n":
            raise RuntimeError(
                f"Unexpected tokenizer mapping for id={DOUBLE_NEWLINE_ID}: "
                f"{got_newline!r} (expected '\\n\\n')."
            )

    def _regular_mask(self) -> torch.Tensor:
        n_vocab = int(self.embedding_weights.shape[0])
        mask = torch.ones(n_vocab, dtype=torch.bool)
        mask[self.special_start :] = False
        for token_id in set(getattr(self.tokenizer, "all_special_ids", [])):
            if 0 <= token_id < n_vocab:
                mask[token_id] = False
        return mask

    def _calibrate_one(self, target_id: int, regular_mask: torch.Tensor) -> tuple[float, list[dict[str, Any]]]:
        target = self.embedding_weights[target_id].unsqueeze(0)
        distances = torch.linalg.vector_norm(self.embedding_weights - target, dim=1)
        distances = distances.masked_fill(~regular_mask, float("inf"))
        k = min(self.neighbor_rank, int(regular_mask.sum().item()))
        values, indices = torch.topk(distances, k=k, largest=False)
        threshold = float(values[-1].item() + self.threshold_margin)
        preview = []
        for value, idx in zip(values[:10].tolist(), indices[:10].tolist()):
            preview.append(
                {
                    "token_id": int(idx),
                    "token": self.tokenizer.convert_ids_to_tokens(int(idx)),
                    "decoded": self.tokenizer.decode([int(idx)]),
                    "distance_l2": float(value),
                }
            )
        return threshold, preview

    def _calibrate_thresholds(self) -> tuple[DistanceThresholds, dict[str, list[dict[str, Any]]]]:
        regular_mask = self._regular_mask()
        thresholds: dict[str, float] = {}
        neighbors: dict[str, list[dict[str, Any]]] = {}
        for name, target_id in TARGET_IDS.items():
            thresholds[name], neighbors[name] = self._calibrate_one(target_id, regular_mask)
        return (
            DistanceThresholds(
                eot=thresholds["eot"],
                start_header=thresholds["start_header"],
                end_header=thresholds["end_header"],
            ),
            neighbors,
        )

    def _distance(self, token_id: int, target_id: int) -> float:
        return float(
            torch.linalg.vector_norm(
                self.embedding_weights[int(token_id)]
                - self.embedding_weights[int(target_id)]
            ).item()
        )

    def inspect_text(self, text: str) -> dict[str, Any]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return self.inspect_ids(ids)

    def inspect_ids(self, input_ids: list[int]) -> dict[str, Any]:
        detections: list[dict[str, Any]] = []
        regular_header_spans: list[dict[str, Any]] = []
        literal_ids = set(TARGET_IDS.values())

        for pos, token_id in enumerate(input_ids):
            if token_id in literal_ids:
                detections.append(
                    {
                        "kind": "literal_special_id",
                        "position": pos,
                        "token_id": int(token_id),
                        "token": self.tokenizer.convert_ids_to_tokens(int(token_id)),
                    }
                )

        threshold_dict = self.thresholds.as_dict()
        for pos in range(0, max(0, len(input_ids) - 4)):
            span = input_ids[pos : pos + 5]
            if span[2] != ASSISTANT_ID or span[4] != DOUBLE_NEWLINE_ID:
                continue
            if any(token_id >= self.special_start for token_id in (span[0], span[1], span[3])):
                continue

            d_eot = self._distance(span[0], TARGET_IDS["eot"])
            d_start = self._distance(span[1], TARGET_IDS["start_header"])
            d_end = self._distance(span[3], TARGET_IDS["end_header"])
            span_info = {
                "position": pos,
                "span_token_ids": [int(x) for x in span],
                "span_decoded": self.tokenizer.decode(span),
                "distances_l2": {
                    "eot": d_eot,
                    "start_header": d_start,
                    "end_header": d_end,
                    "sum": d_eot + d_start + d_end,
                },
                "thresholds_l2": threshold_dict,
            }
            regular_header_spans.append(span_info)
            matched = (
                d_eot <= self.thresholds.eot
                and d_start <= self.thresholds.start_header
                and d_end <= self.thresholds.end_header
            )
            if matched:
                detections.append(
                    {
                        "kind": "l2_mimicry_assistant_header",
                        **span_info,
                    }
                )

        if self.structural_min_spans > 0 and regular_header_spans:
            signatures = [
                tuple(span_info["span_token_ids"])
                for span_info in regular_header_spans
            ]
            counts = Counter(signatures)
            repeated = {
                signature: count
                for signature, count in counts.items()
                if count >= self.structural_min_spans
            }
            if repeated:
                repeated_set = set(repeated)
                repeated_spans = [
                    span_info
                    for span_info in regular_header_spans
                    if tuple(span_info["span_token_ids"]) in repeated_set
                ]
                detections.append(
                    {
                        "kind": "regular_assistant_header_pattern",
                        "n_spans": len(regular_header_spans),
                        "min_required_repetitions": self.structural_min_spans,
                        "repeated_span_counts": {
                            json.dumps(list(signature)): count
                            for signature, count in repeated.items()
                        },
                        "spans": repeated_spans[:10],
                    }
                )

        blocked = bool(detections)
        reasons = sorted({d["kind"] for d in detections})
        return {
            "blocked": blocked,
            "reason": ",".join(reasons) if reasons else None,
            "n_input_tokens": len(input_ids),
            "n_detections": len(detections),
            "detections": detections,
            "neighbor_rank": self.neighbor_rank,
            "threshold_margin": self.threshold_margin,
            "structural_min_spans": self.structural_min_spans,
            "thresholds_l2": threshold_dict,
        }

    def defended_response(self, text: str) -> tuple[str | None, dict[str, Any]]:
        decision = self.inspect_text(text)
        if decision["blocked"]:
            return DEFAULT_REFUSAL, decision
        return None, decision

    def metadata(self) -> dict[str, Any]:
        return {
            "target_ids": TARGET_IDS,
            "assistant_id": ASSISTANT_ID,
            "double_newline_id": DOUBLE_NEWLINE_ID,
            "special_start": self.special_start,
            "neighbor_rank": self.neighbor_rank,
            "threshold_margin": self.threshold_margin,
            "structural_min_spans": self.structural_min_spans,
            "thresholds_l2": self.thresholds.as_dict(),
            "nearest_neighbors_preview": self.nearest_neighbors,
        }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local Llama-3.x HF model path.")
    p.add_argument("--prompts", help="Optional JSONL prompt file to inspect.")
    p.add_argument("--text_field", default="mimicked")
    p.add_argument("--output", help="Where to write inspection JSON.")
    p.add_argument("--neighbor_rank", type=int, default=256)
    p.add_argument("--threshold_margin", type=float, default=0.0)
    p.add_argument("--structural_min_spans", type=int, default=2)
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
    )
    guard = L2MimicryGuard.from_model(
        tokenizer,
        model,
        neighbor_rank=args.neighbor_rank,
        threshold_margin=args.threshold_margin,
        structural_min_spans=args.structural_min_spans,
    )
    out: dict[str, Any] = {"guard": guard.metadata()}

    if args.prompts:
        rows = _load_jsonl(Path(args.prompts))
        inspections = []
        for row in rows:
            text = row[args.text_field]
            inspections.append({"idx": row.get("idx"), "inspection": guard.inspect_text(text)})
        out["n_rows"] = len(rows)
        out["n_blocked"] = sum(1 for row in inspections if row["inspection"]["blocked"])
        out["inspections"] = inspections

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[l2_guard] wrote {out_path}")
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
