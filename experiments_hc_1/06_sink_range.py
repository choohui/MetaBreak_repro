"""Stage 06 (Main.md §3) — narrow the sink-examined token range, then re-threshold.

Reduces the set of positions a sink-based detector must look at, then redoes the
single-threshold TPR/FPR + ASR analysis on that reduced set. Reuses the SAME
stage-03 artifacts as stage 05 (no model re-run), so §2 and §3 run together.

Modes (``--sink_range_mode``):
  header_slots : keep only attack-header-slot positions {B, D, F} (pos_offset 0)
                 -> tests attack(B,D) vs benign-in-slot(F) once the body is excluded.
  topk         : per prompt keep the top-k positions by max-over-layers sink
                 -> tests POS=B u D vs whatever benign tokens are most sink-like.
  sink_pct     : per prompt keep the top ``--sink_keep_pct`` percent of positions by
                 max-over-layers sink (A included) -> a sink-only 1st-stage gate that
                 drops ~(100-pct)% of tokens before the value/hidden threshold.

Outputs (under ``out_dir/pos{offset}/``):
    sink_range_report.json / .md            (header_slots, backward-compatible name)
    sink_range_report__{mode}.json / .md    (other modes, e.g. ...__sink_pct)
"""

from __future__ import annotations

import sys
from pathlib import Path

import math

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):  # repo root makes `experiments_hc_1` importable
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_1.config import ExpConfig, config_from_args, make_stage_parser  # noqa: E402
from experiments_hc_1.core import io  # noqa: E402
from experiments_hc_1.core.labels import CAT_B, CAT_D, CAT_F  # noqa: E402
import experiments_hc_1.analysis_common as ac  # noqa: E402


def _per_prompt_by_sink(rows) -> dict:
    """Group rows by prompt, each list sorted by max-over-layers sink (desc)."""
    by_sample: dict[int, list[dict]] = {}
    for r in rows:
        by_sample.setdefault(r["sample_index"], []).append(r)
    for rs in by_sample.values():
        rs.sort(key=lambda r: max(r["sink"]), reverse=True)
    return by_sample


def _reduce(rows, mode: str, topk: int, keep_pct: float) -> list[dict]:
    if mode == "header_slots":
        return [r for r in rows if r["category"] in (CAT_B, CAT_D, CAT_F)]
    if mode == "topk":
        # rank within each prompt by max-over-layers sink, keep top-k
        kept: list[dict] = []
        for rs in _per_prompt_by_sink(rows).values():
            kept.extend(rs[:topk])
        return kept
    if mode == "sink_pct":
        # 1st-stage gate: keep the top keep_pct% of each prompt's tokens by sink
        # (A included), dropping ~(100-keep_pct)% before the value/hidden test.
        kept = []
        for rs in _per_prompt_by_sink(rows).values():
            k = max(1, math.ceil(len(rs) * keep_pct / 100.0))
            kept.extend(rs[:k])
        return kept
    raise ValueError(f"unknown sink_range_mode {mode!r}")


def _analyze_offset(cfg, rows_all, hidden, success, offset) -> dict:
    rows_off = [r for r in rows_all if r["pos_offset"] == offset]
    if not rows_off:
        return {}
    reduced = _reduce(rows_off, cfg.sink_range_mode, cfg.sink_range_topk, cfg.sink_keep_pct)
    if not reduced:
        print(f"[06] pos{offset}: reduced set empty, skipping")
        return {}

    # A centroid from the full offset pool (reduced set may lack A rows).
    a_centroids = ac.reference_centroids(rows_off, hidden)
    signals = ac.build_signals(reduced, hidden, a_centroids=a_centroids)

    y = ac.binary_labels(reduced)
    evals = ac.evaluate_signals(signals, y)
    per_type = ac.per_type_breakdown(signals, reduced, evals)

    y_asr = ac.binary_labels(reduced, success=success)
    evals_asr = ac.evaluate_signals(signals, y_asr)

    report = {
        "pos_offset": offset,
        "mode": cfg.sink_range_mode,
        "topk": cfg.sink_range_topk if cfg.sink_range_mode == "topk" else None,
        "keep_pct": cfg.sink_keep_pct if cfg.sink_range_mode == "sink_pct" else None,
        "n_full": len(rows_off),
        "n_reduced": len(reduced),
        "reduction_ratio": round(len(reduced) / max(1, len(rows_off)), 4),
        "best_per_signal": ac.best_summary(evals),
        "best_per_signal_asr": ac.best_summary(evals_asr),
        "per_signal": evals,
        "per_signal_asr": evals_asr,
        "per_type": per_type,
    }
    pos_dir = cfg.pos_dir(offset)
    stem = _report_stem(cfg.sink_range_mode)
    io.write_json(pos_dir / f"{stem}.json", report)
    io.write_text(pos_dir / f"{stem}.md", _md(report))
    print(f"[06] pos{offset}: mode={cfg.sink_range_mode} "
          f"reduced {len(rows_off)}->{len(reduced)}; best={ac.best_summary(evals)}")
    return report


def _report_stem(mode: str) -> str:
    """header_slots keeps the original filename; other modes get a suffix so they
    coexist (and don't clobber the Fig-7 source)."""
    return "sink_range_report" if mode == "header_slots" else f"sink_range_report__{mode}"


def _md(r: dict) -> str:
    L = [f"# Stage 06 — sink-range reduction (pos_offset={r['pos_offset']})", ""]
    L.append(f"- mode: **{r['mode']}**  (topk={r['topk']}, keep_pct={r.get('keep_pct')})")
    L.append(f"- reduced token set: {r['n_reduced']} / {r['n_full']} "
             f"(ratio {r['reduction_ratio']})")
    L.append("")
    L.append("## Per-signal best layer on the reduced set")
    L.append("| signal | best layer | AUC | AUC (ASR) |")
    L.append("|---|---|---|---|")
    asr_by = {d["signal"]: d for d in r["best_per_signal_asr"]}
    for d in r["best_per_signal"]:
        a = asr_by.get(d["signal"], {})
        L.append(f"| {d['signal']} | {d['best_layer']} | {d['best_auc']} | "
                 f"{a.get('best_auc')} |")
    return "\n".join(L) + "\n"


def run(cfg: ExpConfig, lm=None) -> dict:  # lm unused (model-free stage)
    rows_all, hidden, success = ac.load_artifacts(cfg.out_dir)
    out = {}
    for off in cfg.pos_offsets:
        out[f"pos{off}"] = _analyze_offset(cfg, rows_all, hidden, success, off)
    return out


def main() -> None:
    args = make_stage_parser(__doc__).parse_args()
    run(config_from_args(args))


if __name__ == "__main__":
    main()
