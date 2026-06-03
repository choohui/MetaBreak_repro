"""Stage 05 (Main.md §2.3) — single-threshold defense feasibility per signal.

For each ``pos_offset`` and each of the 5 measurement signals (sink, hidden_norm,
value_norm, output_norm, cos_to_ref), per layer: ROC-AUC, Youden threshold and
TPR@FPR{1%,5%}. Two views:
  (a) per-type   : POS=B u D vs NEG=C u E u F u G + per-letter flagged-rate table
  (b) ASR-based  : positives restricted to attacks that actually succeeded

Outputs (under ``out_dir/pos{offset}/``):
    threshold_defense.json / .md
    threshold_per_type.csv
    threshold_asr.json
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from config import ExpConfig, config_from_args, make_stage_parser  # noqa: E402
from core import io  # noqa: E402
import analysis_common as ac  # noqa: E402


def _analyze_offset(cfg: ExpConfig, rows_all, hidden, success, offset: int) -> dict:
    rows = [r for r in rows_all if r["pos_offset"] == offset]
    if not rows:
        return {}
    signals = ac.build_signals(rows, hidden)

    # (a) per-type view
    y = ac.binary_labels(rows)
    evals = ac.evaluate_signals(signals, y)
    per_type = ac.per_type_breakdown(signals, rows, evals)

    # (b) ASR-based view
    y_asr = ac.binary_labels(rows, success=success)
    evals_asr = ac.evaluate_signals(signals, y_asr)

    pos_dir = cfg.pos_dir(offset)
    io.write_json(pos_dir / "threshold_defense.json", {
        "pos_offset": offset,
        "labels": "attack(B,D)=1 vs benign(C,E,F,G)=0",
        "n_rows": len(rows),
        "best_per_signal": ac.best_summary(evals),
        "per_signal": evals,
    })
    io.write_csv(pos_dir / "threshold_per_type.csv", per_type,
                 columns=["signal", "layer", "letter", "role", "n", "flagged_rate"])
    io.write_json(pos_dir / "threshold_asr.json", {
        "pos_offset": offset,
        "labels": "attack(B,D) that SUCCEEDED =1 vs benign(C,E,F,G)=0",
        "n_success_prompts": len(success),
        "best_per_signal": ac.best_summary(evals_asr),
        "per_signal": evals_asr,
    })
    io.write_text(pos_dir / "threshold_defense.md",
                  _md(offset, evals, evals_asr, per_type))
    print(f"[05] pos{offset}: best-per-signal={ac.best_summary(evals)}")
    return {"best_per_signal": ac.best_summary(evals)}


def _md(offset, evals, evals_asr, per_type) -> str:
    L = [f"# Stage 05 — single-threshold defense (pos_offset={offset})", ""]
    L.append("## Per-signal best layer (per-type view: B,D vs C,E,F,G)")
    L.append("| signal | best layer | AUC | TPR@1%FPR | TPR@5%FPR |")
    L.append("|---|---|---|---|---|")
    for name, info in evals.items():
        bl = info["best_layer"]
        if bl is None:
            L.append(f"| {name} | - | - | - | - |")
            continue
        m = info["per_layer"][bl]
        ta = m.get("tpr_at_fpr", {})
        L.append(f"| {name} | {bl} | {m.get('auc')} | "
                 f"{ta.get('1pct')} | {ta.get('5pct')} |")
    L.append("")
    L.append("## ASR-based view (succeeded attacks vs benign)")
    L.append("| signal | best layer | AUC |")
    L.append("|---|---|---|")
    for name, info in evals_asr.items():
        bl = info["best_layer"]
        auc = info["per_layer"][bl]["auc"] if bl is not None else None
        L.append(f"| {name} | {bl} | {auc} |")
    L.append("")
    L.append("## Per-type flagged-rate at each signal's Youden threshold")
    L.append("| signal | layer | letter | role | n | flagged |")
    L.append("|---|---|---|---|---|---|")
    for r in per_type:
        L.append(f"| {r['signal']} | {r['layer']} | {r['letter']} | "
                 f"{r['role']} | {r['n']} | {r['flagged_rate']} |")
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
