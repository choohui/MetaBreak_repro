"""Stage 06 (Main.md §3) — does a sink-score 1st-stage filter help?

Sweeps the sink gate over several keep-% values. At each keep-%:
  * **gate diagnostics**: how many tokens survive, and what fraction of the
    attack tokens (B, D recall) and each benign type survive — i.e. does the gate
    keep the attacks while throwing away the body?
  * **post-gate threshold metrics**: re-run the single-threshold analysis on the
    survivors — per-signal AUROC + TPR@1%/5%FPR, per-type flagged rate (esp. the
    E benign-special and C benign-mimicry FPR), and the ASR-based view.

This answers §3 directly: narrowing the token range first should sharpen the
signals (higher AUROC, lower benign FPR) — the evidence behind §4's 1st stage.
Reuses stage-03 artifacts only (no model). Outputs per ``out_dir/pos{offset}/``:
    sink_filter_report.json / .md
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # repro_mb (makes experiments_hc_2 importable)
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_2.config import ExpConfig, config_from_args, make_stage_parser  # noqa: E402
from experiments_hc_2.core import io  # noqa: E402
from experiments_hc_2.core import cascade  # noqa: E402
from experiments_hc_2.core.labels import CAT_A, CAT_TO_LETTER  # noqa: E402
import experiments_hc_2.stages.analysis_common as ac  # noqa: E402

# Min B/D recall the gate must preserve to be "safe" when recommending a keep-%.
_MIN_BD_RECALL = 0.9


def _retained(rows_full: list[dict], reduced: list[dict]) -> dict:
    """Per-category survival through the gate: kept / full + recall."""
    full: dict[str, int] = {}
    kept: dict[str, int] = {}
    for r in rows_full:
        full[r["category"]] = full.get(r["category"], 0) + 1
    for r in reduced:
        kept[r["category"]] = kept.get(r["category"], 0) + 1
    out = {}
    for cat in sorted(full):
        f = full[cat]
        k = kept.get(cat, 0)
        out[CAT_TO_LETTER[cat]] = {"kept": k, "full": f,
                                   "recall": round(k / f, 5) if f else None}
    return out


def _bd_recall(retained: dict) -> dict:
    return {L: retained.get(L, {}).get("recall") for L in ("B", "D")}


def _metrics(rows_sub: list[dict], hidden, success, a_centroids) -> dict:
    """Single-threshold analysis (per-type + ASR view) on a row subset."""
    signals = ac.build_signals(rows_sub, hidden, a_centroids=a_centroids)
    y = ac.binary_labels(rows_sub)
    evals = ac.evaluate_signals(signals, y)
    per_type = ac.per_type_breakdown(signals, rows_sub, evals)
    y_asr = ac.binary_labels(rows_sub, success=success)
    evals_asr = ac.evaluate_signals(signals, y_asr)
    # compact per-type FPR for the key benign controls at each signal's best layer
    fpr: dict[str, dict] = {}
    for row in per_type:
        if row["letter"] in ("C", "E", "F", "G"):
            fpr.setdefault(row["signal"], {})[row["letter"]] = row["flagged_rate"]
    return {
        "best_per_signal": ac.best_summary(evals),
        "best_per_signal_asr": ac.best_summary(evals_asr),
        "benign_fpr_at_best_layer": fpr,
        "per_type": per_type,
    }


def _analyze_offset(cfg: ExpConfig, rows_all, hidden, success, offset: int) -> dict:
    rows_off = [r for r in rows_all if r["pos_offset"] == offset]
    if not rows_off:
        return {}
    # Centroid uses A; the gate/candidate population excludes A (template specials
    # are system-placed and would otherwise consume the gate budget — see §4/H1).
    a_centroids = ac.reference_centroids(rows_off, hidden)
    cand_rows = [r for r in rows_off if r["category"] != CAT_A]
    if not cand_rows:
        return {}

    baseline = _metrics(cand_rows, hidden, success, a_centroids)

    sweep = []
    for pct in cfg.sink_filter_pcts:
        reduced = cascade.sink_gate(cand_rows, pct)
        if not reduced:
            continue
        retained = _retained(cand_rows, reduced)
        sweep.append({
            "keep_pct": pct,
            "n_full": len(cand_rows),
            "n_reduced": len(reduced),
            "reduction_ratio": round(len(reduced) / len(cand_rows), 5),
            "retained": retained,
            "bd_recall": _bd_recall(retained),
            "metrics": _metrics(reduced, hidden, success, a_centroids),
        })

    recommended = _recommend(sweep)
    report = {
        "pos_offset": offset,
        "n_full": len(cand_rows),
        "note": "candidate population excludes reference A; raw (un-balanced) token set.",
        "baseline_no_gate": baseline,
        "sweep": sweep,
        "recommended": recommended,
    }
    pos_dir = cfg.pos_dir(offset)
    io.write_json(pos_dir / "sink_filter_report.json", report)
    io.write_text(pos_dir / "sink_filter_report.md", _md(report))
    rec = recommended.get("keep_pct") if recommended else None
    print(f"[06] pos{offset}: swept {len(sweep)} gates; recommended keep_pct={rec}")
    return report


def _recommend(sweep: list[dict]) -> dict:
    """Smallest keep-% (most reduction) that still preserves B and D recall."""
    safe = [s for s in sweep
            if all((s["bd_recall"].get(L) or 0.0) >= _MIN_BD_RECALL for L in ("B", "D"))]
    if not safe:
        return {}
    best = min(safe, key=lambda s: s["keep_pct"])
    return {
        "keep_pct": best["keep_pct"],
        "reduction_ratio": best["reduction_ratio"],
        "bd_recall": best["bd_recall"],
        "reason": f"smallest keep_pct with B&D recall >= {_MIN_BD_RECALL}",
    }


def _best_auc(metrics_block: dict, signal: str):
    for d in metrics_block["best_per_signal"]:
        if d["signal"] == signal:
            return d["best_auc"]
    return None


def _md(r: dict) -> str:
    signals = [d["signal"] for d in r["baseline_no_gate"]["best_per_signal"]]
    L = [f"# Stage 06 — sink-filter sweep (pos_offset={r['pos_offset']})", "",
         f"Full token set: {r['n_full']}. Each row is a gate keep-% (1st-stage "
         "sink filter); the signal columns are per-signal best-layer AUROC on the "
         "survivors.", ""]
    L.append("| keep_% | n_kept | ratio | B rec | D rec | " + " | ".join(signals) + " |")
    L.append("|" + "---|" * (5 + len(signals)))
    base = r["baseline_no_gate"]
    L.append(f"| 100* | {r['n_full']} | 1.0 | - | - | "
             + " | ".join(str(_best_auc(base, s)) for s in signals) + " |")
    for s in r["sweep"]:
        bd = s["bd_recall"]
        L.append(f"| {s['keep_pct']} | {s['n_reduced']} | {s['reduction_ratio']} | "
                 f"{bd.get('B')} | {bd.get('D')} | "
                 + " | ".join(str(_best_auc(s["metrics"], sig)) for sig in signals) + " |")
    L.append("")
    rec = r.get("recommended") or {}
    L.append(f"**Recommended gate:** keep_pct={rec.get('keep_pct')} "
             f"({rec.get('reason', 'n/a')}).")
    return "\n".join(L) + "\n"


def run(cfg: ExpConfig, lm=None) -> dict:  # lm unused (model-free stage)
    rows_all, hidden, success = ac.load_artifacts(cfg.out_dir, cfg.asr_judge)
    out = {}
    for off in cfg.pos_offsets:
        out[f"pos{off}"] = _analyze_offset(cfg, rows_all, hidden, success, off)
    return out


def main() -> None:
    args = make_stage_parser(__doc__).parse_args()
    run(config_from_args(args))


if __name__ == "__main__":
    main()
