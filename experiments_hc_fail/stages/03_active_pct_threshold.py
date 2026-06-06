"""Stage 03 - active_value top-percent gate plus threshold evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG.parent
for p in (str(REPO_ROOT), str(PKG.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments_hc_fail.config import ExpConfig, config_from_args, make_parser  # noqa: E402
from experiments_hc_fail.core import io  # noqa: E402
from experiments_hc_fail.core.metrics import run_sweep  # noqa: E402


def _success_ids(cfg: ExpConfig) -> set[int] | None:
    if not cfg.responses.exists():
        return None
    rows = io.read_jsonl(cfg.responses)
    return {int(r["sample_index"]) for r in rows if r.get("success_refusal_keyword")}


def _summary_rows(report: dict) -> list[dict]:
    out = []
    for row in report["sweep"]:
        ev = row["evaluation"]
        token = ev["token"]
        per = token["per_letter"]
        prompt = ev["prompt"]
        asr = ev.get("asr") or {}
        thr = row["threshold"]
        out.append({
            "keep_pct": row["keep_pct"],
            "n_full": row["n_full"],
            "n_kept": row["n_kept"],
            "reduction_ratio": row["reduction_ratio"],
            "threshold": thr.get("threshold"),
            "train_tpr": thr.get("train_tpr"),
            "train_fpr": thr.get("train_fpr"),
            "train_auc": thr.get("auc"),
            "B_recall": per.get("B", {}).get("rate"),
            "D_recall": per.get("D", {}).get("rate"),
            "BD_recall": token.get("recall"),
            "C_fpr": per.get("C", {}).get("rate"),
            "E_fpr": per.get("E", {}).get("rate"),
            "F_fpr": per.get("F", {}).get("rate"),
            "G_fpr": per.get("G", {}).get("rate"),
            "precision": token.get("precision"),
            "f1": token.get("f1"),
            "prompt_block_rate": prompt.get("block_rate"),
            "prompt_fpr": prompt.get("prompt_fpr"),
            "asr_before": asr.get("asr_before"),
            "asr_after": asr.get("asr_after"),
            "block_rate_among_successful": asr.get("block_rate_among_successful"),
        })
    return out


def run(cfg: ExpConfig) -> dict:
    rows = io.read_jsonl(cfg.tokens)
    report = run_sweep(rows, cfg.keep_pcts, cfg.fpr, cfg.seed, _success_ids(cfg))
    report.update({
        "stage": "03_active_pct_threshold",
        "score": "active_value = sink * value_norm; row score = max_over_layers(active_value)",
        "selection": "per (sample_index, pos_offset), keep top keep_pct rows by active_value_max",
        "n_rows": len(rows),
        "keep_pcts": cfg.keep_pcts,
    })
    io.write_json(cfg.report_json, report)
    io.write_csv(cfg.sweep_csv, _summary_rows(report))
    print(f"[03] wrote report -> {cfg.report_json}")
    print(f"[03] wrote summary -> {cfg.sweep_csv}")
    return report


def main() -> None:
    run(config_from_args(make_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()

