"""Stage 08 - Active SinkProbe.

Builds sink-rank/order-statistics features plus computationally active sink
features (`sink * value_norm`, `sink * output_norm`) and trains a sparse
logistic probe with prompt-level CV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO_ROOT = PKG.parent
for _p in (str(REPO_ROOT), str(PKG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_3.config import ExpConfig, config_from_args, make_parser, materialize_artifacts  # noqa: E402
from experiments_hc_3.core import artifacts, io, modeling  # noqa: E402
from experiments_hc_3.core.active_features import build_feature_set  # noqa: E402


def _row_score_records(fs, scores: np.ndarray, score_indices: np.ndarray) -> list[dict]:
    score_by_local = {int(i): float(s) for i, s in zip(score_indices, scores)}
    recs = []
    for local_i, r in enumerate(fs.rows):
        y = int(fs.y[local_i])
        rec = {
            "row_id": int(r["row_id"]),
            "sample_index": int(r["sample_index"]),
            "prompt_idx": str(r.get("prompt_idx", "")),
            "variant": r.get("variant"),
            "letter": r.get("letter"),
            "category": r.get("category"),
            "pos_offset": int(r["pos_offset"]),
            "base_position": int(r.get("base_position", -1)),
            "position": int(r.get("position", -1)),
            "token_id": int(r.get("token_id", -1)),
            "decoded": r.get("decoded"),
            "label": y,
            "cv_score": score_by_local.get(local_i),
        }
        recs.append(rec)
    return recs


def _per_letter_scores(records: list[dict]) -> list[dict]:
    out = []
    for letter in "ABCDEFG":
        vals = [r["cv_score"] for r in records if r["letter"] == letter and r["cv_score"] is not None]
        if not vals:
            continue
        out.append({
            "letter": letter,
            "n": len(vals),
            "mean_score": round(float(np.mean(vals)), 5),
            "median_score": round(float(np.median(vals)), 5),
            "p90_score": round(float(np.quantile(vals, 0.9)), 5),
        })
    return out


def _run_offset(cfg: ExpConfig, rows: list[dict], offset: int) -> dict:
    fs = build_feature_set(rows, offset, cfg.top_ks)
    if len(fs.rows) == 0:
        return {"pos_offset": offset, "n_rows": 0}

    cv = modeling.grouped_cv_scores(fs.x, fs.y, fs.groups, seed=cfg.seed)
    mask = modeling.valid_binary_mask(fs.y)
    model = modeling.fit_linear_model(fs.x[mask], fs.y[mask])
    top_coef = modeling.top_coefficients(model, fs.feature_names, k=40)
    single = modeling.single_feature_auc_table(fs.x, fs.y, fs.feature_names, k=80)

    records = _row_score_records(fs, cv["scores"], cv["indices"])
    pos_dir = cfg.pos_dir(offset)
    pos_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pos_dir / "active_sinkprobe_features.npz",
        x=fs.x,
        y=fs.y,
        groups=fs.groups,
        row_ids=np.array([int(r["row_id"]) for r in fs.rows], dtype=int),
        feature_names=np.array(fs.feature_names, dtype=object),
    )
    io.write_jsonl(pos_dir / "active_sinkprobe_scores.jsonl", records)
    io.write_csv(pos_dir / "active_sinkprobe_single_feature_auc.csv", single)

    report = {
        "stage": "08_active_sinkprobe",
        "pos_offset": offset,
        "source_out_dir": str(cfg.source_out_dir),
        "balanced_input": cfg.balanced,
        "n_rows": len(fs.rows),
        "n_features": int(fs.x.shape[1]),
        "labels": "B,D=attack vs C,E,F,G=benign; A excluded from binary fitting",
        "feature_note": "Ranks/top-k are computed among labeled rows in the same prompt and pos_offset.",
        "cv": {k: v for k, v in cv.items() if k not in ("scores", "indices")},
        "per_letter_scores": _per_letter_scores(records),
        "top_coefficients": top_coef,
        "top_single_features": single[:20],
    }
    io.write_json(pos_dir / "active_sinkprobe_report.json", report)
    io.write_text(pos_dir / "active_sinkprobe_report.md", _md(report))
    print(f"[08] pos{offset}: AUC={report['cv'].get('auc')} n={len(fs.rows)} features={fs.x.shape[1]}")
    return report


def _md(r: dict) -> str:
    lines = [
        f"# Stage 08 - Active SinkProbe (pos_offset={r['pos_offset']})",
        "",
        f"- rows: {r['n_rows']}",
        f"- features: {r['n_features']}",
        f"- CV AUC: {r['cv'].get('auc')} ({r['cv'].get('split')}, folds={r['cv'].get('folds')})",
        f"- balanced accuracy: {r['cv'].get('balanced_acc')}",
        "",
        "## Per-letter CV Score",
        "| letter | n | mean | median | p90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in r["per_letter_scores"]:
        lines.append(f"| {row['letter']} | {row['n']} | {row['mean_score']} | "
                     f"{row['median_score']} | {row['p90_score']} |")
    lines.extend(["", "## Top Sparse Coefficients",
                  "| feature | coef |", "|---|---:|"])
    for row in r["top_coefficients"][:25]:
        lines.append(f"| {row['feature']} | {row['coef']} |")
    lines.extend(["", "## Top Single Features",
                  "| feature | AUC | direction |", "|---|---:|---|"])
    for row in r["top_single_features"][:25]:
        lines.append(f"| {row['feature']} | {row['auc']} | {row['direction']} |")
    return "\n".join(lines) + "\n"


def run(cfg: ExpConfig) -> dict:
    materialize_artifacts(cfg)
    rows = artifacts.load_rows(cfg.out_dir, balanced=cfg.balanced)
    out = {}
    for off in cfg.pos_offsets:
        out[f"pos{off}"] = _run_offset(cfg, rows, off)
    return out


def main() -> None:
    p = make_parser(__doc__)
    run(config_from_args(p.parse_args()))


if __name__ == "__main__":
    main()
