"""Stage 11 - Counterfactual validation.

Default mode is artifact-only: compare Active SinkProbe scores against existing
paired controls by prompt_idx:
  B - C, B - F, D - E, D - F

It also writes a counterfactual manifest that can be used for a future true
forward-pass rerun where attack prompts are replaced with safe paired prompts.
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

from experiments_hc_3.config import ExpConfig, config_from_args, make_parser  # noqa: E402
from experiments_hc_3.core import artifacts, io, metrics  # noqa: E402

PAIR_SPECS = [
    ("B_minus_C", "B", "C"),
    ("B_minus_F", "B", "F"),
    ("D_minus_E", "D", "E"),
    ("D_minus_F", "D", "F"),
]


def _load_scores(cfg: ExpConfig, offset: int) -> list[dict]:
    path = cfg.pos_dir(offset) / "active_sinkprobe_scores.jsonl"
    if not path.exists():
        raise SystemExit(f"[11] missing {path}; run stage 08 first.")
    return io.read_jsonl(path)


def _best_by_prompt_letter(records: list[dict]) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for r in records:
        if r.get("cv_score") is None:
            continue
        key = (str(r.get("prompt_idx", "")), str(r.get("letter")))
        old = out.get(key)
        if old is None or float(r["cv_score"]) > float(old["cv_score"]):
            out[key] = r
    return out


def _delta_table(records: list[dict]) -> list[dict]:
    best = _best_by_prompt_letter(records)
    prompt_idxs = sorted({k[0] for k in best})
    rows = []
    for name, attack_letter, control_letter in PAIR_SPECS:
        for idx in prompt_idxs:
            a = best.get((idx, attack_letter))
            c = best.get((idx, control_letter))
            if not a or not c:
                continue
            rows.append({
                "pair": name,
                "prompt_idx": idx,
                "attack_letter": attack_letter,
                "control_letter": control_letter,
                "attack_sample_index": int(a["sample_index"]),
                "control_sample_index": int(c["sample_index"]),
                "attack_score": float(a["cv_score"]),
                "control_score": float(c["cv_score"]),
                "delta": float(a["cv_score"]) - float(c["cv_score"]),
                "attack_decoded": a.get("decoded"),
                "control_decoded": c.get("decoded"),
            })
    return rows


def _summarize_deltas(rows: list[dict]) -> list[dict]:
    out = []
    for pair in [p[0] for p in PAIR_SPECS]:
        vals = np.array([r["delta"] for r in rows if r["pair"] == pair], dtype=np.float64)
        if len(vals) == 0:
            continue
        # AUC view: attack/control score pair as a binary mini-dataset.
        pair_rows = [r for r in rows if r["pair"] == pair]
        scores = []
        labels = []
        for r in pair_rows:
            scores.extend([r["attack_score"], r["control_score"]])
            labels.extend([1, 0])
        out.append({
            "pair": pair,
            "n": int(len(vals)),
            "mean_delta": round(float(vals.mean()), 5),
            "median_delta": round(float(np.median(vals)), 5),
            "p10_delta": round(float(np.quantile(vals, 0.1)), 5),
            "p90_delta": round(float(np.quantile(vals, 0.9)), 5),
            "frac_delta_positive": round(float((vals > 0).mean()), 5),
            "paired_auc": round(float(metrics.roc_auc(np.array(scores), np.array(labels))), 5),
        })
    return out


def _manifest(prompts: list[dict]) -> list[dict]:
    by_idx_variant: dict[tuple[str, str], dict] = {}
    for p in prompts:
        by_idx_variant[(str(p.get("idx", "")), str(p.get("variant")))] = p
    rows = []
    pairs = [
        ("malicious_mimicry", "positioned_regular", "B_to_F"),
        ("malicious_mimicry", "benign_mimicry", "B_to_C"),
        ("malicious_special", "positioned_regular", "D_to_F"),
        ("malicious_special", "benign_special", "D_to_E"),
    ]
    idxs = sorted({str(p.get("idx", "")) for p in prompts})
    for idx in idxs:
        for attack_variant, control_variant, name in pairs:
            a = by_idx_variant.get((idx, attack_variant))
            c = by_idx_variant.get((idx, control_variant))
            if not a or not c:
                continue
            rows.append({
                "counterfactual_pair": name,
                "prompt_idx": idx,
                "attack_sample_index": int(a["sample_index"]),
                "control_sample_index": int(c["sample_index"]),
                "attack_variant": attack_variant,
                "control_variant": control_variant,
                "attack_text": a.get("text"),
                "counterfactual_text": c.get("text"),
            })
    return rows


def _run_offset(cfg: ExpConfig, offset: int) -> dict:
    records = _load_scores(cfg, offset)
    deltas = _delta_table(records)
    summary = _summarize_deltas(deltas)
    prompts = artifacts.load_prompts(cfg.out_dir)
    manifest = _manifest(prompts)

    pos_dir = cfg.pos_dir(offset)
    io.write_csv(pos_dir / "counterfactual_paired_deltas.csv", deltas)
    io.write_jsonl(pos_dir / "counterfactual_manifest.jsonl", manifest)
    report = {
        "stage": "11_counterfactual_validation",
        "pos_offset": offset,
        "mode": "artifact_paired_proxy",
        "n_delta_rows": len(deltas),
        "n_manifest_rows": len(manifest),
        "summary": summary,
        "notes": {
            "proxy": "Deltas compare existing paired controls by prompt_idx; they do not replace tokens in-place.",
            "true_counterfactual": "Use counterfactual_manifest.jsonl to rerun forward extraction on attack_text vs counterfactual_text.",
        },
    }
    io.write_json(pos_dir / "counterfactual_validation_report.json", report)
    io.write_text(pos_dir / "counterfactual_validation_report.md", _md(report))
    print(f"[11] pos{offset}: delta_rows={len(deltas)} manifest_rows={len(manifest)}")
    return report


def _md(r: dict) -> str:
    lines = [
        f"# Stage 11 - Counterfactual Validation (pos_offset={r['pos_offset']})",
        "",
        f"- mode: {r['mode']}",
        f"- paired delta rows: {r['n_delta_rows']}",
        f"- manifest rows: {r['n_manifest_rows']}",
        "",
        "| pair | n | mean delta | median | frac positive | paired AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in r["summary"]:
        lines.append(f"| {row['pair']} | {row['n']} | {row['mean_delta']} | "
                     f"{row['median_delta']} | {row['frac_delta_positive']} | {row['paired_auc']} |")
    lines.extend([
        "",
        "The manifest is for a later true counterfactual rerun. The current report is a paired-control proxy.",
    ])
    return "\n".join(lines) + "\n"


def run(cfg: ExpConfig) -> dict:
    out = {}
    for off in cfg.pos_offsets:
        out[f"pos{off}"] = _run_offset(cfg, off)
    return out


def main() -> None:
    p = make_parser(__doc__)
    run(config_from_args(p.parse_args()))


if __name__ == "__main__":
    main()
