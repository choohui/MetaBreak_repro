from __future__ import annotations

import numpy as np

from experiments_hc_4.config import ExpConfig
from experiments_hc_4.core import io
from experiments_hc_4.core.thresholds import orient_scores


def _prompt_scores(rows: list[dict], x: np.ndarray, names: list[str], rule: dict) -> dict[int, float]:
    idx = {n: i for i, n in enumerate(names)}
    scores = []
    for term in rule["terms"]:
        s = orient_scores(x[:, idx[term["feature"]]], term["direction"]) - float(term["threshold"])
        scores.append(s)
    combo = np.max(np.vstack(scores), axis=0) if scores else np.zeros(len(rows))
    out: dict[int, float] = {}
    for r, s in zip(rows, combo):
        sid = int(r["sample_index"])
        out[sid] = max(out.get(sid, -1e9), float(s))
    return out


def run(cfg: ExpConfig, lm=None) -> dict:
    rows = io.read_jsonl(cfg.out_dir / "balanced_tokens.jsonl")
    inputs = io.read_jsonl(cfg.inputs_path)
    data = np.load(cfg.out_dir / "scalar_values.npz", allow_pickle=True)
    x = data["x"].astype(float)
    names = [str(x) for x in data["feature_names"].tolist()]
    rules = io.read_json(cfg.out_dir / "threshold_rules.json")
    rule = rules["selected"].get("0.01") or next(v for v in rules["selected"].values() if v)
    scores = _prompt_scores(rows, x, names, rule)
    by_variant: dict[str, list[dict]] = {}
    for inp in inputs:
        by_variant.setdefault(inp["variant"], []).append(inp)
    for vs in by_variant.values():
        vs.sort(key=lambda r: str(r.get("idx")))
    pairs = [
        ("malicious_mimicry", "positioned_regular", "B_to_F"),
        ("malicious_special", "positioned_regular", "D_to_F"),
        ("malicious_mimicry", "benign_mimicry", "B_to_C"),
        ("malicious_special", "benign_special", "D_to_E"),
    ]
    report_rows = []
    for a, b, name in pairs:
        left = by_variant.get(a, [])
        right = by_variant.get(b, [])
        n = min(len(left), len(right))
        deltas = []
        for i in range(n):
            ds = scores.get(int(left[i]["sample_index"]))
            cs = scores.get(int(right[i]["sample_index"]))
            if ds is not None and cs is not None:
                deltas.append(ds - cs)
        arr = np.asarray(deltas, dtype=float)
        report_rows.append({
            "pair": name,
            "n": int(len(arr)),
            "mean_delta": round(float(arr.mean()), 5) if len(arr) else None,
            "median_delta": round(float(np.median(arr)), 5) if len(arr) else None,
            "frac_positive": round(float((arr > 0).mean()), 5) if len(arr) else None,
            "note": "both sides were separately forward-captured in this run",
        })
    report = {"stage": "07_counterfactual", "pairs": report_rows}
    io.write_json(cfg.out_dir / "counterfactual_report.json", report)
    io.write_csv(cfg.out_dir / "counterfactual_paired_deltas.csv", report_rows)
    print("[07] wrote counterfactual report")
    return report

