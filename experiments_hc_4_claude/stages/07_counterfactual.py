"""Stage 07 — counterfactual paired-control validation.

Scores every token with the selected clean operating-point scalarizer (oriented so
higher = more attack-like) and compares matched pairs sharing the same source
prompt (``prompt_idx``):

    B - C  (attack mimicry vs the SAME token in a benign context — identity control)
    B - F  (attack mimicry vs a benign word in the attack slot — position control)
    D - E  (attack special vs a benign special token)
    D - F  (attack special vs a positioned benign word)
    F - G  (positioned benign word vs an ordinary body token — pure position/baseline)

A real signal should score the attack token strictly above its control. All five
pairs are always reported (paired_auc = None when no matches exist).

Outputs (per ``pos{off}/``):
    counterfactual_paired_deltas.csv
    counterfactual_validation_report.json
    counterfactual_manifest.jsonl  (for a later true forward-pass rerun)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_4_claude.config import ExpConfig, config_from_args, make_stage_parser  # noqa: E402
from experiments_hc_4_claude.core import io, metrics                  # noqa: E402
from experiments_hc_4_claude.stages import scalar_common as sc        # noqa: E402

PAIR_SPECS = [("B_minus_C", "B", "C"), ("B_minus_F", "B", "F"),
              ("D_minus_E", "D", "E"), ("D_minus_F", "D", "F"),
              ("F_minus_G", "F", "G")]
MANIFEST_PAIRS = [("malicious_mimicry", "benign_mimicry", "B_to_C"),
                  ("malicious_mimicry", "positioned_regular", "B_to_F"),
                  ("malicious_special", "benign_special", "D_to_E"),
                  ("malicious_special", "positioned_regular", "D_to_F")]


def _oriented_scores(col, direction):
    return col if direction != "lower_is_attack" else -col


def _best_by_prompt_letter(rows, scores):
    out: dict[tuple, float] = {}
    for r, s in zip(rows, scores):
        if s != s:        # NaN
            continue
        key = (str(r["prompt_idx"]), r["letter"])
        if key not in out or s > out[key]:
            out[key] = float(s)
    return out


def _run_offset(cfg: ExpConfig, off: int) -> dict:
    rows, mats, meta, arr = sc.load_scalar_scores(cfg, off)
    op = io.read_json(cfg.pos_dir(off) / "operating_points.json")
    sel = op.get("clean")
    report = {"pos_offset": off, "operating_point": sel, "summary": [], "n_delta_rows": 0}

    deltas = []
    if sel and sel["scalarizer"] in mats:
        col = mats[sel["scalarizer"]][:, sel["layer"]]
        scores = _oriented_scores(col, sel["direction"])
        best = _best_by_prompt_letter(rows, scores)
        idxs = sorted({k[0] for k in best})
        for name, a_letter, c_letter in PAIR_SPECS:
            for idx in idxs:
                a = best.get((idx, a_letter)); c = best.get((idx, c_letter))
                if a is None or c is None:
                    continue
                deltas.append({"pair": name, "prompt_idx": idx,
                               "attack_letter": a_letter, "control_letter": c_letter,
                               "attack_score": round(a, 6), "control_score": round(c, 6),
                               "delta": round(a - c, 6)})
    io.write_csv(cfg.pos_dir(off) / "counterfactual_paired_deltas.csv", deltas)
    report["n_delta_rows"] = len(deltas)

    for name, _a, _c in PAIR_SPECS:
        vals = np.array([d["delta"] for d in deltas if d["pair"] == name], dtype=np.float64)
        pair_rows = [d for d in deltas if d["pair"] == name]
        if len(pair_rows) == 0:
            report["summary"].append({"pair": name, "n": 0, "mean_delta": None,
                                      "frac_delta_positive": None, "paired_auc": None})
            continue
        s, lab = [], []
        for d in pair_rows:
            s += [d["attack_score"], d["control_score"]]; lab += [1, 0]
        report["summary"].append({
            "pair": name, "n": int(len(vals)),
            "mean_delta": round(float(vals.mean()), 5),
            "median_delta": round(float(np.median(vals)), 5),
            "frac_delta_positive": round(float((vals > 0).mean()), 5),
            "paired_auc": round(float(metrics.roc_auc(np.array(s), np.array(lab))), 5)})

    io.write_jsonl(cfg.pos_dir(off) / "counterfactual_manifest.jsonl", _manifest(cfg))
    io.write_json(cfg.pos_dir(off) / "counterfactual_validation_report.json", report)
    print(f"[07] pos{off}: {len(deltas)} paired deltas; "
          f"{[(s['pair'], s['paired_auc']) for s in report['summary']]}")
    return report


def _manifest(cfg) -> list[dict]:
    p = cfg.out_dir / "prompts.jsonl"
    if not p.exists():
        return []
    prompts = io.read_jsonl(p)
    by = {(str(r.get("idx", "")), r.get("variant")): r for r in prompts}
    idxs = sorted({str(r.get("idx", "")) for r in prompts})
    rows = []
    for idx in idxs:
        for av, cv, name in MANIFEST_PAIRS:
            a = by.get((idx, av)); c = by.get((idx, cv))
            if not a or not c:
                continue
            rows.append({"counterfactual_pair": name, "prompt_idx": idx,
                         "attack_sample_index": int(a["sample_index"]),
                         "control_sample_index": int(c["sample_index"]),
                         "attack_text": a.get("text"), "counterfactual_text": c.get("text")})
    return rows


def run(cfg: ExpConfig, lm=None) -> dict:
    out = {}
    for off in cfg.pos_offsets:
        out[f"pos{off}"] = _run_offset(cfg, off)
    return out


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
