"""Stage 08 — token-exclusion defense + ASR effect.

Applies the selected clean operating point to flag attack-slot tokens and turns
that into a prompt-level defense. The HEADLINE metric is the model-free block-rate
proxy (a prompt is defended if any of its attack-slot B/D tokens is flagged),
evaluated on the held-out TEST split — the same proxy hc_2/hc_3 used. With
``--real_intervention`` (and a real ``--model``) it additionally DROPS the flagged
tokens and RE-GENERATES to measure ASR for real; that path is never run in smoke.

Outputs (per ``pos{off}/``):
    defense_report.json   - proxy ASR before/after + block-rate-among-successful, per type
    real_asr.json         - only when --real_intervention runs
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

from experiments_hc_4_claude.config import ExpConfig, config_from_args, get_model, make_stage_parser  # noqa: E402
from experiments_hc_4_claude.core import io                                  # noqa: E402
from experiments_hc_4_claude.core import thresholds as TH                    # noqa: E402
from experiments_hc_4_claude.core.cascade import prompt_block_and_asr, per_type_rates  # noqa: E402
from experiments_hc_4_claude.stages import scalar_common as sc              # noqa: E402
from experiments_hc_4_claude.stages.analysis_common import success_set       # noqa: E402


def _proxy(rows, pred, mask, success):
    idx = np.where(mask)[0]
    sub_rows = [rows[i] for i in idx]
    sub_pred = pred[mask]
    res = prompt_block_and_asr(sub_rows, sub_pred, success)
    res["per_type"] = per_type_rates(sub_rows, sub_pred)
    return res


def _run_offset(cfg: ExpConfig, off: int, lm=None) -> dict:
    rows, mats, meta, arr = sc.load_scalar_scores(cfg, off)
    is_test = arr["is_test"]
    success = success_set(cfg.out_dir, cfg.asr_judge)
    op = io.read_json(cfg.pos_dir(off) / "operating_points.json")
    sel = op.get("clean")
    report = {"pos_offset": off, "eval_mode": meta["eval_mode"],
              "operating_point": sel, "asr_judge": cfg.asr_judge}

    if not sel or sel["scalarizer"] not in mats:
        report.update({"asr_before": None, "asr_after": None,
                       "block_rate_among_successful": None, "note": "no operating point"})
        io.write_json(cfg.pos_dir(off) / "defense_report.json", report)
        return report

    col = mats[sel["scalarizer"]][:, sel["layer"]]
    pred = TH.predict(col, sel["threshold"], sel["direction"])
    proxy = {"test": _proxy(rows, pred, is_test, success),
             "full": _proxy(rows, pred, np.ones(len(rows), bool), success)}
    report["proxy"] = proxy
    # surface the held-out headline numbers at the top level
    t = proxy["test"]
    report["asr_before"] = t["asr_before"]
    report["asr_after"] = t["asr_after"]
    report["block_rate_among_successful"] = t["block_rate_among_successful"]

    if cfg.real_intervention:
        if lm is None or getattr(lm, "is_mock", False):
            report["real_note"] = "real_intervention requested but no real model; proxy only."
        else:
            report["real"] = _real(cfg, off, rows, pred, is_test, success, lm)
            io.write_json(cfg.pos_dir(off) / "real_asr.json", report["real"])

    io.write_json(cfg.pos_dir(off) / "defense_report.json", report)
    print(f"[08] pos{off}: proxy held-out ASR {t['asr_before']} -> {t['asr_after']} "
          f"(block_among_succ={t['block_rate_among_successful']})")
    return report


def _real(cfg, off, rows, pred, is_test, success, lm) -> dict:
    """Re-generate held-out successful attacks with their flagged tokens removed."""
    from experiments_hc_4_claude.core.intervene import real_intervention_asr
    # need token positions: reload tokens.jsonl (balanced, this pos), align by row_id
    toks = [r for r in io.read_jsonl(cfg.out_dir / "tokens.jsonl")
            if int(r["pos_offset"]) == off]
    summary = io.read_json(cfg.out_dir / "extract_summary.json")
    keep = set(summary.get("balanced_row_ids", []))
    toks = [r for r in toks if r["row_id"] in keep]
    by_id = {r["row_id"]: r for r in toks}
    prompts = {int(r["sample_index"]): r for r in io.read_jsonl(cfg.out_dir / "prompts.jsonl")}
    drop_by_sample: dict[int, set] = {}
    prompt_by_sample: dict[int, dict] = {}
    for i, r in enumerate(rows):
        if not (is_test[i] and pred[i]):
            continue
        if r["letter"] not in ("B", "D"):
            continue
        tok = by_id.get(r["row_id"])
        if tok is None:
            continue
        s = int(r["sample_index"])
        drop_by_sample.setdefault(s, set()).add(int(tok["position"]))
        if s in prompts:
            prompt_by_sample[s] = prompts[s]
    return real_intervention_asr(lm, prompt_by_sample, drop_by_sample, success,
                                 cfg.max_new_tokens, cfg.temperature)


def run(cfg: ExpConfig, lm=None) -> dict:
    if cfg.real_intervention and lm is None:
        lm = get_model(cfg, None)
    out = {}
    for off in cfg.pos_offsets:
        out[f"pos{off}"] = _run_offset(cfg, off, lm)
    return out


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
