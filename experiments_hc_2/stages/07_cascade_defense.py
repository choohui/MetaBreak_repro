"""Stage 07 (Main.md §4) — the 2-stage cascade defense, evaluated honestly.

Builds the detector from the §2/§3 findings and measures how well it blocks real
attacks **on a held-out split** so the numbers are not in-sample optimistic:

  Stage 1 (sink gate)   : keep only the top ``--cascade_keep_pct``% of each
                          prompt's tokens by sink (drops the body cheaply).
  Stage 2 (threshold)   : on the survivors, flag a token when the chosen signal
                          crosses a threshold fixed at a target benign FPR.

Validity safeguards (vs. a naive in-sample evaluation):
  * **prompt-level holdout**: prompts are split train/test by ``sample_index``;
    the 2nd-stage threshold AND the (signal, layer) selection are fit on TRAIN,
    every reported rate is measured on TEST. Falls back to in-sample only when
    there are too few prompts (flagged as ``eval_mode``).
  * **reference A excluded** from the candidate/gate population (template specials
    are system-placed and would otherwise consume the gate budget).
  * **raw (un-balanced) token set** so the per-prompt distribution the gate sees
    is realistic (the balancing cap is a §2-only view).

Reported per type and vs two baselines (threshold-only, gate-only):
  attack block-rate (B,D), benign FPR per type (esp. C, E), ASR before/after, and
  block-rate-among-successful (the honest efficacy number). Model-free.
Outputs under ``out_dir/pos{offset}/``: cascade_report.json / .md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # repro_mb (makes experiments_hc_2 importable)
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_2.config import ExpConfig, config_from_args, make_stage_parser  # noqa: E402
from experiments_hc_2.core import io  # noqa: E402
from experiments_hc_2.core import cascade  # noqa: E402
from experiments_hc_2.core.labels import CAT_A  # noqa: E402
import experiments_hc_2.stages.analysis_common as ac  # noqa: E402

# Need at least this many candidate prompts to bother with a held-out split.
_MIN_PROMPTS_FOR_HOLDOUT = 9


def _holdout_test_samples(rows: list[dict]) -> set[int] | None:
    """Deterministic prompt-level test split (every 3rd sample_index). Returns
    None when a split would leave a fold without both attack and benign prompts
    (then the caller falls back to in-sample, flagged)."""
    sidx = sorted({int(r["sample_index"]) for r in rows})
    if len(sidx) < _MIN_PROMPTS_FOR_HOLDOUT:
        return None
    test = {s for i, s in enumerate(sidx) if i % 3 == 0}
    train = set(sidx) - test

    def has_both(sub: set[int]) -> bool:
        labs = {(r["letter"] in ("B", "D")) for r in rows if int(r["sample_index"]) in sub}
        return {True, False} <= labs

    if not (has_both(test) and has_both(train)):
        return None
    return test


def _select_signal(cfg: ExpConfig, signals: dict, y: np.ndarray,
                   is_train: np.ndarray) -> tuple[str, int]:
    """Pick (signal, layer) by AUC on the TRAIN rows (or honor the CLI override)."""
    if cfg.cascade_signal and cfg.cascade_layer is not None:
        return cfg.cascade_signal, int(cfg.cascade_layer)
    train_sig = {nm: mat[is_train] for nm, mat in signals.items()}
    evals = ac.evaluate_signals(train_sig, y[is_train])
    names = [cfg.cascade_signal] if cfg.cascade_signal else list(evals.keys())
    best = None  # (name, layer, auc)
    for nm in names:
        info = evals.get(nm)
        if not info or info["best_layer"] is None:
            continue
        auc = info["best_auc"]
        if best is None or (auc is not None and auc == auc and auc > best[2]):
            best = (nm, info["best_layer"], auc)
    if best is None:
        raise SystemExit("[07] no usable signal on the train split (need both classes).")
    return best[0], int(best[1])


def _strategy(rows, col, y, cand_mask, is_train, is_test, success, fpr, with_threshold):
    """Fit on TRAIN candidates, evaluate on TEST. ``cand_mask`` = stage-1 survivors.
    ``with_threshold`` applies stage 2; otherwise candidates are flagged outright."""
    if with_threshold:
        train_cand = cand_mask & is_train
        thr = cascade.threshold_at_fpr(col[train_cand], y[train_cand], fpr)
        pred = cascade.predict(col, thr["threshold"], thr["direction"]) & cand_mask
    else:
        thr = None
        pred = cand_mask.copy()
    test_rows = [r for r, t in zip(rows, is_test) if t]
    pred_test = pred[is_test]
    return {
        "threshold": thr,
        "with_threshold": with_threshold,
        "n_eval": int(is_test.sum()),
        "gate_pass_ratio": round(float((cand_mask & is_test).sum() / max(1, int(is_test.sum()))), 5),
        "per_type": cascade.per_type_rates(test_rows, pred_test),
        "prompt": cascade.prompt_block_and_asr(test_rows, pred_test, success),
    }


def _analyze_offset(cfg: ExpConfig, rows_all, hidden, success, offset: int) -> dict:
    rows_off = [r for r in rows_all if r["pos_offset"] == offset]
    if not rows_off:
        return {}
    # H1: exclude reference A (template specials) from the candidate population;
    # keep them only for the cos_to_ref centroid.
    cand_rows = [r for r in rows_off if r["category"] != CAT_A]
    if not cand_rows:
        return {}

    a_centroids = ac.reference_centroids(rows_off, hidden)
    signals = ac.build_signals(cand_rows, hidden, a_centroids=a_centroids)
    y = cascade.binary_labels(cand_rows)

    test_samples = _holdout_test_samples(cand_rows)
    eval_mode = "holdout" if test_samples is not None else "in_sample"
    if test_samples is None:
        is_test = np.ones(len(cand_rows), dtype=bool)
        is_train = np.ones(len(cand_rows), dtype=bool)
    else:
        is_test = np.array([int(r["sample_index"]) in test_samples for r in cand_rows])
        is_train = ~is_test

    name, layer = _select_signal(cfg, signals, y, is_train)
    if name not in signals:
        raise SystemExit(f"[07] signal {name!r} not available in artifacts.")
    col = signals[name][:, layer]

    gated = cascade.sink_gate(cand_rows, cfg.cascade_keep_pct)
    gated_ids = {r["row_id"] for r in gated}
    gate_mask = np.array([r["row_id"] in gated_ids for r in cand_rows])
    full_mask = np.ones(len(cand_rows), dtype=bool)

    strategies = {
        "one_stage_threshold": _strategy(
            cand_rows, col, y, full_mask, is_train, is_test, success, cfg.cascade_fpr, True),
        "gate_only": _strategy(
            cand_rows, col, y, gate_mask, is_train, is_test, success, cfg.cascade_fpr, False),
        "cascade": _strategy(
            cand_rows, col, y, gate_mask, is_train, is_test, success, cfg.cascade_fpr, True),
    }

    report = {
        "pos_offset": offset,
        "asr_judge": cfg.asr_judge,
        "eval_mode": eval_mode,
        "n_candidates": len(cand_rows),
        "n_train": int(is_train.sum()) if eval_mode == "holdout" else len(cand_rows),
        "n_test": int(is_test.sum()),
        "signal": name,
        "layer": layer,
        "signal_selected_on": "train split" if not cfg.cascade_signal else "cli override",
        "keep_pct": cfg.cascade_keep_pct,
        "operating_fpr": cfg.cascade_fpr,
        "gate_reduction_ratio": round(float(gate_mask.mean()), 5),
        "asr_before": strategies["cascade"]["prompt"]["asr_before"],
        "notes": {
            "gate_only": "gate_only flags ALL gate survivors (no threshold) — its "
                         "per-type 'FPR' is gate survival, a recall/coverage diagnostic, "
                         "NOT a detector FPR.",
            "eval": "rates are on the held-out TEST split; thresholds + (signal,layer) "
                    "fit on TRAIN. eval_mode=in_sample means too few prompts to split.",
        },
        "strategies": strategies,
    }
    pos_dir = cfg.pos_dir(offset)
    io.write_json(pos_dir / "cascade_report.json", report)
    io.write_text(pos_dir / "cascade_report.md", _md(report))
    c = strategies["cascade"]
    print(f"[07] pos{offset}: cascade {name}@L{layer} keep={cfg.cascade_keep_pct}% "
          f"[{eval_mode}] -> B block={_rate(c,'B')} D block={_rate(c,'D')} "
          f"E FPR={_rate(c,'E')} C FPR={_rate(c,'C')} "
          f"ASR {report['asr_before']}->{c['prompt']['asr_after']}")
    return report


def _rate(strategy: dict, letter: str):
    return strategy["per_type"].get(letter, {}).get("rate")


def _md(r: dict) -> str:
    L = [f"# Stage 07 — cascade defense (pos_offset={r['pos_offset']})", "",
         f"- 2nd-stage signal: **{r['signal']}** @ layer {r['layer']} "
         f"(selected on {r['signal_selected_on']})",
         f"- 1st-stage sink gate keep_pct: **{r['keep_pct']}%** "
         f"(candidate tokens reaching stage 2: {r['gate_reduction_ratio']})",
         f"- operating point: threshold at benign FPR = {r['operating_fpr']}",
         f"- evaluation: **{r['eval_mode']}** (n_train={r['n_train']}, n_test={r['n_test']}), "
         f"ASR judge={r['asr_judge']}", "",
         "## Block-rate / FPR / ASR by strategy (held-out test)", "",
         "| strategy | B block | D block | C FPR | E FPR | F FPR | G FPR | "
         "ASR before | ASR after | blk@succ | gate-pass |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in ("one_stage_threshold", "gate_only", "cascade"):
        s = r["strategies"][key]
        p = s["prompt"]
        L.append(
            f"| {key} | {_rate(s,'B')} | {_rate(s,'D')} | {_rate(s,'C')} | "
            f"{_rate(s,'E')} | {_rate(s,'F')} | {_rate(s,'G')} | "
            f"{p['asr_before']} | {p['asr_after']} | "
            f"{p.get('block_rate_among_successful')} | {s['gate_pass_ratio']} |")
    L.append("")
    L.append("> `gate_only` flags every gate survivor (no threshold); its C/E/F/G "
             "column is gate-survival (coverage), not a detector FPR.")
    L.append("")
    L.append("Goal: the **cascade** row keeps B/D block-rate high while pushing C/E "
             "FPR below the threshold-only baseline (the §5 special-detector trap), "
             "with ASR-after < ASR-before on held-out prompts.")
    return "\n".join(L) + "\n"


def run(cfg: ExpConfig, lm=None) -> dict:  # lm unused (model-free stage)
    # §4 uses the RAW (un-balanced) token set so the gate's per-prompt view is real.
    rows_all, hidden, success = ac.load_artifacts(cfg.out_dir, cfg.asr_judge, balanced=False)
    offs = [cfg.cascade_pos_offset] if cfg.cascade_pos_offset in cfg.pos_offsets \
        else cfg.pos_offsets
    out = {}
    for off in offs:
        out[f"pos{off}"] = _analyze_offset(cfg, rows_all, hidden, success, off)
    return out


def main() -> None:
    args = make_stage_parser(__doc__).parse_args()
    run(config_from_args(args))


if __name__ == "__main__":
    main()
