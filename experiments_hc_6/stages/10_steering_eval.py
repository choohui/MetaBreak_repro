from __future__ import annotations

from statistics import median

import numpy as np

from experiments_hc_6.config import ExpConfig
from experiments_hc_6.core import io
from experiments_hc_6.core.defense import flagged_base_positions
from experiments_hc_6.core.interventions import generate_with_steering
from experiments_hc_6.core.model import get_model, refusal_success
from experiments_hc_6.core.thresholds import predict_rule


def _default_rule(rules: dict) -> tuple[str, dict]:
    if "0.01" in rules.get("selected", {}) and rules["selected"]["0.01"]:
        return "0.01", rules["selected"]["0.01"]
    for k, v in rules.get("selected", {}).items():
        if v:
            return k, v
    raise RuntimeError("no selected threshold rule")


def _asr(rows: list[dict]) -> float | None:
    atk = [r for r in rows if r.get("letter") in ("B", "D")]
    if not atk:
        return None
    return round(sum(bool(r.get("success")) for r in atk) / len(atk), 5)


def _semantic_summary(test_inputs: list[dict], baseline: dict[int, dict], current: list[dict]) -> dict:
    benign_ids = {int(r["sample_index"]) for r in test_inputs if r.get("letter") in ("C", "E", "F", "G")}
    cur = {int(r["sample_index"]): r for r in current}
    label_matches = []
    ratios = []
    for sid in benign_ids:
        b = baseline.get(sid)
        c = cur.get(sid)
        if not b or not c:
            continue
        label_matches.append(bool(b.get("success")) == bool(c.get("success")))
        blen = max(1, len(str(b.get("response", ""))))
        ratios.append(len(str(c.get("response", ""))) / blen)
    return {
        "semantic_label_match": round(float(np.mean(label_matches)), 5) if label_matches else None,
        "response_length_ratio_median": round(float(median(ratios)), 5) if ratios else None,
    }


def _run_grid_item(lm, cfg: ExpConfig, test_inputs: list[dict], flags: dict[int, set[int]],
                   vectors: dict[str, np.ndarray], layers: list[int], mode: str, alpha: float) -> list[dict]:
    out = []
    for rec in test_inputs:
        sid = int(rec["sample_index"])
        input_ids = [int(x) for x in rec["input_ids"]]
        text = generate_with_steering(
            lm,
            input_ids,
            flags.get(sid, set()),
            vectors,
            layers,
            mode,
            alpha,
            cfg.max_new_tokens,
        )
        out.append({
            "sample_index": sid,
            "variant": rec.get("variant"),
            "letter": rec.get("letter"),
            "mode": mode,
            "alpha": float(alpha),
            "n_flagged_positions": len(flags.get(sid, set())),
            "success": refusal_success(text),
            "response": text,
        })
    return out


def run(cfg: ExpConfig, lm=None) -> dict:
    lm = get_model(cfg, lm)
    rows = io.read_jsonl(cfg.out_dir / "balanced_tokens.jsonl")
    inputs = io.read_jsonl(cfg.inputs_path)
    data = np.load(cfg.out_dir / "scalar_values.npz", allow_pickle=True)
    x = data["x"].astype(float)
    names = [str(x) for x in data["feature_names"].tolist()]
    rules = io.read_json(cfg.out_dir / "threshold_rules.json")
    _fpr_key, rule = _default_rule(rules)
    split_mask = np.asarray([r["split"] == "test" for r in rows])
    pred_test = predict_rule(x[split_mask], names, rule)
    test_rows = [r for r, m in zip(rows, split_mask) if m]
    flags = flagged_base_positions(test_rows, pred_test)
    split_by_group = {str(r["sample_index"]): r["split"] for r in rows}
    test_inputs = [
        r for r in inputs
        if split_by_group.get(str(r["sample_index"])) == "test"
        and r.get("letter") in ("B", "C", "D", "E", "F", "G")
    ]
    vec_npz = np.load(cfg.out_dir / "steering_vectors.npz", allow_pickle=True)
    vectors = {
        "directions": vec_npz["directions"],
        "benign_mu": vec_npz["benign_mu"],
        "attack_mu": vec_npz["attack_mu"],
        "gaps": vec_npz["gaps"],
    }
    layers = [int(x) for x in vec_npz["selected_layers"].tolist()]
    combos = [("no_op", 0.0)]
    for mode in cfg.steer_modes:
        for alpha in cfg.steer_alphas:
            if float(alpha) == 0.0:
                continue
            combos.append((mode, float(alpha)))

    eval_rows = []
    baseline_by_sample: dict[int, dict] = {}
    artifacts = {}
    if cfg.skip_generation:
        for mode, alpha in combos:
            eval_rows.append({"mode": mode, "alpha": alpha, "generation_skipped": True})
    else:
        for mode, alpha in combos:
            records = _run_grid_item(lm, cfg, test_inputs, flags, vectors, layers, mode, alpha)
            name = f"steered_{mode}_{str(alpha).replace('.', 'p')}.jsonl"
            io.write_jsonl(cfg.out_dir / name, records)
            if mode == "no_op":
                baseline_by_sample = {int(r["sample_index"]): r for r in records}
            summary = {
                "mode": mode,
                "alpha": alpha,
                "layers": layers,
                "n_prompts": len(test_inputs),
                "n_flagged_prompts": int(sum(bool(flags.get(int(r["sample_index"]))) for r in test_inputs)),
                "asr_before": _asr(list(baseline_by_sample.values())) if baseline_by_sample else _asr(records),
                "asr_after": _asr(records),
            }
            summary.update(_semantic_summary(test_inputs, baseline_by_sample, records))
            eval_rows.append(summary)
            artifacts[f"{mode}_{alpha}"] = name

    best = None
    for rec in eval_rows:
        if rec.get("mode") == "no_op" or not isinstance(rec.get("asr_after"), (int, float)):
            continue
        if best is None or rec["asr_after"] < best.get("asr_after", 1e9):
            best = rec
    baseline = next((r for r in eval_rows if r.get("mode") == "no_op"), {})
    report = {
        "stage": "10_steering_eval",
        "layers": layers,
        "grid": eval_rows,
        "artifacts": artifacts,
        "best": best,
        "acceptance": {
            "best_reduces_asr_vs_no_op": bool(
                best
                and isinstance(baseline.get("asr_after"), (int, float))
                and best.get("asr_after", 1e9) < baseline["asr_after"]
            )
        },
    }
    io.write_json(cfg.out_dir / "steering_eval.json", report)
    io.write_csv(cfg.out_dir / "steering_grid.csv", eval_rows)
    io.write_csv(cfg.out_dir / "intervention_eval.csv", eval_rows)
    print(f"[10] combos={len(combos)} best={best}")
    return report
