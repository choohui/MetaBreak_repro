from __future__ import annotations

import numpy as np

from experiments_hc_5.config import ExpConfig
from experiments_hc_5.core import io
from experiments_hc_5.core.defense import evaluate_rule_on_split, flagged_base_positions, generate_defended
from experiments_hc_5.core.labels import binary_label
from experiments_hc_5.core.model import get_model
from experiments_hc_5.core.thresholds import predict_rule


def _default_rule(rules: dict) -> tuple[str, dict]:
    if "0.01" in rules["selected"] and rules["selected"]["0.01"]:
        return "0.01", rules["selected"]["0.01"]
    for k, v in rules["selected"].items():
        if v:
            return k, v
    raise RuntimeError("no selected threshold rule")


def _asr(rows: list[dict]) -> float | None:
    atk = [r for r in rows if r.get("letter") in ("B", "D")]
    if not atk:
        return None
    return round(sum(bool(r.get("success")) for r in atk) / len(atk), 5)


def run(cfg: ExpConfig, lm=None) -> dict:
    rows = io.read_jsonl(cfg.out_dir / "balanced_tokens.jsonl")
    inputs = io.read_jsonl(cfg.inputs_path)
    responses = io.read_jsonl(cfg.out_dir / "responses.jsonl") if (cfg.out_dir / "responses.jsonl").exists() else []
    data = np.load(cfg.out_dir / "scalar_values.npz", allow_pickle=True)
    x = data["x"].astype(float)
    names = [str(x) for x in data["feature_names"].tolist()]
    rules = io.read_json(cfg.out_dir / "threshold_rules.json")
    fpr_key, rule = _default_rule(rules)
    split_mask = np.asarray([r["split"] == "test" and binary_label(r["letter"]) >= 0 for r in rows])
    pred_test = predict_rule(x[split_mask], names, rule)
    test_rows = [r for r, m in zip(rows, split_mask) if m]
    flags = flagged_base_positions(test_rows, pred_test)
    token_eval = evaluate_rule_on_split(x, names, rows, rule, split="test")
    split_by_group = {str(r["sample_index"]): r["split"] for r in rows}
    test_inputs = [r for r in inputs if split_by_group.get(str(r["sample_index"])) == "test"
                   and r.get("letter") in ("B", "C", "D", "E", "F", "G")]
    test_responses = [r for r in responses if split_by_group.get(str(r["sample_index"])) == "test"]

    out = {
        "selected_fpr": fpr_key,
        "selected_rule": rule,
        "token_eval": token_eval,
        "actions": {},
    }
    if cfg.skip_generation:
        for action in cfg.defense_actions:
            out["actions"][action] = {"generation_skipped": True}
    else:
        lm = get_model(cfg, lm)
        for action in cfg.defense_actions:
            defended = generate_defended(lm, test_inputs, flags, action, cfg.max_new_tokens)
            io.write_jsonl(cfg.out_dir / f"defended_{action}.jsonl", defended)
            benign = [r for r in test_inputs if r.get("letter") in ("C", "E", "F", "G")]
            benign_flagged = sum(bool(flags.get(int(r["sample_index"]))) for r in benign)
            succ_before = _asr(test_responses)
            succ_after = _asr(defended)
            successful_before = {int(r["sample_index"]) for r in test_responses
                                 if r.get("letter") in ("B", "D") and r.get("success")}
            blocked_success = sum(bool(flags.get(sid)) for sid in successful_before)
            out["actions"][action] = {
                "n_prompts": len(test_inputs),
                "asr_before": succ_before,
                "asr_after": succ_after,
                "prompt_fpr": round(benign_flagged / max(1, len(benign)), 5),
                "block_rate_among_successful": round(blocked_success / max(1, len(successful_before)), 5)
                if successful_before else None,
            }
    io.write_json(cfg.out_dir / "defense_eval.json", out)
    print(f"[06] selected fpr={fpr_key} action eval keys={list(out['actions'])}")
    return out
