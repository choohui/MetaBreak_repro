from __future__ import annotations

import re
from statistics import median

import numpy as np

from experiments_hc_6.config import ExpConfig
from experiments_hc_6.core import io
from experiments_hc_6.core.defense import clean_ids, evaluate_rule_on_split, flagged_base_positions
from experiments_hc_6.core.interventions import first_token_metrics
from experiments_hc_6.core.model import get_model, refusal_success
from experiments_hc_6.core.thresholds import predict_rule


BASELINE_ACTIONS = ["no_op", "unk_or_eos_mask", "drop_token", "drop_token_pm1", "drop_detected_span", "prompt_block"]


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


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:80]


def _flag_infos(rows: list[dict], pred: np.ndarray) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r, p in zip(rows, pred):
        if not p:
            continue
        sid = int(r["sample_index"])
        info = {
            "base_position": int(r["base_position"]),
            "target_token_index": r.get("target_token_index"),
            "letter": r["letter"],
        }
        if info not in out.setdefault(sid, []):
            out[sid].append(info)
    return out


def _apply_candidate(input_ids: list[int], infos: list[dict], candidate: dict) -> list[int]:
    ids = list(int(x) for x in input_ids)
    cids = [int(x) for x in candidate["candidate_ids"]]
    for info in infos:
        pos = int(info["base_position"])
        if not (0 <= pos < len(ids)):
            continue
        if candidate["kind"] == "tuple":
            slot = info.get("target_token_index")
            idx = int(slot) if slot is not None and 0 <= int(slot) < len(cids) else 0
            ids[pos] = cids[idx]
        else:
            ids[pos] = cids[0]
    return ids


def _ids_for_action(lm, rec: dict, infos: list[dict], action: dict) -> list[int]:
    input_ids = [int(x) for x in rec["input_ids"]]
    positions = {int(i["base_position"]) for i in infos}
    kind = action["kind"]
    if kind == "candidate":
        return _apply_candidate(input_ids, infos, action["candidate"])
    if kind == "unk_or_eos_mask":
        return clean_ids(input_ids, positions, "mask_token", lm.tokenizer, lm.template)
    if kind in {"drop_token", "drop_token_pm1", "drop_detected_span", "no_op"}:
        return clean_ids(input_ids, positions, kind, lm.tokenizer, lm.template)
    return input_ids


def _semantic_summary(rows: list[dict], baseline: dict[int, dict], current: list[dict], metrics: list[dict]) -> dict:
    benign_ids = {int(r["sample_index"]) for r in rows if r.get("letter") in ("C", "E", "F", "G")}
    cur = {int(r["sample_index"]): r for r in current}
    top1 = [m["top1_match"] for m in metrics]
    kls = [float(m["kl"]) for m in metrics]
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
        "semantic_top1_match": round(float(np.mean(top1)), 5) if top1 else 1.0,
        "semantic_kl_median": round(float(median(kls)), 6) if kls else 0.0,
        "semantic_label_match": round(float(np.mean(label_matches)), 5) if label_matches else None,
        "response_length_ratio_median": round(float(median(ratios)), 5) if ratios else None,
    }


def _run_action(lm, cfg: ExpConfig, test_inputs: list[dict], info_by_sample: dict[int, list[dict]],
                action: dict, baseline_by_sample: dict[int, dict] | None) -> tuple[list[dict], list[dict]]:
    out = []
    semantic_metrics = []
    for rec in test_inputs:
        sid = int(rec["sample_index"])
        infos = info_by_sample.get(sid, [])
        blocked = action["kind"] == "prompt_block" and bool(infos)
        if blocked:
            response = ""
            success = False
        else:
            after_ids = _ids_for_action(lm, rec, infos, action)
            if rec.get("letter") in ("C", "E", "F", "G") and len(semantic_metrics) < cfg.semantic_eval_n:
                semantic_metrics.append(first_token_metrics(lm, rec["input_ids"], after_ids))
            response = lm.generate_from_ids(after_ids, max_new_tokens=cfg.max_new_tokens)
            success = refusal_success(response)
        out.append({
            "sample_index": sid,
            "variant": rec.get("variant"),
            "letter": rec.get("letter"),
            "action": action["name"],
            "blocked": bool(blocked),
            "n_flagged_positions": len(infos),
            "success": bool(success),
            "response": response,
        })
    return out, semantic_metrics


def run(cfg: ExpConfig, lm=None) -> dict:
    lm = get_model(cfg, lm)
    rows = io.read_jsonl(cfg.out_dir / "balanced_tokens.jsonl")
    inputs = io.read_jsonl(cfg.inputs_path)
    responses = io.read_jsonl(cfg.out_dir / "responses.jsonl") if (cfg.out_dir / "responses.jsonl").exists() else []
    data = np.load(cfg.out_dir / "scalar_values.npz", allow_pickle=True)
    x = data["x"].astype(float)
    names = [str(x) for x in data["feature_names"].tolist()]
    rules = io.read_json(cfg.out_dir / "threshold_rules.json")
    fpr_key, rule = _default_rule(rules)
    split_mask = np.asarray([r["split"] == "test" for r in rows])
    test_rows = [r for r, m in zip(rows, split_mask) if m]
    pred_test = predict_rule(x[split_mask], names, rule)
    info_by_sample = _flag_infos(test_rows, pred_test)
    split_by_group = {str(r["sample_index"]): r["split"] for r in rows}
    test_inputs = [
        r for r in inputs
        if split_by_group.get(str(r["sample_index"])) == "test"
        and r.get("letter") in ("B", "C", "D", "E", "F", "G")
    ]
    benign = [r for r in test_inputs if r.get("letter") in ("C", "E", "F", "G")]
    candidates = (io.read_json(cfg.out_dir / "mask_candidate_eval.json").get("top") or [])[: cfg.mask_top_n]
    actions = [{"name": a, "kind": a} for a in BASELINE_ACTIONS]
    for c in candidates:
        if set(c.get("sources", [])) & {"unk", "eos", "unk_or_eos_baseline"}:
            continue
        actions.append({"name": "mask__" + _slug(c["candidate"]), "kind": "candidate", "candidate": c})

    out = {
        "selected_fpr": fpr_key,
        "selected_rule": rule,
        "token_eval": evaluate_rule_on_split(x, names, rows, rule, split="test"),
        "actions": {},
    }
    rows_for_csv = []
    baseline_by_sample: dict[int, dict] = {}
    if cfg.skip_generation:
        for action in actions:
            rec = {"generation_skipped": True}
            out["actions"][action["name"]] = rec
            rows_for_csv.append({"action": action["name"], **rec})
    else:
        for action in actions:
            defended, sem = _run_action(lm, cfg, test_inputs, info_by_sample, action, baseline_by_sample)
            io.write_jsonl(cfg.out_dir / f"defended_{_slug(action['name'])}.jsonl", defended)
            if action["name"] == "no_op":
                baseline_by_sample = {int(r["sample_index"]): r for r in defended}
            benign_flagged = sum(bool(info_by_sample.get(int(r["sample_index"]))) for r in benign)
            successful_before = {
                int(r["sample_index"]) for r in responses
                if r.get("letter") in ("B", "D") and r.get("success")
            }
            blocked_success = sum(bool(info_by_sample.get(sid)) for sid in successful_before)
            summary = {
                "n_prompts": len(test_inputs),
                "asr_before": _asr(responses),
                "asr_after": _asr(defended),
                "prompt_fpr": round(benign_flagged / max(1, len(benign)), 5),
                "block_rate_among_successful": round(blocked_success / max(1, len(successful_before)), 5)
                if successful_before else None,
            }
            summary.update(_semantic_summary(test_inputs, baseline_by_sample, defended, sem))
            if action["kind"] == "candidate":
                summary["candidate_ids"] = action["candidate"]["candidate_ids"]
                summary["candidate_text"] = action["candidate"]["candidate_text"]
                summary["candidate_sources"] = action["candidate"].get("sources", [])
            out["actions"][action["name"]] = summary
            rows_for_csv.append({"action": action["name"], **summary})

    unk = out["actions"].get("unk_or_eos_mask", {})
    drop_pm1 = out["actions"].get("drop_token_pm1", {})
    best_non_unk = None
    for name, rec in out["actions"].items():
        if not name.startswith("mask__") or not isinstance(rec.get("asr_after"), (int, float)):
            continue
        if set(rec.get("candidate_sources", [])) & {"unk", "eos", "unk_or_eos_baseline"}:
            continue
        if best_non_unk is None or rec["asr_after"] < best_non_unk[1].get("asr_after", 1e9):
            best_non_unk = (name, rec)
    out["best_non_unk_mask"] = best_non_unk[0] if best_non_unk else None
    out["acceptance"] = {
        "non_unk_beats_unk_or_eos": bool(
            best_non_unk
            and isinstance(unk.get("asr_after"), (int, float))
            and best_non_unk[1]["asr_after"] < unk["asr_after"]
        ),
        "non_unk_drift_below_drop_pm1": bool(
            best_non_unk
            and isinstance(drop_pm1.get("semantic_kl_median"), (int, float))
            and best_non_unk[1].get("semantic_kl_median", 1e9) < drop_pm1["semantic_kl_median"]
        ),
    }
    io.write_json(cfg.out_dir / "mask_eval.json", out)
    io.write_csv(cfg.out_dir / "mask_eval.csv", rows_for_csv)
    # Compatibility with hc5 report readers: the richer hc6 report uses mask_eval.json.
    io.write_json(cfg.out_dir / "defense_eval.json", out)
    print(f"[08] actions={len(actions)} best_non_unk={out['best_non_unk_mask']}")
    return out
