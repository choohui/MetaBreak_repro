from __future__ import annotations

import math
import re
from statistics import median

import numpy as np
from tqdm import tqdm

from experiments_hc_6.config import ExpConfig
from experiments_hc_6.core import io
from experiments_hc_6.core.capture import forward_capture_ids
from experiments_hc_6.core.defense import flagged_base_positions
from experiments_hc_6.core.interventions import first_token_metrics
from experiments_hc_6.core.labels import binary_label
from experiments_hc_6.core.model import get_model
from experiments_hc_6.core.template import content_bounds, find_literal_assistant_spans, find_regular_assistant_spans
from experiments_hc_6.core.thresholds import orient_scores, predict_rule


NEUTRAL_TEXTS = [
    " ", ".", ",", ":", ";", "-", "_", "\n", " the", " a", " and", " text",
    " neutral", " placeholder", " token", " note", " item",
]


def _default_rule(rules: dict) -> tuple[str, dict]:
    if "0.01" in rules.get("selected", {}) and rules["selected"]["0.01"]:
        return "0.01", rules["selected"]["0.01"]
    for k, v in rules.get("selected", {}).items():
        if v:
            return k, v
    raise RuntimeError("no selected threshold rule")


def _strip_feature_name(name: str) -> str:
    for prefix in ("rz__", "benign_lowtail__", "benign_hightail__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _layer_from_name(name: str) -> int | None:
    m = re.search(r"_L(\d+)$", _strip_feature_name(name))
    return int(m.group(1)) if m else None


def _fit_hidden_refs(rows: list[dict], hidden: np.ndarray) -> dict:
    row_ids = np.asarray([int(r["row_id"]) for r in rows], dtype=int)
    h = hidden[row_ids].astype(np.float64)
    train = np.asarray([r["split"] == "train" for r in rows])
    letters = np.asarray([r["letter"] for r in rows])
    atk = train & np.isin(letters, ["B", "D"])
    ben = train & np.isin(letters, ["C", "E", "F", "G"])
    if atk.sum() == 0 or ben.sum() == 0:
        raise RuntimeError("cannot fit hidden refs without train attack and benign rows")
    mu_attack = h[atk].mean(axis=0)
    mu_benign = h[ben].mean(axis=0)
    var_benign = h[ben].var(axis=0) + 1e-4
    direction = mu_attack - mu_benign
    norm = np.linalg.norm(direction, axis=1) + 1e-12
    return {
        "mu_attack": mu_attack,
        "mu_benign": mu_benign,
        "var_benign": var_benign,
        "attack_minus_benign": direction,
        "attack_minus_benign_norm": norm,
    }


def _score_feature(name: str, hvec: np.ndarray, token_id: int, refs: dict, repl_ids: set[int], special_ids: set[int]) -> float:
    base = _strip_feature_name(name)
    layer = _layer_from_name(base)
    if base == "exact_special_token":
        return float(int(token_id) in special_ids)
    if base in {"known_replacement_token", "l2_neighbor_proxy"}:
        return float(int(token_id) in repl_ids)
    if layer is None:
        return 0.0
    h = hvec[layer].astype(np.float64)
    if base.startswith("proto_attack_minus_benign_L"):
        d = refs["attack_minus_benign"][layer]
        return float(h @ d / refs["attack_minus_benign_norm"][layer])
    if base.startswith("cos_to_attack_L"):
        mu = refs["mu_attack"][layer]
        return float((h @ mu) / ((np.linalg.norm(h) + 1e-12) * (np.linalg.norm(mu) + 1e-12)))
    if base.startswith("dist_benign_centroid_L"):
        return float(np.linalg.norm(h - refs["mu_benign"][layer]))
    if base.startswith("diag_mahal_benign_L"):
        return float(np.sqrt(np.mean((h - refs["mu_benign"][layer]) ** 2 / refs["var_benign"][layer])))
    return 0.0


def _predict_from_rule(scores: np.ndarray, rule: dict) -> np.ndarray:
    preds = []
    for j, term in enumerate(rule.get("terms", [])):
        s = orient_scores(scores[:, j], term["direction"])
        preds.append(s >= float(term["threshold"]))
    if not preds:
        return np.zeros(scores.shape[0], dtype=bool)
    mat = np.vstack(preds)
    if rule.get("kind") in {"and2", "two_of_k"}:
        return mat.sum(axis=0) >= min(2, mat.shape[0])
    return mat.any(axis=0)


def _encode_one(tokenizer, text: str) -> int | None:
    try:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    except Exception:
        return None
    return int(ids[0]) if len(ids) == 1 else None


def _token_text(tokenizer, tid: int) -> str:
    try:
        return str(tokenizer.convert_ids_to_tokens(int(tid)))
    except Exception:
        try:
            return str(tokenizer.decode([int(tid)]))
        except Exception:
            return f"<id:{int(tid)}>"


def _candidate_sources(lm, rows: list[dict], cfg: ExpConfig) -> tuple[list[dict], list[list[int]]]:
    excluded = set(int(x) for x in lm.template.header_ids + lm.template.target_token_ids)
    candidates: dict[int, dict] = {}

    def add(tid: int | None, source: str, slot: int | None = None) -> None:
        if tid is None or int(tid) in excluded or int(tid) < 0:
            return
        rec = candidates.setdefault(int(tid), {
            "id": int(tid),
            "text": _token_text(lm.tokenizer, int(tid)),
            "sources": [],
            "slots": [],
        })
        if source not in rec["sources"]:
            rec["sources"].append(source)
        if slot is not None and slot not in rec["slots"]:
            rec["slots"].append(slot)

    add(getattr(lm.tokenizer, "unk_token_id", None), "unk")
    add(getattr(lm.tokenizer, "eos_token_id", None), "eos")
    for text in NEUTRAL_TEXTS:
        add(_encode_one(lm.tokenizer, text), "neutral_text")

    vocab = getattr(lm.tokenizer, "get_vocab", lambda: {})()
    for tok, tid in list(vocab.items())[: max(0, len(vocab))]:
        if "reserved_special" in str(tok):
            add(int(tid), "reserved_special")
            if len([r for r in candidates.values() if "reserved_special" in r["sources"]]) >= cfg.mask_candidate_k:
                break

    slot_lists: list[list[int]] = [[], [], []]
    if not lm.is_mock and hasattr(lm.model, "get_input_embeddings"):
        import torch

        emb = lm.model.get_input_embeddings().weight.detach().float()
        train_benign = [
            r for r in rows
            if r["split"] == "train" and r["letter"] in ("C", "E", "F", "G")
        ]
        for slot in range(3):
            ids = [int(r["token_id"]) for r in train_benign if r.get("target_token_index") == slot]
            if not ids:
                ids = [int(r["token_id"]) for r in train_benign]
            if not ids:
                continue
            center = emb[torch.tensor(ids, dtype=torch.long, device=emb.device)].mean(dim=0)
            dist = torch.linalg.vector_norm(emb - center, dim=1)
            if excluded:
                idx = torch.tensor(sorted(x for x in excluded if 0 <= x < dist.numel()), device=dist.device)
                dist[idx] = float("inf")
            k = min(max(cfg.mask_candidate_k, cfg.mask_top_n), int(dist.numel()))
            vals, idxs = torch.topk(-dist, k=k, largest=True)
            for tid in [int(x) for x in idxs.tolist()]:
                add(tid, "embedding_benign_nearest", slot)
                slot_lists[slot].append(tid)
    else:
        for slot, defaults in enumerate(([791, 2579, 6437], [791, 2579, 6437], [791, 2579, 6437])):
            for tid in defaults:
                add(tid, "mock_neutral", slot)
                slot_lists[slot].append(tid)

    singles = sorted(candidates.values(), key=lambda r: (0 if "unk" in r["sources"] or "eos" in r["sources"] else 1, r["id"]))
    return singles[: max(cfg.mask_candidate_k, cfg.mask_top_n)], slot_lists


def _candidate_records(lm, rows: list[dict], cfg: ExpConfig) -> list[dict]:
    singles, slot_lists = _candidate_sources(lm, rows, cfg)
    out = []
    seen = set()
    for rec in singles:
        key = ("single", tuple([rec["id"]]))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "candidate": f"single_{rec['id']}",
            "kind": "single",
            "candidate_ids": [int(rec["id"])],
            "candidate_text": rec["text"],
            "sources": rec["sources"],
        })
    unk = getattr(lm.tokenizer, "unk_token_id", None) or getattr(lm.tokenizer, "eos_token_id", None)
    if unk is not None:
        out.append({
            "candidate": "tuple_unk_or_eos",
            "kind": "tuple",
            "candidate_ids": [int(unk), int(unk), int(unk)],
            "candidate_text": "unk/eos per slot",
            "sources": ["unk_or_eos_baseline"],
        })
    for rank in range(cfg.mask_top_n):
        ids = []
        for slot in range(3):
            pool = slot_lists[slot] or [r["id"] for r in singles]
            if not pool:
                break
            ids.append(int(pool[min(rank, len(pool) - 1)]))
        if len(ids) == 3:
            key = ("tuple", tuple(ids))
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "candidate": "tuple_" + "_".join(str(x) for x in ids),
                "kind": "tuple",
                "candidate_ids": ids,
                "candidate_text": " | ".join(_token_text(lm.tokenizer, x) for x in ids),
                "sources": ["slot_embedding_benign_nearest"],
            })
    return out


def _flag_infos(rows: list[dict], pred: np.ndarray, split: str, letters: set[str]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r, p in zip(rows, pred):
        if not p or r["split"] != split or r["letter"] not in letters:
            continue
        sid = int(r["sample_index"])
        info = {
            "base_position": int(r["base_position"]),
            "position": int(r["position"]),
            "target_token_index": r.get("target_token_index"),
            "letter": r["letter"],
            "row_id": int(r["row_id"]),
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


def _span_removed(lm, ids: list[int], infos: list[dict]) -> bool:
    lo, hi = content_bounds(lm.tokenizer, ids)
    spans = find_literal_assistant_spans(ids, lm.template) + find_regular_assistant_spans(ids, lm.template)
    active_slots = set()
    for span in spans:
        if lo <= span.start and span.start + len(span.ids) <= hi:
            active_slots.update(span.start + int(off) for off in lm.template.replace_positions)
    return not any(int(info["base_position"]) in active_slots for info in infos)


def _eval_candidate(candidate: dict, lm, rows: list[dict], inputs_by_id: dict[int, dict],
                    attack_infos: dict[int, list[dict]], benign_infos: dict[int, list[dict]],
                    rule: dict, refs: dict, repl_ids: set[int], cfg: ExpConfig) -> dict:
    terms = rule.get("terms", [])
    special_ids = set(int(x) for x in lm.template.target_token_ids)
    after_preds = []
    span_removed = []
    for sid, infos in tqdm(list(attack_infos.items())[: cfg.semantic_eval_n], desc=f"[07] {candidate['candidate']}", leave=False):
        rec = inputs_by_id.get(sid)
        if not rec:
            continue
        after_ids = _apply_candidate(rec["input_ids"], infos, candidate)
        cap = forward_capture_ids(lm, after_ids)
        scores = []
        for info in infos:
            pos = int(info["position"])
            if pos >= cap.hidden.shape[0]:
                continue
            token_id = int(after_ids[pos])
            scores.append([
                _score_feature(t["feature"], cap.hidden[pos], token_id, refs, repl_ids, special_ids)
                for t in terms
            ])
        if scores:
            after_preds.extend(_predict_from_rule(np.asarray(scores, dtype=float), rule).tolist())
        span_removed.append(_span_removed(lm, after_ids, infos))

    semantic_rows = list(benign_infos.items())[: cfg.semantic_eval_n]
    kls = []
    top1 = []
    for sid, infos in semantic_rows:
        rec = inputs_by_id.get(sid)
        if not rec:
            continue
        after_ids = _apply_candidate(rec["input_ids"], infos, candidate)
        m = first_token_metrics(lm, rec["input_ids"], after_ids)
        kls.append(float(m["kl"]))
        top1.append(bool(m["top1_match"]))

    clear_rate = 1.0 - float(np.mean(after_preds)) if after_preds else None
    span_rate = float(np.mean(span_removed)) if span_removed else None
    top1_rate = float(np.mean(top1)) if top1 else 1.0
    kl_med = float(median(kls)) if kls else 0.0
    score = (clear_rate or 0.0) + 0.25 * (span_rate or 0.0) + 0.25 * top1_rate - min(1.0, kl_med)
    return {
        **candidate,
        "n_attack_eval": int(len(after_preds)),
        "val_attack_cleared_rate": round(clear_rate, 5) if clear_rate is not None else None,
        "injected_span_removed_rate": round(span_rate, 5) if span_rate is not None else None,
        "semantic_top1_match": round(top1_rate, 5),
        "semantic_kl_median": round(kl_med, 6),
        "rank_score": round(float(score), 6),
    }


def run(cfg: ExpConfig, lm=None) -> dict:
    lm = get_model(cfg, lm)
    rows = io.read_jsonl(cfg.out_dir / "balanced_tokens.jsonl")
    inputs = io.read_jsonl(cfg.inputs_path)
    inputs_by_id = {int(r["sample_index"]): r for r in inputs}
    hidden = np.load(cfg.features_path, allow_pickle=True)["hidden"]
    data = np.load(cfg.out_dir / "scalar_values.npz", allow_pickle=True)
    x = data["x"].astype(float)
    names = [str(x) for x in data["feature_names"].tolist()]
    rules = io.read_json(cfg.out_dir / "threshold_rules.json")
    fpr_key, rule = _default_rule(rules)
    pred = predict_rule(x, names, rule)
    attack_infos = _flag_infos(rows, pred, "val", {"B", "D"})
    benign_infos = _flag_infos(rows, pred, "val", {"C", "E", "F", "G"})
    if not benign_infos:
        # Keep the semantic metric defined even when the detector has zero benign false positives.
        for r in inputs:
            if r.get("letter") in ("C", "E", "F", "G"):
                benign_infos[int(r["sample_index"])] = []
            if len(benign_infos) >= cfg.semantic_eval_n:
                break
    refs = _fit_hidden_refs(rows, hidden)
    repl_ids = set(int(x) for x in io.load_replacement(cfg.replacement_path).get("best_triple_ids", []))
    candidates = _candidate_records(lm, rows, cfg)
    io.write_json(cfg.out_dir / "mask_candidates.json", {
        "selected_fpr": fpr_key,
        "selected_rule": rule,
        "n_candidates": len(candidates),
        "candidates": candidates,
    })
    eval_rows = [
        _eval_candidate(c, lm, rows, inputs_by_id, attack_infos, benign_infos, rule, refs, repl_ids, cfg)
        for c in candidates
    ]
    eval_rows.sort(key=lambda r: r["rank_score"], reverse=True)
    io.write_json(cfg.out_dir / "mask_candidate_eval.json", {
        "stage": "07_mask_candidate_search",
        "selected_fpr": fpr_key,
        "n_val_attack_flagged_prompts": len(attack_infos),
        "n_val_benign_flagged_prompts": len([k for k, v in benign_infos.items() if v]),
        "top": eval_rows[: cfg.mask_top_n],
    })
    io.write_csv(cfg.out_dir / "mask_candidate_eval.csv", eval_rows)
    print(f"[07] candidates={len(candidates)} top={eval_rows[0]['candidate'] if eval_rows else None}")
    return {"n_candidates": len(candidates), "top": eval_rows[: cfg.mask_top_n]}
