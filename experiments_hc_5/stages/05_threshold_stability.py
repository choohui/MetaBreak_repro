from __future__ import annotations

import numpy as np

from experiments_hc_5.config import ExpConfig
from experiments_hc_5.core import io
from experiments_hc_5.core.labels import binary_label
from experiments_hc_5.core.metrics import binary_eval, oriented_auc, threshold_at_fpr
from experiments_hc_5.core.thresholds import orient_scores, predict_rule


def _split_eval(x: np.ndarray, names: list[str], rows: list[dict], rule: dict) -> dict:
    y = np.asarray([binary_label(r["letter"]) for r in rows], dtype=int)
    split = np.asarray([r["split"] for r in rows])
    out = {}
    for name in ("train", "val", "test"):
        mask = (split == name) & (y >= 0)
        out[name] = binary_eval(predict_rule(x[mask], names, rule), y[mask])
    return out


def _group_folds(groups: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    uniq = np.asarray(sorted(set(int(g) for g in groups)))
    if len(uniq) == 0:
        return []
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    k = max(2, min(k, len(uniq)))
    return [np.asarray(chunk, dtype=int) for chunk in np.array_split(uniq, k) if len(chunk)]


def _term_stability(
    x: np.ndarray,
    names: list[str],
    rows: list[dict],
    term: dict,
    fpr_target: float,
    seed: int,
    n_boot: int = 200,
) -> dict:
    y = np.asarray([binary_label(r["letter"]) for r in rows], dtype=int)
    split = np.asarray([r["split"] for r in rows])
    groups = np.asarray([int(r["sample_index"]) for r in rows], dtype=int)
    j = names.index(term["feature"])
    direction = term["direction"]
    score = orient_scores(x[:, j], direction)
    train = (split == "train") & (y >= 0)
    train_groups = np.asarray(sorted(set(int(g) for g in groups[train])))

    folds = []
    for fold_groups in _group_folds(train_groups, 5, seed):
        hold = train & np.isin(groups, fold_groups)
        fit = train & ~np.isin(groups, fold_groups)
        if fit.sum() == 0 or hold.sum() == 0 or len(np.unique(y[fit])) < 2:
            continue
        thr = threshold_at_fpr(score[fit], y[fit], fpr_target)
        pred = score[hold] >= thr
        auc, _ = oriented_auc(x[hold, j], y[hold])
        folds.append({
            "threshold": round(float(thr), 6),
            "held_auc": round(float(auc), 5) if not np.isnan(auc) else None,
            "held_eval": binary_eval(pred, y[hold]),
            "n_hold": int(hold.sum()),
        })

    boot = []
    if len(train_groups):
        rng = np.random.default_rng(seed + 17)
        by_group = {g: np.where(train & (groups == g))[0] for g in train_groups}
        for _ in range(n_boot):
            sampled = rng.choice(train_groups, size=len(train_groups), replace=True)
            idx = np.concatenate([by_group[int(g)] for g in sampled if len(by_group[int(g)])])
            if len(idx) == 0 or len(np.unique(y[idx])) < 2:
                continue
            boot.append(threshold_at_fpr(score[idx], y[idx], fpr_target))
    boot_arr = np.asarray(boot, dtype=float)
    boot_summary = {
        "n": int(len(boot_arr)),
        "mean": round(float(boot_arr.mean()), 6) if len(boot_arr) else None,
        "std": round(float(boot_arr.std()), 6) if len(boot_arr) else None,
        "q05": round(float(np.quantile(boot_arr, 0.05)), 6) if len(boot_arr) else None,
        "q95": round(float(np.quantile(boot_arr, 0.95)), 6) if len(boot_arr) else None,
    }
    return {
        "feature": term["feature"],
        "direction": direction,
        "selected_threshold": round(float(term["threshold"]), 6),
        "group_cv": folds,
        "bootstrap_threshold": boot_summary,
    }


def run(cfg: ExpConfig, lm=None) -> dict:
    rows = io.read_jsonl(cfg.out_dir / "balanced_tokens.jsonl")
    data = np.load(cfg.out_dir / "scalar_values.npz", allow_pickle=True)
    x = data["x"].astype(float)
    names = [str(x) for x in data["feature_names"].tolist()]
    rules = io.read_json(cfg.out_dir / "threshold_rules.json")

    per_rule = {}
    csv_rows = []
    for fpr_key, rule in (rules.get("selected") or {}).items():
        if not rule:
            continue
        fpr_target = float(rule.get("fpr_target", fpr_key))
        term_reports = [
            _term_stability(x, names, rows, term, fpr_target, cfg.seed + i)
            for i, term in enumerate(rule.get("terms", []))
            if term.get("feature") in names
        ]
        evals = _split_eval(x, names, rows, rule)
        per_rule[fpr_key] = {
            "rule_kind": rule.get("kind"),
            "split_eval": evals,
            "terms": term_reports,
        }
        for tr in term_reports:
            bs = tr["bootstrap_threshold"]
            csv_rows.append({
                "fpr_target": fpr_key,
                "feature": tr["feature"],
                "direction": tr["direction"],
                "selected_threshold": tr["selected_threshold"],
                "bootstrap_n": bs["n"],
                "bootstrap_mean": bs["mean"],
                "bootstrap_std": bs["std"],
                "bootstrap_q05": bs["q05"],
                "bootstrap_q95": bs["q95"],
            })

    out = {
        "stage": "05_threshold_stability",
        "n_rules": len(per_rule),
        "rules": per_rule,
    }
    io.write_json(cfg.out_dir / "threshold_stability.json", out)
    io.write_csv(cfg.out_dir / "threshold_stability.csv", csv_rows)
    print(f"[05] stability rules={len(per_rule)}")
    return out
