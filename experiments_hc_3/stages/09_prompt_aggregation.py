"""Stage 09 - Prompt-level aggregation cascade.

Uses the high-recall Active SinkProbe token scores from stage 08, then makes a
prompt-level decision from max/count/top-k score summaries. This tests the idea
that token-level FPR can be tolerated if the final defense is prompt-level.
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
from experiments_hc_3.core import artifacts, io, modeling  # noqa: E402


def _load_scores(cfg: ExpConfig, offset: int) -> list[dict]:
    path = cfg.pos_dir(offset) / "active_sinkprobe_scores.jsonl"
    if not path.exists():
        raise SystemExit(f"[09] missing {path}; run stage 08 first.")
    return io.read_jsonl(path)


def _prompt_rows(records: list[dict], threshold: float | None = None) -> tuple[list[dict], np.ndarray, np.ndarray]:
    by_sample: dict[int, list[dict]] = {}
    for r in records:
        if r.get("label") in (0, 1) and r.get("cv_score") is not None:
            by_sample.setdefault(int(r["sample_index"]), []).append(r)

    rows = []
    feats = []
    labels = []
    for sid, rs in sorted(by_sample.items()):
        scores = np.array([float(r["cv_score"]) for r in rs], dtype=np.float64)
        y = np.array([int(r["label"]) for r in rs], dtype=int)
        label = int((y == 1).any())
        sorted_scores = np.sort(scores)[::-1]
        top3 = sorted_scores[: min(3, len(sorted_scores))]
        count_hi = float((scores >= threshold).sum()) if threshold is not None else 0.0
        frac_hi = count_hi / max(1, len(scores))
        letters = sorted({r["letter"] for r in rs})
        variants = sorted({str(r.get("variant")) for r in rs})
        rows.append({
            "sample_index": sid,
            "label": label,
            "letters": ",".join(letters),
            "variants": ",".join(variants),
            "n_tokens": len(rs),
        })
        feats.append([
            float(scores.max()),
            float(scores.mean()),
            float(top3.mean()),
            float(scores.std()),
            count_hi,
            frac_hi,
        ])
        labels.append(label)
    return rows, np.asarray(feats, dtype=np.float64), np.asarray(labels, dtype=int)


def _split_prompt(rows: list[dict], seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(rows)
    if n < 6:
        mask = np.ones(n, dtype=bool)
        return mask, mask
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(n / 3)))
    test = set(int(i) for i in idx[:n_test])
    is_test = np.array([i in test for i in range(n)], dtype=bool)
    return ~is_test, is_test


def _prompt_eval(rows: list[dict], pred: np.ndarray, success: set[int]) -> dict:
    attack = np.array([r["label"] == 1 for r in rows], dtype=bool)
    benign = ~attack
    blocked_attack = pred & attack
    fpr = float((pred & benign).sum() / max(1, benign.sum()))
    block_rate = float(blocked_attack.sum() / max(1, attack.sum()))
    attack_ids = {int(r["sample_index"]) for r in rows if r["label"] == 1}
    blocked_ids = {int(r["sample_index"]) for r, p in zip(rows, pred) if r["label"] == 1 and p}
    succeeded = attack_ids & success
    still = succeeded - blocked_ids
    return {
        "n_prompt": len(rows),
        "n_attack": int(attack.sum()),
        "n_benign": int(benign.sum()),
        "block_rate": round(block_rate, 5),
        "prompt_fpr": round(fpr, 5),
        "n_succeeded": len(succeeded),
        "asr_before": round(len(succeeded) / max(1, len(attack_ids)), 5) if attack_ids else None,
        "asr_after": round(len(still) / max(1, len(attack_ids)), 5) if attack_ids else None,
        "block_rate_among_successful": round((len(succeeded) - len(still)) / max(1, len(succeeded)), 5)
        if succeeded else None,
    }


def _run_offset(cfg: ExpConfig, offset: int) -> dict:
    records = _load_scores(cfg, offset)
    valid = [r for r in records if r.get("label") in (0, 1) and r.get("cv_score") is not None]
    token_scores = np.array([float(r["cv_score"]) for r in valid], dtype=np.float64)
    token_y = np.array([int(r["label"]) for r in valid], dtype=int)
    token_thr = modeling.threshold_for_recall(token_scores, token_y, cfg.token_recall)

    prows, px, py = _prompt_rows(records, threshold=token_thr)
    is_train, is_test = _split_prompt(prows, cfg.seed)
    success = artifacts.success_set(cfg.out_dir, cfg.asr_judge)

    # Strategy A: candidate count/max rule from high-recall token threshold.
    pred_rule = px[:, 4] >= 1.0
    rule_eval = _prompt_eval([r for r, t in zip(prows, is_test) if t], pred_rule[is_test], success)

    # Strategy B: prompt-level logistic aggregation at target prompt FPR.
    if len(np.unique(py[is_train])) < 2:
        pred_lr = pred_rule.copy()
        prompt_thr = None
        lr_auc = None
    else:
        model = modeling.fit_linear_model(px[is_train], py[is_train], c=1.0)
        train_scores = modeling.predict_score(model, px[is_train])
        all_scores = modeling.predict_score(model, px)
        prompt_thr = modeling.threshold_for_fpr(train_scores, py[is_train], cfg.fpr)
        pred_lr = all_scores >= prompt_thr
        lr_auc = modeling.metrics.roc_auc(all_scores[is_test], py[is_test]) if len(np.unique(py[is_test])) == 2 else None

    lr_eval = _prompt_eval([r for r, t in zip(prows, is_test) if t], pred_lr[is_test], success)

    report = {
        "stage": "09_prompt_aggregation",
        "pos_offset": offset,
        "token_recall_target": cfg.token_recall,
        "token_threshold": token_thr,
        "prompt_fpr_target": cfg.fpr,
        "n_prompts": len(prows),
        "n_train": int(is_train.sum()),
        "n_test": int(is_test.sum()),
        "feature_names": ["max_score", "mean_score", "top3_mean", "std_score",
                          "count_above_token_threshold", "frac_above_token_threshold"],
        "strategies": {
            "token_any_prompt_block": rule_eval,
            "prompt_logreg": {**lr_eval, "threshold": prompt_thr, "auc": lr_auc},
        },
    }
    pos_dir = cfg.pos_dir(offset)
    io.write_json(pos_dir / "prompt_aggregation_report.json", report)
    io.write_text(pos_dir / "prompt_aggregation_report.md", _md(report))
    print(f"[09] pos{offset}: any-token block={rule_eval['block_rate']} FPR={rule_eval['prompt_fpr']} "
          f"logreg block={lr_eval['block_rate']} FPR={lr_eval['prompt_fpr']}")
    return report


def _md(r: dict) -> str:
    lines = [
        f"# Stage 09 - Prompt Aggregation (pos_offset={r['pos_offset']})",
        "",
        f"- token recall target: {r['token_recall_target']}",
        f"- token threshold: {r['token_threshold']}",
        f"- prompt FPR target: {r['prompt_fpr_target']}",
        f"- train/test prompts: {r['n_train']} / {r['n_test']}",
        "",
        "| strategy | block | prompt FPR | ASR before | ASR after | block@succ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in r["strategies"].items():
        lines.append(f"| {name} | {s['block_rate']} | {s['prompt_fpr']} | "
                     f"{s['asr_before']} | {s['asr_after']} | {s['block_rate_among_successful']} |")
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

