"""Stage 10 - Two-branch cascade.

Train separate Active SinkProbe detectors for:
  * B branch: malicious mimicry regular token vs benign controls
  * D branch: literal special-token misuse vs benign controls

Then combine both branch scores with max(score_B, score_D) and evaluate a
prompt-level block decision on a held-out prompt split.
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
from experiments_hc_3.core import artifacts, cascade, io, metrics, modeling  # noqa: E402
from experiments_hc_3.core.splits import holdout_mask  # noqa: E402


def _load_feature_bundle(cfg: ExpConfig, offset: int):
    pdir = cfg.pos_dir(offset)
    feat_path = pdir / "active_sinkprobe_features.npz"
    score_path = pdir / "active_sinkprobe_scores.jsonl"
    if not feat_path.exists() or not score_path.exists():
        raise SystemExit(f"[10] missing stage-08 outputs in {pdir}; run stage 08 first.")
    npz = np.load(feat_path, allow_pickle=True)
    records = io.read_jsonl(score_path)
    return npz["x"], records


def _branch_labels(letters: np.ndarray, branch: str) -> np.ndarray:
    y = np.full(len(letters), -1, dtype=int)
    if branch == "B":
        y[letters == "B"] = 1
        y[np.isin(letters, ["C", "E", "F", "G"])] = 0
    elif branch == "D":
        y[letters == "D"] = 1
        y[np.isin(letters, ["C", "E", "F", "G"])] = 0
    else:
        raise ValueError(branch)
    return y


def _fit_branch(x: np.ndarray, y: np.ndarray, train: np.ndarray):
    m = train & modeling.valid_binary_mask(y)
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return None
    return modeling.fit_linear_model(x[m], y[m], c=0.5)


def _score_branch(model, x: np.ndarray) -> np.ndarray:
    if model is None:
        return np.zeros(x.shape[0], dtype=np.float64)
    return modeling.predict_score(model, x)


def _branch_auc(score: np.ndarray, y: np.ndarray, test: np.ndarray) -> float | None:
    m = test & modeling.valid_binary_mask(y)
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return None
    return round(float(metrics.roc_auc(score[m], y[m])), 5)


def _prompt_eval(rows: list[dict], pred: np.ndarray, success: set[int]) -> dict:
    per_type = cascade.per_type_rates(rows, pred)
    prompt = cascade.prompt_block_and_asr(rows, pred, success)
    benign = np.array([r["letter"] in ("C", "E", "F", "G") for r in rows], dtype=bool)
    prompt_ids = sorted({int(r["sample_index"]) for r in rows})
    blocked_prompt = {
        int(r["sample_index"]) for r, p in zip(rows, pred)
        if p and r["letter"] in ("B", "C", "D", "E", "F", "G")
    }
    benign_prompt_ids = {
        int(r["sample_index"]) for r in rows if r["letter"] in ("C", "E", "F", "G")
    }
    attack_prompt_ids = {
        int(r["sample_index"]) for r in rows if r["letter"] in ("B", "D")
    }
    return {
        "per_type": per_type,
        "prompt": prompt,
        "token_benign_fpr": round(float((pred & benign).sum() / max(1, benign.sum())), 5),
        "prompt_fpr": round(len(blocked_prompt & benign_prompt_ids) / max(1, len(benign_prompt_ids)), 5),
        "prompt_block_rate_any_suspicious": round(len(blocked_prompt & attack_prompt_ids) / max(1, len(attack_prompt_ids)), 5),
        "n_prompts": len(prompt_ids),
    }


def _run_offset(cfg: ExpConfig, offset: int) -> dict:
    x, records = _load_feature_bundle(cfg, offset)
    letters = np.array([r["letter"] for r in records])
    global_y = np.array([int(r["label"]) for r in records], dtype=int)
    is_train, is_test = holdout_mask(records, seed=cfg.seed)

    y_b = _branch_labels(letters, "B")
    y_d = _branch_labels(letters, "D")
    model_b = _fit_branch(x, y_b, is_train)
    model_d = _fit_branch(x, y_d, is_train)
    score_b = _score_branch(model_b, x)
    score_d = _score_branch(model_d, x)
    score = np.maximum(score_b, score_d)

    train_valid = is_train & (global_y == 0)
    if train_valid.sum() == 0:
        threshold = float("inf")
    else:
        threshold = modeling.threshold_for_fpr(score[is_train & modeling.valid_binary_mask(global_y)],
                                               global_y[is_train & modeling.valid_binary_mask(global_y)],
                                               cfg.fpr)
    pred = score >= threshold

    rows_test = [r for r, t in zip(records, is_test) if t]
    pred_test = pred[is_test]
    success = artifacts.success_set(cfg.out_dir, cfg.asr_judge)
    eval_report = _prompt_eval(rows_test, pred_test, success)

    report = {
        "stage": "10_two_branch_cascade",
        "pos_offset": offset,
        "fpr_target": cfg.fpr,
        "n_rows": len(records),
        "n_train": int(is_train.sum()),
        "n_test": int(is_test.sum()),
        "threshold": threshold,
        "branch_auc": {
            "B_branch_test_auc": _branch_auc(score_b, y_b, is_test),
            "D_branch_test_auc": _branch_auc(score_d, y_d, is_test),
            "combined_global_test_auc": _branch_auc(score, global_y, is_test),
        },
        "evaluation": eval_report,
        "notes": {
            "branch_B": "B is positive; C/E/F/G are negatives; D is excluded from branch fitting.",
            "branch_D": "D is positive; C/E/F/G are negatives; B is excluded from branch fitting.",
            "combined": "Deployment proxy uses max(B_score, D_score) with a threshold fit on train benign rows.",
        },
    }
    pos_dir = cfg.pos_dir(offset)
    io.write_json(pos_dir / "two_branch_cascade_report.json", report)
    io.write_text(pos_dir / "two_branch_cascade_report.md", _md(report))
    ev = report["evaluation"]
    print(f"[10] pos{offset}: combined AUC={report['branch_auc']['combined_global_test_auc']} "
          f"block={ev['prompt']['block_rate_prompt']} promptFPR={ev['prompt_fpr']}")
    return report


def _rate(ev: dict, letter: str):
    return ev["per_type"].get(letter, {}).get("rate")


def _md(r: dict) -> str:
    ev = r["evaluation"]
    lines = [
        f"# Stage 10 - Two-Branch Cascade (pos_offset={r['pos_offset']})",
        "",
        f"- train/test rows: {r['n_train']} / {r['n_test']}",
        f"- FPR target: {r['fpr_target']}",
        f"- combined threshold: {r['threshold']}",
        "",
        "## AUC",
        "| branch | test AUC |",
        "|---|---:|",
    ]
    for k, v in r["branch_auc"].items():
        lines.append(f"| {k} | {v} |")
    lines.extend([
        "",
        "## Held-out Rates",
        "| B block | D block | C FPR | E FPR | F FPR | G FPR | prompt FPR | ASR before | ASR after |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {_rate(ev,'B')} | {_rate(ev,'D')} | {_rate(ev,'C')} | {_rate(ev,'E')} | "
        f"{_rate(ev,'F')} | {_rate(ev,'G')} | {ev['prompt_fpr']} | "
        f"{ev['prompt']['asr_before']} | {ev['prompt']['asr_after']} |",
    ])
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

