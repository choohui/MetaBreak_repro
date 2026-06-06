"""Stage 00 (Main.md §1) — token-EMBEDDING-level special-vs-regular geometry.

Confirmatory check: do special tokens differ from regular tokens in the input
embedding table by L2 norm or cosine geometry? (The spec concluded there is no
clean separation — hence the need to look at *internal* representations.)

Outputs (under ``out_dir``):
    embedding_analysis.json   - norm/cosine/centroid stats + norm-separability AUC
    embedding_analysis.md     - human-readable summary
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from experiments_hc_4_claude.core import io

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # repro_mb (makes experiments_hc_4_claude importable)
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_4_claude.config import ExpConfig, config_from_args, get_model, make_stage_parser  # noqa: E402
from experiments_hc_4_claude.core import metrics  # noqa: E402


def _stats(x: np.ndarray) -> dict:
    return {
        "n": int(len(x)),
        "mean": round(float(np.mean(x)), 5),
        "std": round(float(np.std(x)), 5),
        "min": round(float(np.min(x)), 5),
        "max": round(float(np.max(x)), 5),
        "median": round(float(np.median(x)), 5),
    }


def _mean_pairwise_cosine(rows: np.ndarray, max_pairs: int = 2000, seed: int = 0) -> float:
    n = len(rows)
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(rows, axis=1) + 1e-12
    unit = rows / norms[:, None]
    pairs = min(max_pairs, n * (n - 1) // 2)
    acc = 0.0
    for _ in range(pairs):
        i, j = rng.integers(0, n), rng.integers(0, n)
        while j == i:
            j = rng.integers(0, n)
        acc += float(np.dot(unit[i], unit[j]))
    return round(acc / pairs, 5)


def run(cfg: ExpConfig, lm=None) -> dict:
    lm = get_model(cfg, lm)
    emb = lm.embedding.numpy() if hasattr(lm.embedding, "numpy") else np.asarray(lm.embedding)
    vocab = emb.shape[0]

    special_ids = sorted(i for i in lm.template.special_token_ids if 0 <= i < vocab)
    special_set = set(special_ids)
    rng = np.random.default_rng(0)
    n_reg = min(2000, vocab - len(special_ids))
    regular_ids = []
    while len(regular_ids) < n_reg:
        cand = int(rng.integers(0, vocab))
        if cand not in special_set:
            regular_ids.append(cand)
    regular_ids = np.array(regular_ids)

    spec = emb[np.array(special_ids)] if special_ids else np.zeros((0, emb.shape[1]))
    reg = emb[regular_ids]

    spec_norm = np.linalg.norm(spec, axis=1) if len(spec) else np.array([0.0])
    reg_norm = np.linalg.norm(reg, axis=1)

    # Funnel / outlier check: distance to the global embedding centroid.
    centroid = emb.mean(axis=0)
    spec_cdist = np.linalg.norm(spec - centroid, axis=1) if len(spec) else np.array([0.0])
    reg_cdist = np.linalg.norm(reg - centroid, axis=1)

    # Norm-based separability of special(1) vs regular(0).
    if len(spec):
        scores = np.concatenate([spec_norm, reg_norm])
        labels = np.concatenate([np.ones(len(spec_norm)), np.zeros(len(reg_norm))])
        norm_auc = metrics.binary_metrics(scores, labels)
        cdist_scores = np.concatenate([spec_cdist, reg_cdist])
        cdist_auc = metrics.binary_metrics(cdist_scores, labels)
    else:
        norm_auc = cdist_auc = {"auc": float("nan")}

    report = {
        "vocab_size": int(vocab),
        "dim": int(emb.shape[1]),
        "n_special": len(special_ids),
        "n_regular_sampled": int(len(regular_ids)),
        "special_token_ids": special_ids,
        "l2_norm": {"special": _stats(spec_norm), "regular": _stats(reg_norm)},
        "dist_to_global_centroid": {
            "special": _stats(spec_cdist), "regular": _stats(reg_cdist)},
        "mean_pairwise_cosine": {
            "within_special": _mean_pairwise_cosine(spec) if len(spec) > 1 else None,
            "within_regular": _mean_pairwise_cosine(reg),
            "special_vs_regular_centroid": round(
                metrics.cosine(spec.mean(axis=0), reg.mean(axis=0)), 5) if len(spec) else None,
        },
        "separability_auc": {
            "by_l2_norm": norm_auc.get("auc"),
            "by_dist_to_centroid": cdist_auc.get("auc"),
        },
        "note": "AUC near 0.5 => special and regular tokens are NOT separable at "
                "the embedding level; internal representations are required.",
    }
    io.write_json(cfg.out_dir / "embedding_analysis.json", report)
    io.write_text(cfg.out_dir / "embedding_analysis.md", _md(report))
    print(f"[00] embedding analysis -> {cfg.out_dir/'embedding_analysis.json'} "
          f"(norm-AUC={report['separability_auc']['by_l2_norm']})")
    return report


def _md(r: dict) -> str:
    L = ["# Stage 00 — Embedding-level special-vs-regular geometry (Main.md §1)", ""]
    L.append(f"- vocab={r['vocab_size']}, dim={r['dim']}, "
             f"n_special={r['n_special']}, n_regular_sampled={r['n_regular_sampled']}")
    L.append("")
    L.append("## L2 norm")
    L.append(f"- special: {r['l2_norm']['special']}")
    L.append(f"- regular: {r['l2_norm']['regular']}")
    L.append("## Distance to global centroid (funnel check)")
    L.append(f"- special: {r['dist_to_global_centroid']['special']}")
    L.append(f"- regular: {r['dist_to_global_centroid']['regular']}")
    L.append("## Separability AUC (special=1 vs regular=0)")
    L.append(f"- by L2 norm: {r['separability_auc']['by_l2_norm']}")
    L.append(f"- by distance to centroid: {r['separability_auc']['by_dist_to_centroid']}")
    L.append("")
    L.append(f"> {r['note']}")
    return "\n".join(L) + "\n"


def main() -> None:
    args = make_stage_parser(__doc__).parse_args()
    run(config_from_args(args))


if __name__ == "__main__":
    main()
