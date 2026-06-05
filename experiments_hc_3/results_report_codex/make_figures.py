"""Generate compact figures for the hc_3 report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "hc3_active_sink"
OUT = ROOT / "results_report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

LETTERS = list("ABCDEFG")
ATTACK = {"B", "D"}
COLORS = {
    "attack": "#d95f02",
    "benign": "#1b9e77",
    "ref": "#7570b3",
    "pos0": "#4c78a8",
    "pos1": "#f58518",
    "asr": "#e45756",
    "fpr": "#72b7b2",
}


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_float(x):
    return np.nan if x is None else float(x)


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close()


def fig01_data_asr():
    ex = read_json(RESULTS / "extract_summary.json")
    raw = ex["raw_census"]
    bal = ex["census"]
    asr = read_json(RESULTS / "asr_summary.json")["per_variant"]
    cats = [
        "A_system_special", "B_malicious_mimicry", "C_benign_mimicry",
        "D_malicious_special", "E_benign_special", "F_positioned_regular",
        "G_ordinary_regular",
    ]
    x = np.arange(len(cats))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(x - 0.18, [raw.get(c, 0) for c in cats], width=0.36, color="#8da0cb")
    axes[0].bar(x + 0.18, [bal.get(c, 0) for c in cats], width=0.36, color="#66c2a5")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(LETTERS)
    axes[0].set_ylabel("rows")
    axes[0].set_title("Token Rows")
    axes[0].legend(["raw", "balanced"], frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    names = ["malicious_mimicry", "malicious_special", "positioned_regular"]
    x2 = np.arange(len(names))
    kw = [asr[n]["asr_refusal_keyword"] for n in names]
    gd = [asr[n]["asr_llama_guard"] for n in names]
    axes[1].bar(x2 - 0.18, kw, width=0.36, color="#fc8d62")
    axes[1].bar(x2 + 0.18, gd, width=0.36, color="#8da0cb")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["B", "D", "F"])
    axes[1].set_ylabel("ASR (%)")
    axes[1].set_title("Attack Success")
    axes[1].legend(["keyword", "guard"], frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    savefig("fig01_data_asr.png")


def fig02_score_distributions():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    score_letters = list("BCDEFG")
    for ax, pos in zip(axes, [0, 1]):
        rows = read_jsonl(RESULTS / f"pos{pos}" / "active_sinkprobe_scores.jsonl")
        data = []
        for L in score_letters:
            vals = [float(r["cv_score"]) for r in rows
                    if r.get("letter") == L and r.get("cv_score") is not None]
            data.append(vals)
        vp = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(COLORS["attack"] if score_letters[i] in ATTACK else COLORS["benign"])
            body.set_edgecolor("black")
            body.set_alpha(0.75)
        vp["cmedians"].set_color("black")
        ax.set_xticks(np.arange(1, len(score_letters) + 1))
        ax.set_xticklabels(score_letters)
        ax.set_ylim(-0.04, 1.04)
        ax.set_title(f"pos{pos}")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Active SinkProbe score")
    savefig("fig02_active_sinkprobe_scores.png")


def oriented_auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    m = np.isfinite(scores) & ((labels == 0) | (labels == 1))
    scores, labels = scores[m], labels[m]
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ss = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return max(float(auc), 1.0 - float(auc))


def fig03_feature_auc_heatmap():
    families = ["sink", "sink_rank_pct", "value_norm", "output_norm",
                "active_value", "active_output", "hidden_norm"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.8), sharex=True)
    for ax, pos in zip(axes, [0, 1]):
        npz = np.load(RESULTS / f"pos{pos}" / "active_sinkprobe_features.npz", allow_pickle=True)
        x, y = npz["x"], npz["y"]
        names = [str(n) for n in npz["feature_names"]]
        layers = list(range(33))
        mat = np.full((len(families), len(layers)), np.nan)
        for fi, fam in enumerate(families):
            for L in layers:
                name = f"{fam}_L{L}"
                if name in names:
                    mat[fi, L] = oriented_auc(x[:, names.index(name)], y)
        im = ax.imshow(mat, aspect="auto", vmin=0.5, vmax=1.0, cmap="viridis")
        ax.set_yticks(np.arange(len(families)))
        ax.set_yticklabels(families)
        ax.set_title(f"pos{pos}")
        ax.set_ylabel("feature family")
    axes[-1].set_xlabel("layer")
    axes[-1].set_xticks(np.arange(0, 33, 2))
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86)
    cbar.set_label("oriented AUC")
    plt.savefig(OUT / "fig03_feature_auc_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close()


def fig04_prompt_aggregation():
    rows = []
    for pos in [0, 1]:
        r = read_json(RESULTS / f"pos{pos}" / "prompt_aggregation_report.json")
        for name, s in r["strategies"].items():
            rows.append((pos, name, s))
    metrics = ["block_rate", "prompt_fpr", "asr_after", "block_rate_among_successful"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.8), sharey=True)
    for ax, met in zip(axes, metrics):
        labels = []
        vals = []
        colors = []
        for pos, name, s in rows:
            labels.append(f"p{pos}\n{name.replace('_', ' ')}")
            vals.append(safe_float(s[met]))
            colors.append(COLORS["pos0"] if pos == 0 else COLORS["pos1"])
        ax.bar(np.arange(len(vals)), vals, color=colors)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(met)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    savefig("fig04_prompt_aggregation.png")


def fig05_two_branch_rates():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, pos in zip(axes, [0, 1]):
        r = read_json(RESULTS / f"pos{pos}" / "two_branch_cascade_report.json")
        rates = [r["evaluation"]["per_type"].get(L, {}).get("rate", np.nan) for L in LETTERS]
        colors = [COLORS["ref"] if L == "A" else COLORS["attack"] if L in ATTACK else COLORS["benign"]
                  for L in LETTERS]
        ax.bar(np.arange(len(LETTERS)), rates, color=colors)
        ax.set_xticks(np.arange(len(LETTERS)))
        ax.set_xticklabels(LETTERS)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"pos{pos}")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("flagged rate")
    savefig("fig05_two_branch_per_type.png")


def fig06_counterfactual_deltas():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, pos in zip(axes, [0, 1]):
        path = RESULTS / f"pos{pos}" / "counterfactual_paired_deltas.csv"
        by_pair = {}
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                by_pair.setdefault(row["pair"], []).append(float(row["delta"]))
        pairs = sorted(by_pair)
        data = [by_pair[p] for p in pairs]
        vp = ax.violinplot(data, showmedians=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor("#a6d854")
            body.set_edgecolor("black")
            body.set_alpha(0.8)
        vp["cmedians"].set_color("black")
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(np.arange(1, len(pairs) + 1))
        ax.set_xticklabels([p.replace("_minus_", "-") for p in pairs])
        ax.set_title(f"pos{pos}")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("attack score - control score")
    savefig("fig06_counterfactual_deltas.png")


def fig07_coeff_family_layer():
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.8), sharex=True)
    families = ["sink", "sink_rank_pct", "value_norm", "output_norm",
                "active_value", "active_output", "hidden_norm"]
    for ax, pos in zip(axes, [0, 1]):
        report = read_json(RESULTS / f"pos{pos}" / "active_sinkprobe_report.json")
        mat = np.zeros((len(families), 33), dtype=float)
        for row in report["top_coefficients"]:
            feat = row["feature"]
            if "_L" not in feat:
                continue
            fam, layer = feat.rsplit("_L", 1)
            if fam in families and layer.isdigit():
                L = int(layer)
                if L < 33:
                    mat[families.index(fam), L] += float(row["coef"])
        vmax = max(0.1, np.nanmax(np.abs(mat)))
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_yticks(np.arange(len(families)))
        ax.set_yticklabels(families)
        ax.set_title(f"pos{pos}")
        ax.set_ylabel("feature family")
    axes[-1].set_xlabel("layer")
    axes[-1].set_xticks(np.arange(0, 33, 2))
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86)
    cbar.set_label("sparse coefficient")
    plt.savefig(OUT / "fig07_sparse_coefficients.png", dpi=220, bbox_inches="tight")
    plt.close()


def main():
    fig01_data_asr()
    fig02_score_distributions()
    fig03_feature_auc_heatmap()
    fig04_prompt_aggregation()
    fig05_two_branch_rates()
    fig06_counterfactual_deltas()
    fig07_coeff_family_layer()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
