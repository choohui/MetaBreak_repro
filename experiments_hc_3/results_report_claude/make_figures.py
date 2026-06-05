"""Generate figures for the experiments_hc_3 Claude report.

Design rule for this report: figures carry NO explanatory text (no titles,
no annotations). All interpretation lives in report.md. Only minimal axis
ticks / short data identifiers (category letters, layer indices, feature
family names) appear, because they are part of the data, not prose.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc

mpl.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "hc3_active_sink"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Consistent colors
C_ATTACK = "#c0392b"
C_BENIGN = "#2471a3"
C_REF = "#7f8c8d"
LETTER_COLOR = {
    "A": C_REF, "B": C_ATTACK, "C": C_BENIGN, "D": "#e67e22",
    "E": "#16a085", "F": "#2980b9", "G": "#8e44ad",
}
FAMILY_COLOR = {
    "sink": "#c0392b", "sink_rank_pct": "#e74c3c",
    "value_norm": "#27ae60", "output_norm": "#f39c12",
    "active_value": "#8e44ad", "active_output": "#2980b9",
    "hidden_norm": "#16a085",
}


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jlload(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------- fig01 census
def fig01():
    s = jload(RES / "extract_summary.json")
    raw = s["raw_census"]; bal = s["census"]
    letters = ["A", "B", "C", "D", "E", "F", "G"]
    name = {l: [k for k in raw if k.startswith(l + "_")][0] for l in letters}
    rawv = [raw[name[l]] for l in letters]
    balv = [bal[name[l]] for l in letters]
    cols = [LETTER_COLOR[l] for l in letters]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(letters, rawv, color=cols)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("token count (log)")
    ax[1].bar(letters, balv, color=cols)
    ax[1].set_ylabel("token count")
    for a in ax:
        a.set_xlabel("token type")
    fig.tight_layout()
    fig.savefig(FIG / "fig01_dataset_census.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig02 ASR
def fig02():
    s = jload(RES / "asr_summary.json")
    pv = s["per_variant"]
    variants = ["malicious_special", "malicious_mimicry", "positioned_regular"]
    kw = [pv[v]["asr_refusal_keyword"] for v in variants]
    gd = [pv[v]["asr_llama_guard"] for v in variants]
    x = np.arange(len(variants)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.bar(x - w / 2, kw, w, color="#34495e", label="keyword")
    ax.bar(x + w / 2, gd, w, color="#e74c3c", label="Llama-Guard")
    ax.set_xticks(x); ax.set_xticklabels(["D", "B", "F"])
    ax.set_ylabel("ASR (%)")
    ax.set_xlabel("attack variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig02_asr_baseline.png")
    plt.close(fig)


# ----------------------------------------------------- fig03 per-letter scores
def fig03():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, off in zip(axes, [0, 1]):
        recs = jlload(RES / f"pos{off}" / "active_sinkprobe_scores.jsonl")
        letters = ["B", "C", "D", "E", "F", "G"]
        data = {l: [r["cv_score"] for r in recs
                    if r["letter"] == l and r["cv_score"] is not None] for l in letters}
        bp = ax.boxplot([data[l] for l in letters], positions=range(len(letters)),
                        widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black"))
        for patch, l in zip(bp["boxes"], letters):
            patch.set_facecolor(LETTER_COLOR[l]); patch.set_alpha(0.75)
        ax.set_xticks(range(len(letters))); ax.set_xticklabels(letters)
        ax.set_xlabel("token type")
    axes[0].set_ylabel("SinkProbe CV score")
    fig.tight_layout()
    fig.savefig(FIG / "fig03_perletter_scores.png")
    plt.close(fig)


# ------------------------------------------------- fig04 score distribution
def fig04():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, off in zip(axes, [0, 1]):
        recs = jlload(RES / f"pos{off}" / "active_sinkprobe_scores.jsonl")
        atk = [r["cv_score"] for r in recs if r["label"] == 1 and r["cv_score"] is not None]
        ben = [r["cv_score"] for r in recs if r["label"] == 0 and r["cv_score"] is not None]
        ax.hist(ben, bins=bins, color=C_BENIGN, alpha=0.6, label="benign (C,E,F,G)", density=True)
        ax.hist(atk, bins=bins, color=C_ATTACK, alpha=0.6, label="attack (B,D)", density=True)
        ax.set_xlabel("SinkProbe CV score")
    axes[0].set_ylabel("density")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig04_score_distribution.png")
    plt.close(fig)


# -------------------------------------------------------------- fig05 ROC
def fig05():
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for off, col in zip([0, 1], ["#c0392b", "#2980b9"]):
        recs = jlload(RES / f"pos{off}" / "active_sinkprobe_scores.jsonl")
        y = [r["label"] for r in recs if r["label"] in (0, 1) and r["cv_score"] is not None]
        s = [r["cv_score"] for r in recs if r["label"] in (0, 1) and r["cv_score"] is not None]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"pos{off} (AUC={sk_auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig05_token_roc.png")
    plt.close(fig)


# ------------------------------------------- fig06 layerwise single-feature AUC
def _auc_1d(score, y):
    fpr, tpr, _ = roc_curve(y, score)
    a = sk_auc(fpr, tpr)
    return max(a, 1 - a)  # direction-agnostic separability


def fig06():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    families = ["sink", "value_norm", "output_norm", "active_value",
                "active_output", "hidden_norm"]
    for ax, off in zip(axes, [0, 1]):
        npz = np.load(RES / f"pos{off}" / "active_sinkprobe_features.npz", allow_pickle=True)
        x = npz["x"]; y = npz["y"]; names = list(npz["feature_names"])
        m = (y == 0) | (y == 1)
        x, y = x[m], y[m]
        for fam in families:
            idx, layers = [], []
            for j, nm in enumerate(names):
                if nm.startswith(fam + "_L") and nm[len(fam) + 2:].isdigit():
                    layers.append(int(nm[len(fam) + 2:])); idx.append(j)
            if not idx:
                continue
            order = np.argsort(layers)
            layers = np.array(layers)[order]; idx = np.array(idx)[order]
            aucs = [_auc_1d(x[:, j], y) for j in idx]
            ax.plot(layers, aucs, marker="o", ms=3, lw=1.5,
                    color=FAMILY_COLOR[fam], label=fam)
        ax.set_xlabel("layer index")
        ax.set_ylim(0.45, 1.02)
    axes[0].set_ylabel("single-feature AUC")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig06_layerwise_auc.png")
    plt.close(fig)


# --------------------------------------------- fig07 top single-feature AUC bars
def fig07():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, off in zip(axes, [0, 1]):
        rep = jload(RES / f"pos{off}" / "active_sinkprobe_report.json")
        rows = rep["top_single_features"][:15][::-1]
        names = [r["feature"] for r in rows]
        aucs = [r["auc"] for r in rows]
        cols = [FAMILY_COLOR.get(n.rsplit("_L", 1)[0].split("_early")[0].split("_max")[0], "#555")
                for n in names]
        # map family by prefix
        def fam_of(n):
            for f in FAMILY_COLOR:
                if n.startswith(f):
                    return f
            return None
        cols = [FAMILY_COLOR.get(fam_of(n), "#555") for n in names]
        ax.barh(range(len(rows)), aucs, color=cols)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0.8, 0.96)
        ax.set_xlabel("single-feature AUC")
    fig.tight_layout()
    fig.savefig(FIG / "fig07_top_single_features.png")
    plt.close(fig)


# ------------------------------------------------ fig08 sparse coefficients
def fig08():
    rep = jload(RES / "pos0" / "active_sinkprobe_report.json")
    rows = [r for r in rep["top_coefficients"] if r["coef"] != 0][:22][::-1]
    names = [r["feature"] for r in rows]
    coef = [r["coef"] for r in rows]
    cols = ["#c0392b" if c > 0 else "#2980b9" for c in coef]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.barh(range(len(rows)), coef, color=cols)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("L1 logistic coefficient")
    fig.tight_layout()
    fig.savefig(FIG / "fig08_sparse_coefficients.png")
    plt.close(fig)


# ---------------------------------------------- fig09 prompt aggregation ASR
def fig09():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    # ASR before/after per strategy per pos
    labels = ["any-token\npos0", "logreg\npos0", "any-token\npos1", "logreg\npos1"]
    before, after = [], []
    for off in [0, 1]:
        rep = jload(RES / f"pos{off}" / "prompt_aggregation_report.json")
        for strat in ["token_any_prompt_block", "prompt_logreg"]:
            s = rep["strategies"][strat]
            before.append(s["asr_before"]); after.append(s["asr_after"])
    x = np.arange(len(labels)); w = 0.38
    axes[0].bar(x - w / 2, before, w, color="#7f8c8d", label="ASR before")
    axes[0].bar(x + w / 2, after, w, color="#c0392b", label="ASR after")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("ASR")
    axes[0].legend()
    # block rate vs prompt FPR
    br, fpr = [], []
    for off in [0, 1]:
        rep = jload(RES / f"pos{off}" / "prompt_aggregation_report.json")
        for strat in ["token_any_prompt_block", "prompt_logreg"]:
            s = rep["strategies"][strat]
            br.append(s["block_rate"]); fpr.append(s["prompt_fpr"])
    axes[1].bar(x - w / 2, br, w, color="#16a085", label="attack block rate")
    axes[1].bar(x + w / 2, fpr, w, color="#e67e22", label="prompt FPR")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig09_prompt_aggregation.png")
    plt.close(fig)


# ---------------------------------------------- fig10 two-branch per-type rate
def fig10():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    letters = ["A", "B", "C", "D", "E", "F", "G"]
    for ax, off in zip(axes, [0, 1]):
        rep = jload(RES / f"pos{off}" / "two_branch_cascade_report.json")
        pt = rep["evaluation"]["per_type"]
        rates = [pt[l]["rate"] for l in letters]
        ax.bar(letters, rates, color=[LETTER_COLOR[l] for l in letters])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("token type")
    axes[0].set_ylabel("held-out flag rate")
    fig.tight_layout()
    fig.savefig(FIG / "fig10_two_branch_per_type.png")
    plt.close(fig)


# ---------------------------------------------- fig11 counterfactual deltas
def fig11():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, off in zip(axes, [0, 1]):
        import csv
        rows = list(csv.DictReader(
            (RES / f"pos{off}" / "counterfactual_paired_deltas.csv").open(encoding="utf-8")))
        pairs = ["B_minus_F", "D_minus_F"]
        data = {p: [float(r["delta"]) for r in rows if r["pair"] == p] for p in pairs}
        bp = ax.boxplot([data[p] for p in pairs], positions=range(len(pairs)),
                        widths=0.55, patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=3, alpha=0.4),
                        medianprops=dict(color="black"))
        for patch, c in zip(bp["boxes"], ["#c0392b", "#e67e22"]):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_xticks(range(len(pairs))); ax.set_xticklabels(["B−F", "D−F"])
        ax.set_xlabel("counterfactual pair")
    axes[0].set_ylabel("paired score delta")
    fig.tight_layout()
    fig.savefig(FIG / "fig11_counterfactual_deltas.png")
    plt.close(fig)


# ---------------------------------------------- fig12 counterfactual scatter
def fig12():
    import csv
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, off in zip(axes, [0, 1]):
        rows = list(csv.DictReader(
            (RES / f"pos{off}" / "counterfactual_paired_deltas.csv").open(encoding="utf-8")))
        for p, c in [("B_minus_F", "#c0392b"), ("D_minus_F", "#e67e22")]:
            xs = [float(r["control_score"]) for r in rows if r["pair"] == p]
            ys = [float(r["attack_score"]) for r in rows if r["pair"] == p]
            ax.scatter(xs, ys, s=14, color=c, alpha=0.55, label=p.replace("_minus_", "−"))
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_xlabel("control token score")
    axes[0].set_ylabel("attack token score")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig12_counterfactual_scatter.png")
    plt.close(fig)


if __name__ == "__main__":
    for fn in [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08,
               fig09, fig10, fig11, fig12]:
        fn()
        print("ok", fn.__name__)
    print("figures ->", FIG)
