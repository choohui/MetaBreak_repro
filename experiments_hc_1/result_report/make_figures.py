"""Generate report figures for experiments_hc_1 (Main.md §1-§3).

Reads the result JSON/CSV produced by stages 00/02/04/05/06 and writes PNGs into
``result_report/figures/``. Pure-stdlib + numpy + matplotlib; no model needed.

Run:  python experiments_hc_1/result_report/make_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "hc1_llama31_8b"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Consistent per-type colors
COL = {
    "A": "#7f7f7f", "B": "#d62728", "C": "#8c564b", "D": "#ff7f0e",
    "E": "#1f77b4", "F": "#2ca02c", "G": "#9467bd",
}
SIGS = ["hidden_norm", "sink", "value_norm", "output_norm", "cos_to_ref"]


def load(p):
    return json.load(open(RES / p, encoding="utf-8"))


def load_opt(p):
    """Load a result JSON, or return None if it does not exist yet."""
    path = RES / p
    if not path.exists():
        return None
    return json.load(open(path, encoding="utf-8"))


def _pertype_from_report(report):
    """report['per_type'] (list of {signal,letter,flagged_rate}) -> {signal: {letter: rate}}."""
    data = {s: {} for s in SIGS}
    for r in report.get("per_type", []):
        data.setdefault(r["signal"], {})[r["letter"]] = float(r["flagged_rate"])
    return data


def save(fig, name):
    fig.tight_layout()
    out = FIG / name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(HERE.parent))


# ---------------------------------------------------------------- Fig 1: embedding §1
def fig_embedding():
    em = load("embedding_analysis.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in [
        (axes[0], "l2_norm", "Token-embedding L2 norm"),
        (axes[1], "dist_to_global_centroid", "Distance to global centroid"),
    ]:
        sp = em[key]["special"]
        rg = em[key]["regular"]
        ax.bar(["special\n(256, incl. unused)", "regular\n(2000 sampled)"],
               [sp["mean"], rg["mean"]], yerr=[sp["std"], rg["std"]],
               color=["#d62728", "#9467bd"], capsize=5, alpha=.85)
        ax.scatter([0, 0], [sp["median"], sp["max"]], color="k", zorder=5, s=14)
        ax.annotate(f"median={sp['median']:.2f}", (0, sp["median"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=7)
        ax.set_title(title)
        ax.set_ylabel("value")
    auc = em["separability_auc"]["by_l2_norm"]
    fig.suptitle("Fig 1. Embedding-level geometry: the 256 'special' slots (IDs 128000-128255) are\n"
                 f"dominated by untrained reserved tokens with ~0 norm (median 0.0); the AUC={auc:.3f} is an artifact",
                 fontsize=9)
    save(fig, "fig01_embedding_norms.png")


# ---------------------------------------------------------------- Fig 2: layerwise norms
def fig_layer_norms():
    pos = load("pos0/representation_metrics.json")
    layers = [r["layer"] for r in pos["per_layer"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for t in ["A", "B", "D", "E", "F", "G"]:
        key = f"{t}__mean_norm"
        ys = [r.get(key) for r in pos["per_layer"]]
        ax.plot(layers, ys, marker="o", ms=3, color=COL[t], label=f"{t}")
    ax.set_yscale("log")
    ax.set_xlabel("hidden layer index")
    ax.set_ylabel("mean hidden-state L2 norm (log)")
    ax.set_title("Fig 2. Per-layer mean hidden norm by token type (pos_offset=0)\n"
                 "A/E (special) carry a huge attention-sink norm; B/D/F/G (regular) stay small")
    ax.legend(title="type", ncol=6, fontsize=8)
    ax.grid(alpha=.3, which="both")
    save(fig, "fig02_layerwise_norms.png")


# ---------------------------------------------------------------- Fig 3: cosine pairs
def fig_cosine():
    pos = load("pos0/representation_metrics.json")
    layers = [r["layer"] for r in pos["per_layer"]]
    pairs = [("cos__A_B", "cos(A,B)"), ("cos__A_D", "cos(A,D)"), ("cos__A_G", "cos(A,G)"),
             ("cos__B_D", "cos(B,D)"), ("cos__B_F", "cos(B,F)")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, lab in pairs:
        ys = [r.get(key) for r in pos["per_layer"]]
        style = "-" if key in ("cos__B_D", "cos__B_F") else "--"
        ax.plot(layers, ys, style, marker=".", label=lab)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("hidden layer index")
    ax.set_ylabel("centroid-to-centroid cosine")
    ax.set_title("Fig 3. Layerwise cosine between type centroids (pos_offset=0)\n"
                 "Attacks do NOT align to the template-special centroid A (cos~0);\n"
                 "the two attack families B and D converge internally (cos up to ~0.77)")
    ax.legend(fontsize=8, ncol=5)
    ax.grid(alpha=.3)
    save(fig, "fig03_cosine_pairs.png")


# ---------------------------------------------------------------- Fig 4: probe AUC
def fig_probe():
    fig, ax = plt.subplots(figsize=(8, 4))
    for pos, c in [("pos0", "#d62728"), ("pos1", "#1f77b4")]:
        m = load(f"{pos}/representation_metrics.json")
        layers = [r["layer"] for r in m["per_layer"]]
        ys = [r.get("probe_auc") for r in m["per_layer"]]
        ax.plot(layers, ys, marker="o", ms=3, color=c, label=f"{pos}")
    ax.set_ylim(0.95, 1.005)
    ax.set_xlabel("hidden layer index")
    ax.set_ylabel("logreg probe ROC-AUC (5-fold CV)")
    ax.set_title("Fig 4. Full-hidden-vector probe: attack(B,D) vs benign(E,F,G)\n"
                 "near-perfect linear separation at essentially every layer")
    ax.legend()
    ax.grid(alpha=.3)
    save(fig, "fig04_probe_auc.png")


# ---------------------------------------------------------------- Fig 5: signal AUC bars
def fig_signal_auc():
    td = load("pos0/threshold_defense.json")
    ta = load("pos0/threshold_asr.json")
    per_type = {b["signal"]: b["best_auc"] for b in td["best_per_signal"]}
    per_asr = {b["signal"]: b["best_auc"] for b in ta["best_per_signal"]}
    x = np.arange(len(SIGS))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - .2, [per_type[s] for s in SIGS], .4, label="per-type (B,D vs E,F,G)", color="#1f77b4")
    ax.bar(x + .2, [per_asr[s] for s in SIGS], .4, label="ASR-based (succeeded atk vs benign)", color="#ff7f0e")
    ax.axhline(0.5, color="k", lw=.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(SIGS, rotation=15)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("best-layer ROC-AUC")
    ax.set_title("Fig 5. Single-signal threshold detectability (pos_offset=0)")
    for i, s in enumerate(SIGS):
        ax.text(i - .2, per_type[s] + .005, f"{per_type[s]:.2f}", ha="center", fontsize=7)
        ax.text(i + .2, per_asr[s] + .005, f"{per_asr[s]:.2f}", ha="center", fontsize=7)
    ax.legend(fontsize=8)
    save(fig, "fig05_signal_auc.png")


# ---------------------------------------------------------------- Fig 6: per-type flagged
def fig_pertype():
    rows = list(csv.DictReader(open(RES / "pos0" / "threshold_per_type.csv", encoding="utf-8")))
    present = {r["letter"] for r in rows}
    types = [t for t in ["A", "B", "C", "D", "E", "F", "G"] if t in present]
    data = {s: {} for s in SIGS}
    for r in rows:
        data[r["signal"]][r["letter"]] = float(r["flagged_rate"])
    x = np.arange(len(SIGS))
    w = 0.8 / max(1, len(types))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for j, t in enumerate(types):
        ax.bar(x + (j - (len(types) - 1) / 2) * w, [data[s].get(t, 0) for s in SIGS], w,
               color=COL[t], label=f"{t}")
    ax.set_xticks(x)
    ax.set_xticklabels(SIGS, rotation=15)
    ax.set_ylabel("flagged rate at Youden threshold")
    ax.set_title("Fig 6. Per-type flagged rate at each signal's Youden threshold (pos_offset=0)\n"
                 "positives B,D (red/orange) should be high; negatives E,F,G low.\n"
                 "Note: value_norm/cos_to_ref also flag E (benign special) -> they are 'special detectors'")
    ax.legend(title="type", ncol=6, fontsize=8)
    ax.grid(alpha=.3, axis="y")
    save(fig, "fig06_pertype_flagged.png")


# ---------------------------------------------------------------- Fig 7: sink-range reduction
def fig_sinkrange():
    full = load("pos0/threshold_defense.json")
    red = load("pos0/sink_range_report.json")
    pct = load_opt("pos0/sink_range_report__sink_pct.json")  # optional 3rd group

    def aucs(rep):
        return {b["signal"]: b["best_auc"] for b in rep["best_per_signal"]}

    series = [
        ("full token set", f"full (n={full.get('n_rows', '?')})", "#9467bd", aucs(full)),
        ("header-slots",
         f"header-slots (n={red['n_reduced']}, {red['reduction_ratio']:.0%})",
         "#2ca02c", aucs(red)),
    ]
    if pct is not None:
        series.append((
            "sink top-%",
            f"sink top-{pct.get('keep_pct', '?'):g}% (n={pct['n_reduced']}, {pct['reduction_ratio']:.0%})",
            "#1f77b4", aucs(pct)))

    x = np.arange(len(SIGS))
    n = len(series)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for j, (_name, lab, col, auc) in enumerate(series):
        off = (j - (n - 1) / 2) * w
        ax.bar(x + off, [auc.get(s, np.nan) for s in SIGS], w, label=lab, color=col)
        for i, s in enumerate(SIGS):
            v = auc.get(s)
            if v is not None:
                ax.text(i + off, v + .004, f"{v:.3f}", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(SIGS, rotation=15)
    ax.set_ylim(0.5, 1.03)
    ax.set_ylabel("best-layer ROC-AUC")
    ax.set_title("Fig 7. §3 token-range reduction sharpens the threshold signal (pos_offset=0)\n"
                 "header-slots lifts value_norm to ~1.0; sink top-% is a sink-only 1st-stage gate")
    ax.legend(fontsize=8)
    save(fig, "fig07_sinkrange.png")


# -------------------------------------------------- Fig 9: per-type flagged rate, reduced sets
def fig_sinkrange_pertype():
    """Fig 6-style per-type flagged rate, but on the reduced sets (header-slots and
    the sink top-% gate), so the flagged composition behind Fig 7's AUCs is visible."""
    panels = [("header-slots", load_opt("pos0/sink_range_report.json"))]
    pct = load_opt("pos0/sink_range_report__sink_pct.json")
    if pct is not None:
        panels.append((f"sink top-{pct.get('keep_pct', '?'):g}%", pct))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.6), squeeze=False)
    x = np.arange(len(SIGS))
    for ax, (name, rep) in zip(axes[0], panels):
        if rep is None:
            ax.set_visible(False)
            continue
        data = _pertype_from_report(rep)
        present = {lt for s in data for lt in data[s]}
        types = [t for t in ["A", "B", "C", "D", "E", "F", "G"] if t in present]
        w = 0.8 / max(1, len(types))
        for j, t in enumerate(types):
            ax.bar(x + (j - (len(types) - 1) / 2) * w,
                   [data.get(s, {}).get(t, 0) for s in SIGS], w, color=COL[t], label=t)
        ax.set_xticks(x)
        ax.set_xticklabels(SIGS, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("flagged rate at Youden threshold")
        ax.set_title(f"{name}  (n={rep['n_reduced']}, {rep['reduction_ratio']:.0%})")
        ax.legend(title="type", ncol=7, fontsize=7)
        ax.grid(alpha=.3, axis="y")
    fig.suptitle("Fig 9. Per-type flagged rate on the reduced sets (pos_offset=0)\n"
                 "positives B,D high; benign C,E,F,G low. Reveals which negatives survive each gate.",
                 fontsize=9)
    save(fig, "fig09_sinkrange_pertype.png")


# ---------------------------------------------------------------- Fig 8: ASR
def fig_asr():
    a = load("asr_summary.json")["per_variant"]
    order = [("malicious_special", "D malicious_special"),
             ("malicious_mimicry", "B malicious_mimicry"),
             ("positioned_regular", "F positioned_regular\n(benign control)")]
    labels = [lab for _, lab in order]
    vals = [a[k]["asr_refusal_keyword"] for k, _ in order]
    cols = [COL["D"], COL["B"], COL["F"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=cols, alpha=.85)
    ax.set_ylabel("ASR (%) — refusal-keyword heuristic")
    ax.set_ylim(0, 70)
    ax.set_title("Fig 8. Attack success rate by variant (n=150 each, Llama-3.1-8B-Instruct)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center")
    save(fig, "fig08_asr.png")


if __name__ == "__main__":
    fig_embedding()
    fig_layer_norms()
    fig_cosine()
    fig_probe()
    fig_signal_auc()
    fig_pertype()
    fig_sinkrange()
    fig_sinkrange_pertype()
    fig_asr()
    print("done.")
