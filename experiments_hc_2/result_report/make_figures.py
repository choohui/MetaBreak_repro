"""Generate report figures for experiments_hc_2 (Main.md §1-§4).

Reads the result JSON/CSV produced by stages 00/02/04/05/06/07 and writes PNGs
into ``result_report/figures/``. Pure stdlib + numpy + matplotlib; no model.
Every figure is defensive: a missing input is skipped, not fatal, so this runs
after a partial pipeline too.

Run (after a real model run):  python -m experiments_hc_2.result_report.make_figures
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
RES = HERE.parent / "results" / "hc2_llama31_8b"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COL = {
    "A": "#7f7f7f", "B": "#d62728", "C": "#8c564b", "D": "#ff7f0e",
    "E": "#1f77b4", "F": "#2ca02c", "G": "#9467bd",
}
SIGS = ["hidden_norm", "sink", "value_norm", "output_norm", "cos_to_ref"]


def load_opt(p):
    path = RES / p
    if not path.exists():
        print(f"  skip (missing): {p}")
        return None
    return json.load(open(path, encoding="utf-8"))


def save(fig, name):
    fig.tight_layout()
    out = FIG / name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(HERE.parent))


def _aucs(best_per_signal):
    return {b["signal"]: b["best_auc"] for b in best_per_signal}


# ---------------------------------------------------------------- Fig 1: embedding §1
def fig_embedding():
    em = load_opt("embedding_analysis.json")
    if em is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in [
        (axes[0], "l2_norm", "Token-embedding L2 norm"),
        (axes[1], "dist_to_global_centroid", "Distance to global centroid"),
    ]:
        sp, rg = em[key]["special"], em[key]["regular"]
        ax.bar(["special", "regular"], [sp["mean"], rg["mean"]],
               yerr=[sp["std"], rg["std"]], color=["#d62728", "#9467bd"],
               capsize=5, alpha=.85)
        ax.set_title(title)
        ax.set_ylabel("value")
    auc = em["separability_auc"]["by_l2_norm"]
    fig.suptitle(f"Fig 1 (§1). Embedding geometry — special-vs-regular AUC={auc:.3f} is a "
                 "reserved-token artifact; use internal representations instead", fontsize=9)
    save(fig, "fig01_embedding_norms.png")


# ---------------------------------------------------------------- Fig 2: layerwise norms
def fig_layer_norms():
    pos = load_opt("pos0/representation_metrics.json")
    if pos is None:
        return
    layers = [r["layer"] for r in pos["per_layer"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for t in ["A", "B", "C", "D", "E", "F", "G"]:
        ys = [r.get(f"{t}__mean_norm") for r in pos["per_layer"]]
        if any(v is not None for v in ys):
            ax.plot(layers, ys, marker="o", ms=3, color=COL[t], label=t)
    ax.set_yscale("log")
    ax.set_xlabel("hidden layer index")
    ax.set_ylabel("mean hidden-state L2 norm (log)")
    ax.set_title("Fig 2 (§2.2). Per-layer mean hidden norm by token type (pos0)")
    ax.legend(title="type", ncol=7, fontsize=8)
    ax.grid(alpha=.3, which="both")
    save(fig, "fig02_layerwise_norms.png")


# ---------------------------------------------------------------- Fig 3: cosine pairs
def fig_cosine():
    pos = load_opt("pos0/representation_metrics.json")
    if pos is None:
        return
    layers = [r["layer"] for r in pos["per_layer"]]
    pairs = [("cos__A_B", "cos(A,B)"), ("cos__A_D", "cos(A,D)"), ("cos__A_G", "cos(A,G)"),
             ("cos__B_C", "cos(B,C)"), ("cos__B_D", "cos(B,D)"), ("cos__B_F", "cos(B,F)")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, lab in pairs:
        ys = [r.get(key) for r in pos["per_layer"]]
        if any(v is not None for v in ys):
            style = "-" if key in ("cos__B_D", "cos__B_C") else "--"
            ax.plot(layers, ys, style, marker=".", label=lab)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("hidden layer index")
    ax.set_ylabel("centroid-to-centroid cosine")
    ax.set_title("Fig 3 (§2.3). Layerwise cosine between type centroids (pos0)\n"
                 "now incl. cos(B,C): same mimicry token in benign context")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=.3)
    save(fig, "fig03_cosine_pairs.png")


# ---------------------------------------------------------------- Fig 4: probe AUC naive vs grouped
def fig_probe():
    m = load_opt("pos0/representation_metrics.json")
    if m is None:
        return
    layers = [r["layer"] for r in m["per_layer"]]
    naive = [r.get("probe_auc") for r in m["per_layer"]]
    grouped = [r.get("probe_auc_grouped") for r in m["per_layer"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, naive, marker="o", ms=3, color="#d62728",
            label="naive per-token CV (may leak)")
    ax.plot(layers, grouped, marker="s", ms=3, color="#1f77b4",
            label="prompt-level GroupKFold (honest)")
    ax.set_xlabel("hidden layer index")
    ax.set_ylabel("logreg probe ROC-AUC")
    ax.set_title("Fig 4 (§2.3, limitation #3). Full-hidden probe — leakage check\n"
                 "gap between naive and grouped shows how much per-token CV inflated AUC")
    ax.legend()
    ax.grid(alpha=.3)
    save(fig, "fig04_probe_auc.png")


# ---------------------------------------------------------------- Fig 5: signal AUC bars
def fig_signal_auc():
    td = load_opt("pos0/threshold_defense.json")
    ta = load_opt("pos0/threshold_asr.json")
    if td is None or ta is None:
        return
    per_type = _aucs(td["best_per_signal"])
    per_asr = _aucs(ta["best_per_signal"])
    x = np.arange(len(SIGS))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - .2, [per_type.get(s) for s in SIGS], .4, label="per-type (B,D vs C,E,F,G)",
           color="#1f77b4")
    ax.bar(x + .2, [per_asr.get(s) for s in SIGS], .4, label="ASR-based", color="#ff7f0e")
    ax.axhline(0.5, color="k", lw=.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(SIGS, rotation=15)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("best-layer ROC-AUC")
    ax.set_title("Fig 5 (§2.3). Single-signal threshold detectability (pos0)")
    ax.legend(fontsize=8)
    save(fig, "fig05_signal_auc.png")


# ---------------------------------------------------------------- Fig 6: per-type flagged
def fig_pertype():
    path = RES / "pos0" / "threshold_per_type.csv"
    if not path.exists():
        print("  skip (missing): pos0/threshold_per_type.csv")
        return
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    present = {r["letter"] for r in rows}
    types = [t for t in "ABCDEFG" if t in present]
    data = {s: {} for s in SIGS}
    for r in rows:
        data[r["signal"]][r["letter"]] = float(r["flagged_rate"])
    x = np.arange(len(SIGS))
    w = 0.8 / max(1, len(types))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for j, t in enumerate(types):
        ax.bar(x + (j - (len(types) - 1) / 2) * w, [data[s].get(t, 0) for s in SIGS], w,
               color=COL[t], label=t)
    ax.set_xticks(x)
    ax.set_xticklabels(SIGS, rotation=15)
    ax.set_ylabel("flagged rate at Youden threshold")
    ax.set_title("Fig 6 (§2.3). Per-type flagged rate at each signal's Youden threshold (pos0)\n"
                 "value_norm/cos_to_ref also flag E (benign special) -> 'special detector' trap")
    ax.legend(title="type", ncol=7, fontsize=8)
    ax.grid(alpha=.3, axis="y")
    save(fig, "fig06_pertype_flagged.png")


# ---------------------------------------------------------------- Fig 7: §3 sink-filter sweep
def fig_sink_filter():
    sf = load_opt("pos0/sink_filter_report.json")
    if sf is None:
        return
    base = _aucs(sf["baseline_no_gate"]["best_per_signal"])
    pcts = [s["keep_pct"] for s in sf["sweep"]]
    fig, (axA, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    # left: AUROC vs keep_pct per signal
    for sig in SIGS:
        ys = [_aucs(s["metrics"]["best_per_signal"]).get(sig) for s in sf["sweep"]]
        axA.plot(pcts, ys, marker="o", ms=3, label=sig)
        axA.axhline(base.get(sig), ls=":", lw=.6, alpha=.4)
    axA.invert_xaxis()
    axA.set_xlabel("sink gate keep-% (← more aggressive)")
    axA.set_ylabel("best-layer ROC-AUC on survivors")
    axA.set_title("Fig 7a (§3). Sink filtering vs AUROC (pos0)")
    axA.legend(fontsize=7)
    axA.grid(alpha=.3)
    # right: attack recall vs keep_pct
    for L, c in [("B", COL["B"]), ("D", COL["D"])]:
        ys = [s["bd_recall"].get(L) for s in sf["sweep"]]
        axR.plot(pcts, ys, marker="s", ms=3, color=c, label=f"{L} recall")
    axR.invert_xaxis()
    axR.set_ylim(0, 1.05)
    axR.set_xlabel("sink gate keep-%")
    axR.set_ylabel("attack-token recall through gate")
    axR.set_title("Fig 7b (§3). Gate keeps attacks while dropping the body")
    axR.legend(fontsize=8)
    axR.grid(alpha=.3)
    rec = (sf.get("recommended") or {}).get("keep_pct")
    if rec is not None:
        for ax in (axA, axR):
            ax.axvline(rec, color="k", lw=.8, ls="--", alpha=.6)
    save(fig, "fig07_sink_filter.png")


# ---------------------------------------------------------------- Fig 8: ASR
def fig_asr():
    a = load_opt("asr_summary.json")
    if a is None:
        return
    pv = a["per_variant"]
    order = [("malicious_special", "D malicious_special", COL["D"]),
             ("malicious_mimicry", "B malicious_mimicry", COL["B"]),
             ("positioned_regular", "F positioned_regular\n(benign control)", COL["F"])]
    order = [(k, lab, c) for k, lab, c in order if k in pv]
    labels = [lab for _, lab, _ in order]
    kw = [pv[k]["asr_refusal_keyword"] for k, _, _ in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - .2, kw, .4, color=[c for _, _, c in order], alpha=.85, label="refusal-keyword")
    if any(pv[k].get("asr_llama_guard") is not None for k, _, _ in order):
        grd = [pv[k].get("asr_llama_guard") or 0 for k, _, _ in order]
        ax.bar(x + .2, grd, .4, color="#555555", alpha=.7, label="Llama-Guard")
        ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ASR (%)")
    ax.set_title("Fig 8 (§2.1). Attack success rate by variant\n"
                 "F (benign control) > 0 under keyword = heuristic false positive")
    save(fig, "fig08_asr.png")


# ---------------------------------------------------------------- Fig 9: §4 cascade
def fig_cascade():
    cr = load_opt("pos0/cascade_report.json")
    if cr is None:
        return
    strat = cr["strategies"]
    keys = ["one_stage_threshold", "gate_only", "cascade"]
    metrics = [("B", "B block"), ("D", "D block"), ("C", "C FPR"), ("E", "E FPR")]
    x = np.arange(len(metrics))
    w = 0.25
    fig, (axM, axA) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw={"width_ratios": [3, 1.4]})
    for j, k in enumerate(keys):
        pt = strat[k]["per_type"]
        ys = [(pt.get(L) or {}).get("rate") or 0 for L, _ in metrics]
        axM.bar(x + (j - 1) * w, ys, w, label=k)
    axM.set_xticks(x)
    axM.set_xticklabels([lab for _, lab in metrics])
    axM.set_ylim(0, 1.05)
    axM.set_ylabel("rate")
    axM.set_title(f"Fig 9a (§4). Cascade vs baselines — signal={cr['signal']}@L{cr['layer']}, "
                  f"keep={cr['keep_pct']}%, FPR target={cr['operating_fpr']}")
    axM.legend(fontsize=8)
    axM.grid(alpha=.3, axis="y")
    # ASR before/after
    before = strat["cascade"]["prompt"]["asr_before"]
    afters = [strat[k]["prompt"]["asr_after"] for k in keys]
    axA.bar(["before"] + keys, [before] + afters,
            color=["#999999", "#1f77b4", "#2ca02c", "#d62728"], alpha=.85)
    axA.set_ylabel("ASR (fraction)")
    axA.set_title("Fig 9b. ASR before vs after")
    axA.tick_params(axis="x", labelrotation=30, labelsize=7)
    save(fig, "fig09_cascade.png")


if __name__ == "__main__":
    fig_embedding()
    fig_layer_norms()
    fig_cosine()
    fig_probe()
    fig_signal_auc()
    fig_pertype()
    fig_sink_filter()
    fig_asr()
    fig_cascade()
    print("done.")
