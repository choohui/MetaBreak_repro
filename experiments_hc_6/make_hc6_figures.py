from __future__ import annotations

"""Render text-free figures for the experiments_hc_6 report.

Every figure is intentionally text-free: no titles, no axis labels, no tick
labels, no legends, no in-figure annotations. Tick *positions* are kept only so
that light gridlines give a visual sense of scale; their text labels are blanked.
All quantitative meaning is carried by the report prose. Colour conventions:

    attack (B/D)          -> red      (#C44536)
    benign (C/E/F/G)      -> blue     (#277DA1)
    reference (A)         -> gray     (#777777)
    neutral / placeholder -> green    (#4D908E)
    highlight / gold      -> gold     (#F9C74F)
    secondary             -> purple   (#6D597A)
"""

import collections
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).resolve().parent / "results" / "llama31_hc6_n150"
OUT = RES / "figures"
OUT.mkdir(exist_ok=True)

C = {
    "attack": "#C44536",
    "benign": "#277DA1",
    "ref": "#777777",
    "neutral": "#4D908E",
    "gold": "#F9C74F",
    "purple": "#6D597A",
    "light": "#E6E6E6",
}


def load_json(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def finish(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def blank(ax, *, x=True, y=True, grid="y") -> None:
    """Strip all text but keep tick positions so gridlines survive."""
    if grid in ("y", "both"):
        ax.grid(axis="y", color=C["light"], linewidth=0.9)
    if grid in ("x", "both"):
        ax.grid(axis="x", color=C["light"], linewidth=0.9)
    ax.set_axisbelow(True)
    if x:
        ax.set_xticklabels([])
    if y:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
def fig0_census() -> None:
    cap = load_json("capture_summary.json")["census"]
    order = list("ABCDEFG")
    vals = [cap[f"{L}_pos0"] + cap[f"{L}_pos1"] for L in order]
    colors = [C["ref"] if L == "A" else C["attack"] if L in "BD" else C["benign"] for L in order]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(np.arange(len(order)), vals, color=colors, width=0.7)
    ax.set_xticks(np.arange(len(order)))
    ax.set_yticks([0, 1000, 2000, 3000])
    blank(ax)
    finish(fig, "fig0_dataset_census.png")


def fig1_auc_by_layer() -> None:
    rows = list(csv.DictReader((RES / "scalar_discovery.csv").open(encoding="utf-8")))
    proto, cos = {}, {}
    for r in rows:
        f = r["feature"]
        if f.startswith("proto_attack_minus_benign_L"):
            proto[int(f.rsplit("L", 1)[1])] = float(r["val_auc"])
        elif f.startswith("cos_to_attack_L"):
            cos[int(f.rsplit("L", 1)[1])] = float(r["val_auc"])
    L = sorted(set(proto) | set(cos))
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(L, [proto.get(i, np.nan) for i in L], "-o", color=C["attack"], lw=2.2, ms=4)
    ax.plot(L, [cos.get(i, np.nan) for i in L], "-s", color=C["purple"], lw=2.0, ms=3.5)
    ax.axhline(0.5, color=C["ref"], lw=1.0, ls="--")
    ax.set_ylim(0.45, 1.02)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    blank(ax)
    finish(fig, "fig1_detector_auc_by_layer.png")


def fig2_score_hist() -> None:
    d = np.load(RES / "scalar_values.npz", allow_pickle=True)
    names = list(d["feature_names"])
    x = d["x"]
    rid = d["row_ids"]
    col = names.index("proto_attack_minus_benign_L11")
    letter = {}
    for line in (RES / "balanced_tokens.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        letter[int(r["row_id"])] = r["letter"]
    by = collections.defaultdict(list)
    for i, rr in enumerate(rid):
        by[letter[int(rr)]].append(float(x[i, col]))
    atk = np.array(by["B"] + by["D"])
    ben = np.array(by["C"] + by["E"] + by["F"] + by["G"])
    lo, hi = -2.0, 2.0
    bins = np.linspace(lo, hi, 46)
    thr = load_json("threshold_rules.json")["selected"]["0.01"]["terms"][0]["threshold"]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.hist(np.clip(ben, lo, hi), bins=bins, color=C["benign"], alpha=0.8)
    ax.hist(np.clip(atk, lo, hi), bins=bins, color=C["attack"], alpha=0.7)
    ax.axvline(thr, color="black", lw=1.6, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_xticks([-2, -1, 0, 1, 2])
    blank(ax, grid="y")
    finish(fig, "fig2_score_separation_hist.png")


def fig3_per_letter() -> None:
    per = load_json("mask_eval.json")["token_eval"]["per_letter"]
    order = list("BCDEFG")
    vals = [per[L]["rate"] for L in order]
    colors = [C["attack"] if L in "BD" else C["benign"] for L in order]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(np.arange(len(order)), vals, color=colors, width=0.66)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(len(order)))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    blank(ax)
    finish(fig, "fig3_per_letter_flag_rate.png")


def fig4_generalization() -> None:
    st = load_json("threshold_stability.json")["rules"]["0.01"]["split_eval"]
    splits = ["train", "val", "test"]
    recall = [st[s]["recall"] for s in splits]
    fpr = [st[s]["fpr"] for s in splits]
    x = np.arange(len(splits))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.bar(x - w / 2, recall, w, color=C["neutral"])
    ax.bar(x + w / 2, fpr, w, color=C["attack"])
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    blank(ax)
    finish(fig, "fig4_split_generalization.png")


def fig5_counterfactual() -> None:
    pairs = load_json("counterfactual_report.json")["pairs"]
    order = ["B_to_F", "D_to_F", "B_to_C", "D_to_E"]
    look = {p["pair"]: p for p in pairs}
    vals = [look[k]["median_delta"] for k in order]
    colors = [C["attack"], C["attack"], C["gold"], C["gold"]]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.bar(np.arange(len(order)), vals, color=colors, width=0.62)
    ax.axhline(0.0, color=C["ref"], lw=1.0)
    ax.set_xticks(np.arange(len(order)))
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    blank(ax)
    finish(fig, "fig5_counterfactual_delta.png")


def fig6_mask_candidates() -> None:
    rows = list(csv.DictReader((RES / "mask_candidate_eval.csv").open(encoding="utf-8")))
    words = [r for r in rows if "neutral_text" in r["sources"]][:6]
    specials = [r for r in rows if r["candidate_text"].startswith("<|reserved")][:3]
    sel = words + specials
    sel = sorted(sel, key=lambda r: float(r["val_attack_cleared_rate"]))
    vals = [float(r["val_attack_cleared_rate"]) for r in sel]
    colors = [C["neutral"] if "neutral_text" in r["sources"] else C["ref"] for r in sel]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh(np.arange(len(sel)), vals, color=colors, height=0.66)
    ax.set_xlim(0, 0.55)
    ax.set_yticks(np.arange(len(sel)))
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    blank(ax, grid="x")
    finish(fig, "fig6_mask_candidates.png")


def fig7_actions() -> None:
    a = load_json("mask_eval.json")["actions"]
    order = [
        ("no_op", C["ref"]),
        ("unk_or_eos_mask", C["ref"]),
        ("mask__single_4037", C["gold"]),
        ("drop_token", C["benign"]),
        ("drop_detected_span", C["benign"]),
        ("drop_token_pm1", C["neutral"]),
        ("prompt_block", C["attack"]),
    ]
    vals = [a[k]["asr_after"] for k, _ in order]
    colors = [c for _, c in order]
    before = a["no_op"]["asr_before"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.bar(np.arange(len(order)), vals, color=colors, width=0.7)
    ax.axhline(before, color="black", lw=1.4, ls="--")
    ax.set_ylim(0, 0.72)
    ax.set_xticks(np.arange(len(order)))
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    blank(ax)
    finish(fig, "fig7_defense_actions_asr.png")


def fig8_steering() -> None:
    grid = load_json("steering_eval.json")["grid"]
    modes = {"add": C["attack"], "project_out": C["benign"], "pull_to_benign": C["neutral"]}
    series = collections.defaultdict(list)
    for g in grid:
        if g["mode"] == "no_op":
            base = g["asr_after"]
        else:
            series[g["mode"]].append((g["alpha"], g["asr_after"]))
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for m, col in modes.items():
        pts = sorted(series[m])
        xs = [0.0] + [p[0] for p in pts]
        ys = [base] + [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color=col, lw=2.2, ms=5)
    ax.axhline(base, color="black", lw=1.2, ls="--")
    ax.set_ylim(0.35, 0.68)
    ax.set_xticks([0, 0.25, 0.5, 1.0, 2.0])
    ax.set_yticks([0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    blank(ax)
    finish(fig, "fig8_steering_curve.png")


def main() -> None:
    fig0_census()
    fig1_auc_by_layer()
    fig2_score_hist()
    fig3_per_letter()
    fig4_generalization()
    fig5_counterfactual()
    fig6_mask_candidates()
    fig7_actions()
    fig8_steering()
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
