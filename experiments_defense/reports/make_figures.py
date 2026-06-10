#!/usr/bin/env python
"""Render the experiments_defense (choan.md §4) figures from summary.json.

Minimal-text style: keep axis ticks + short axis labels, drop titles / value
annotations; one compact legend per figure so bars stay interpretable.
"""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SUMMARY = HERE.parent / "results" / "def_all" / "summary.json"
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

S = json.loads(SUMMARY.read_text())
MODELS = S["models"]                       # llama, qwen, gemma
DEFENSES = S["defenses"]                    # ours, llama_guard, jbshield, guard_slm
M = S["metrics"]

# stable colours per defense
COL = {
    "ours": "#2e7d32",          # green  — token sanitize (headline)
    "llama_guard": "#1565c0",   # blue
    "jbshield": "#ef6c00",      # orange
    "guard_slm": "#7b1fa2",     # purple
}
BASE = "#c62828"  # red — no-defense baseline

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def grouped(ax, value_fn, ylabel, ylim=None, baseline_fn=None):
    x = np.arange(len(MODELS))
    w = 0.8 / len(DEFENSES)
    for i, d in enumerate(DEFENSES):
        vals = [value_fn(m, d) for m in MODELS]
        ax.bar(x + (i - (len(DEFENSES) - 1) / 2) * w, vals, w,
               color=COL[d], label=d)
    if baseline_fn is not None:
        for j, m in enumerate(MODELS):
            ax.hlines(baseline_fn(m), x[j] - 0.42, x[j] + 0.42,
                      color=BASE, ls="--", lw=2,
                      label="no-defense baseline" if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(ncol=2, fontsize=8, loc="upper right")


# ---- fig1 : ASR after defense (lower = better), baseline = asr_before -------
fig, ax = plt.subplots(figsize=(7, 4))
grouped(ax,
        lambda m, d: M[m]["defenses"][d]["attack"]["asr_after"],
        "ASR after defense", ylim=(0, 1.0),
        baseline_fn=lambda m: M[m]["baseline"]["asr_before"])
fig.tight_layout(); fig.savefig(OUT / "fig1_asr_after.png"); plt.close(fig)

# ---- fig2 : block rate among originally-successful (higher = better) -------
fig, ax = plt.subplots(figsize=(7, 4))
grouped(ax,
        lambda m, d: M[m]["defenses"][d]["attack"]["block_rate_among_successful"],
        "block rate (of successful attacks)", ylim=(0, 1.05))
fig.tight_layout(); fig.savefig(OUT / "fig2_block_rate.png"); plt.close(fig)

# ---- fig3 : GSM8k(+header) accuracy / utility (higher = better) -----------
#  bars: ours/guard/jbshield/guard_slm ; baseline line = hdr_nodef
fig, ax = plt.subplots(figsize=(7, 4))
grouped(ax,
        lambda m, d: M[m]["defenses"][d]["gsm8k_header"]["acc_after"],
        "GSM8k(+header) accuracy", ylim=(0, 0.8),
        baseline_fn=lambda m: M[m]["baseline"]["gsm8k_acc_header_nodef"])
fig.tight_layout(); fig.savefig(OUT / "fig3_gsm8k_utility.png"); plt.close(fig)

# ---- fig4 : benign refuse/flag rate (lower = better) ----------------------
def benign_fp(m, d):
    b = M[m]["defenses"][d]["benign"]
    return b["flag_rate"] if d == "ours" else b["refuse_rate"]


fig, ax = plt.subplots(figsize=(7, 4))
grouped(ax, benign_fp, "benign false-positive rate", ylim=(0, 0.75))
fig.tight_layout(); fig.savefig(OUT / "fig4_benign_fpr.png"); plt.close(fig)

# ---- fig5 : security vs utility trade-off (the headline) ------------------
#  x = security (block rate of successful attacks) , y = retained utility
#  (GSM8k+header acc_after / hdr_nodef).  marker per model, colour per defense.
MARK = {"llama": "o", "qwen": "s", "gemma": "^"}
fig, ax = plt.subplots(figsize=(6.4, 5.2))
for m in MODELS:
    base_u = M[m]["baseline"]["gsm8k_acc_header_nodef"]
    for d in DEFENSES:
        sec = M[m]["defenses"][d]["attack"]["block_rate_among_successful"]
        util = M[m]["defenses"][d]["gsm8k_header"]["acc_after"] / base_u
        ax.scatter(sec, util, s=170, marker=MARK[m], color=COL[d],
                   edgecolor="k", linewidth=0.6, zorder=3)
ax.axhline(1.0, color="0.6", ls=":", lw=1)   # full-utility line
ax.set_xlabel("security  →  block rate of successful attacks")
ax.set_ylabel("utility  →  retained GSM8k acc (/ no-defense)")
ax.set_xlim(-0.03, 1.05)
ax.set_ylim(-0.05, 1.15)
# two compact legends: colour=defense, marker=model
h_def = [plt.Line2D([], [], marker="o", ls="", color=COL[d], mec="k", label=d)
         for d in DEFENSES]
h_mod = [plt.Line2D([], [], marker=MARK[m], ls="", color="0.4", mec="k", label=m)
         for m in MODELS]
leg1 = ax.legend(handles=h_def, fontsize=8, loc="lower left", title="defense")
ax.add_artist(leg1)
ax.legend(handles=h_mod, fontsize=8, loc="lower right", title="model")
fig.tight_layout(); fig.savefig(OUT / "fig5_tradeoff.png"); plt.close(fig)

print("wrote:", *(p.name for p in sorted(OUT.glob("*.png"))))
