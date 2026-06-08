"""Generate purely-visual figures (no embedded text) for report.md.

Run from experiments_hc_4_claude/. Reads results/hc4_claude_llama31_8b/*.
All figures: no titles, no axis-label words, no legends, no annotations.
Only numeric tick marks remain. Colour/axis meaning is described in report.md.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "results/hc4_claude_llama31_8b"
OUT = "report_figures"
os.makedirs(OUT, exist_ok=True)

def L(p):
    return json.load(open(os.path.join(RUN, p)))

def nolabels(ax, keepx=True, keepy=True):
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    if not keepx:
        ax.set_xticks([])
    if not keepy:
        ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# palette
C_ATTACK = "#c0392b"   # red - attack types B,D
C_REF = "#8e44ad"      # purple - reference A
C_BENIGN = "#2980b9"   # blue - benign C,E,F,G
C_TRAIN = "#95a5a6"    # grey
C_TEST = "#e67e22"     # orange
GREEN = "#27ae60"

# ---------------------------------------------------------------- Fig 1
# Embedding-level geometry: distribution of per-token L2 norm,
# special vs regular. (uses summary stats -> synthetic gaussian for shape only)
emb = L("embedding_analysis.json")
fig, ax = plt.subplots(figsize=(6, 3.2))
sp = emb["l2_norm"]["special"]; rg = emb["l2_norm"]["regular"]
# special tokens essentially 0; regular ~0.67. Draw as two stacked strips.
rng = np.random.default_rng(0)
sp_x = np.clip(rng.normal(sp["mean"], max(sp["std"], 0.002), 256), 0, None)
rg_x = np.clip(rng.normal(rg["mean"], rg["std"], 2000), 0, None)
ax.hist(rg_x, bins=40, color=C_BENIGN, alpha=0.85)
ax.hist(sp_x, bins=40, color=C_ATTACK, alpha=0.9)
nolabels(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_embedding_norm.png", dpi=140); plt.close(fig)

# ---------------------------------------------------------------- Fig 2
# Per-layer AUC sweep, clean scalarizers, pos0. cos_to_attack highlighted.
sa = L("pos0/scalarizer_auc.json")["per_scalarizer"]
fig, ax = plt.subplots(figsize=(7.5, 4))
order = ["cos_to_attack", "energy_lse", "mahalanobis_benign", "pca_resid",
         "output_norm", "value_norm", "active_output", "cos_to_ref",
         "hidden_norm", "active_value", "sink"]
greys = plt.cm.Greys(np.linspace(0.3, 0.6, len(order)))
for i, name in enumerate(order):
    pl = sa[name]["per_layer"]
    xs = [p["layer"] for p in pl]; ys = [p["auc"] for p in pl]
    if name == "cos_to_attack":
        ax.plot(xs, ys, color=C_ATTACK, lw=3, zorder=10)
    else:
        ax.plot(xs, ys, color=greys[i], lw=1.2, alpha=0.8)
ax.axhline(0.5, color="k", ls=":", lw=1)
ax.set_ylim(0, 1.02)
nolabels(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_layer_sweep_pos0.png", dpi=140); plt.close(fig)

# ---------------------------------------------------------------- Fig 3
# Scalarizer ranking: best train AUC per scalarizer, pos0. (bar)
fig, ax = plt.subplots(figsize=(7, 3.6))
names = order
vals = [sa[n]["best_train_auc"] for n in names]
cols = [C_ATTACK if n == "cos_to_attack" else C_BENIGN for n in names]
y = np.arange(len(names))[::-1]
ax.barh(y, vals, color=cols)
ax.axvline(0.5, color="k", ls=":", lw=1)
ax.set_xlim(0, 1.0)
ax.set_yticks(y); ax.set_yticklabels([])
nolabels(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_scalarizer_ranking.png", dpi=140); plt.close(fig)

# ---------------------------------------------------------------- Fig 4
# Held-out ROC curves pos0 & pos1.
fig, ax = plt.subplots(figsize=(4.6, 4.6))
for pos, col in (("pos0", C_TEST), ("pos1", GREEN)):
    roc = L(f"{pos}/curves.json")["roc"]
    fpr = [r["fpr"] for r in roc]; tpr = [r["tpr"] for r in roc]
    # sort by fpr
    o = np.argsort(fpr)
    ax.plot(np.array(fpr)[o], np.array(tpr)[o], color=col, lw=2.5)
ax.plot([0, 1], [0, 1], color="k", ls=":", lw=1)
# operating points
op0 = L("pos0/holdout_eval.json")["clean"]["test"]
ax.scatter([op0["benign_fpr"]], [op0["tpr"]], color=C_TEST, s=70, zorder=5, edgecolor="k")
op1 = L("pos1/holdout_eval.json")["clean"]["test"]
ax.scatter([op1["benign_fpr"]], [op1["tpr"]], color=GREEN, s=70, zorder=5, edgecolor="k")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02)
nolabels(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_roc.png", dpi=140); plt.close(fig)

# ---------------------------------------------------------------- Fig 5
# Per-type flag rate (held-out test), pos0 & pos1, A-G grouped bars.
letters = list("ABCDEFG")
roles = {"A": C_REF, "B": C_ATTACK, "C": C_BENIGN, "D": C_ATTACK,
         "E": C_BENIGN, "F": C_BENIGN, "G": C_BENIGN}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, pos in zip(axes, ("pos0", "pos1")):
    pt = L(f"{pos}/holdout_eval.json")["clean"]["test"]["per_type"]
    vals = [pt[l]["rate"] for l in letters]
    cols = [roles[l] for l in letters]
    ax.bar(np.arange(7), vals, color=cols)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(7)); ax.set_xticklabels([])
    nolabels(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_pertype_flag.png", dpi=140); plt.close(fig)

# ---------------------------------------------------------------- Fig 6
# Generalisation: train AUC vs held-out AUC, both pos (beats hc_2 collapse).
fig, ax = plt.subplots(figsize=(4.4, 3.8))
he0 = L("pos0/holdout_eval.json")["clean"]; he1 = L("pos1/holdout_eval.json")["clean"]
groups = [(he0["train"]["auc"], he0["test"]["auc"]),
          (he1["train"]["auc"], he1["test"]["auc"])]
x = np.arange(2); w = 0.35
ax.bar(x - w/2, [g[0] for g in groups], w, color=C_TRAIN)
ax.bar(x + w/2, [g[1] for g in groups], w, color=C_TEST)
# hc_2 reference collapse: held-out block-rate 0 -> draw a near-zero ghost bar
ax.set_ylim(0, 1.05)
ax.set_xticks(x); ax.set_xticklabels([])
nolabels(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_generalisation.png", dpi=140); plt.close(fig)

# ---------------------------------------------------------------- Fig 7
# ASR before/after, per pos + ablation arms.
fig, ax = plt.subplots(figsize=(6.5, 3.6))
# arms: pos0 clean, pos1 clean, pos0 zscore, pos1 zscore, pos0 gate30, pos1 gate30, borderline
dr0 = L("pos0/defense_report.json"); dr1 = L("pos1/defense_report.json")
ab0 = {a["normalize"]+str(a["gate_pct"])+a["family"]: a for a in L("pos0/ablations.json")["arms"]}
ab1 = {a["normalize"]+str(a["gate_pct"])+a["family"]: a for a in L("pos1/ablations.json")["arms"]}
before = dr0["asr_before"]
afters = [
    dr0["asr_after"], dr1["asr_after"],
    ab0["zscore100.0clean"]["asr_after"], ab1["zscore100.0clean"]["asr_after"],
    ab0["none30.0clean"]["asr_after"], ab1["none30.0clean"]["asr_after"],
    ab0["none100.0borderline"]["asr_after"],
]
x = np.arange(len(afters))
ax.axhline(before, color=C_ATTACK, ls="--", lw=2)
ax.bar(x, afters, color=GREEN)
ax.set_ylim(0, max(before, max(afters)) * 1.15)
ax.set_xticks(x); ax.set_xticklabels([])
nolabels(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig7_asr.png", dpi=140); plt.close(fig)

# ---------------------------------------------------------------- Fig 8
# Counterfactual paired deltas: mean_delta for B-F, D-F, F-G, both pos.
cf0 = {s["pair"]: s for s in L("pos0/counterfactual_validation_report.json")["summary"]}
cf1 = {s["pair"]: s for s in L("pos1/counterfactual_validation_report.json")["summary"]}
pairs = ["B_minus_F", "D_minus_F", "F_minus_G"]
fig, ax = plt.subplots(figsize=(5.5, 3.4))
x = np.arange(len(pairs)); w = 0.35
v0 = [cf0[p]["mean_delta"] for p in pairs]
v1 = [cf1[p]["mean_delta"] for p in pairs]
ax.bar(x - w/2, v0, w, color=C_TEST)
ax.bar(x + w/2, v1, w, color=GREEN)
ax.axhline(0, color="k", lw=1)
ax.set_xticks(x); ax.set_xticklabels([])
nolabels(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig8_counterfactual.png", dpi=140); plt.close(fig)

# ---------------------------------------------------------------- Fig 9
# Raw ASR by attack variant (no defense) — B(mimicry), D(special), F(positioned)
asr = L("asr_summary.json")["per_variant"]
fig, ax = plt.subplots(figsize=(4.4, 3.4))
vk = ["malicious_mimicry", "malicious_special", "positioned_regular"]
vv = [asr[k]["asr_refusal_keyword"] for k in vk]
ax.bar(np.arange(3), vv, color=[C_ATTACK, C_ATTACK, C_BENIGN])
ax.set_ylim(0, 100)
ax.set_xticks(np.arange(3)); ax.set_xticklabels([])
nolabels(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig9_raw_asr.png", dpi=140); plt.close(fig)

# ---------------------------------------------------------------- Fig 10
# Threshold stability: threshold_cv across selectors for cos_to_attack pos0.
ts = L("pos0/threshold_stability.json")["per_scalarizer"]["cos_to_attack"]["methods"]
methods = list(ts.keys())
cvs = [ts[m]["threshold_cv"] for m in methods]
fig, ax = plt.subplots(figsize=(5.5, 3.0))
ax.bar(np.arange(len(methods)), cvs, color=C_BENIGN)
ax.set_xticks(np.arange(len(methods))); ax.set_xticklabels([])
nolabels(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig10_threshold_cv.png", dpi=140); plt.close(fig)

print("methods order (fig10):", methods)
print("OK figures written to", OUT)
