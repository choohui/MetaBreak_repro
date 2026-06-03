"""Plot per-prompt cos(A,D) distribution for exp1 & exp2 (final layer)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments_hwichan.per_prompt_cos_AD import (
    RESULTS, load, per_prompt_cos, global_cos,
)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, (name, sub) in zip(axes, [
    ("EXP1 — tail attack", "exp1_llama31_8b"),
    ("EXP2 — position-distributed", "exp2_llama31_8b"),
]):
    rows, hidden = load(RESULTS / sub, pos_offset=0)
    layer = hidden.shape[1] - 1
    vals, _ = per_prompt_cos(rows, hidden, layer)
    g = global_cos(rows, hidden, layer)
    ax.hist(vals, bins=30, color="#4c72b0", alpha=0.85, edgecolor="white")
    ax.axvline(vals.mean(), color="#c44e52", lw=2,
               label=f"per-prompt mean = {vals.mean():+.3f}")
    ax.axvline(g, color="#55a868", lw=2, ls="--",
               label=f"global centroid (report) = {g:+.3f}")
    ax.set_title(f"{name}\n(n={len(vals)} prompts, final layer {layer}, pos0)")
    ax.set_xlabel("cos(A, D)  per prompt")
    ax.set_ylabel("number of prompts")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

fig.suptitle("Per-prompt cos(A, D) distribution  —  within-prompt centroids",
             fontsize=13, y=1.02)
fig.tight_layout()
out = Path(__file__).parent / "results_md" / "per_prompt_cos_AD.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved -> {out}")
