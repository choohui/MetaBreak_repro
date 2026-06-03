"""Per-prompt cos(A, D) distribution.

For each prompt, take the A-token hidden states and D-token hidden states at a
given layer, form the within-prompt centroids, and compute cos(centroid_A_p,
centroid_D_p). Collect one value per (qualifying) prompt -> distribution.

Contrast with the report's `cos_to_D`, which uses GLOBAL centroids
(mean over ALL tokens of A vs mean over ALL tokens of D).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CAT_MIMICRY = "A_mimicry_regular"
CAT_SYSTEM = "D_system_special"

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def load(out_dir: Path, pos_offset: int = 0):
    rows = [json.loads(l) for l in (out_dir / "tokens.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    hidden = np.load(out_dir / "features.npz")["hidden"].astype(np.float32)  # [N, L+1, dim]
    keep = [i for i, r in enumerate(rows) if r["pos_offset"] == pos_offset]
    return [rows[i] for i in keep], hidden[keep]


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def per_prompt_cos(rows, hidden, layer):
    # group token-row indices by prompt and category
    by_prompt = defaultdict(lambda: {"A": [], "D": []})
    for i, r in enumerate(rows):
        if r["category"] == CAT_MIMICRY:
            by_prompt[r["sample_index"]]["A"].append(i)
        elif r["category"] == CAT_SYSTEM:
            by_prompt[r["sample_index"]]["D"].append(i)
    vals = []
    n_with_A = 0
    for p, d in by_prompt.items():
        if not d["A"]:
            continue
        n_with_A += 1
        if not d["D"]:
            continue
        cA = hidden[d["A"], layer, :].mean(axis=0)
        cD = hidden[d["D"], layer, :].mean(axis=0)
        vals.append(cos(cA, cD))
    return np.array(vals), n_with_A


def global_cos(rows, hidden, layer):
    Aidx = [i for i, r in enumerate(rows) if r["category"] == CAT_MIMICRY]
    Didx = [i for i, r in enumerate(rows) if r["category"] == CAT_SYSTEM]
    cA = hidden[Aidx, layer, :].mean(axis=0)
    cD = hidden[Didx, layer, :].mean(axis=0)
    return cos(cA, cD)


def ascii_hist(vals, bins=20, width=50):
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(vals, bins=edges)
    mx = counts.max() if counts.max() > 0 else 1
    lines = []
    for k in range(bins):
        bar = "#" * int(round(counts[k] / mx * width))
        lines.append(f"  [{edges[k]:+.3f}, {edges[k+1]:+.3f})  {counts[k]:4d} |{bar}")
    return "\n".join(lines)


def summarize(name, out_dir, layer, pos_offset=0):
    rows, hidden = load(out_dir, pos_offset)
    L = hidden.shape[1]
    layer = L - 1 if layer is None else layer
    vals, n_with_A = per_prompt_cos(rows, hidden, layer)
    g = global_cos(rows, hidden, layer)
    print("=" * 70)
    print(f"{name}  (pos_offset={pos_offset}, layer={layer} / last={L-1})")
    print("-" * 70)
    print(f"  attack prompts with >=1 A token : {n_with_A}")
    print(f"  of which also have >=1 D token   : {len(vals)}  (per-prompt cos computable)")
    if len(vals) == 0:
        print("  -> no prompt has both A and D; per-prompt cos undefined.")
        return
    qs = np.percentile(vals, [0, 5, 25, 50, 75, 95, 100])
    print(f"\n  per-prompt cos(A,D)  mean={vals.mean():+.3f}  std={vals.std():.3f}")
    print(f"    min={qs[0]:+.3f}  p5={qs[1]:+.3f}  p25={qs[2]:+.3f}  "
          f"median={qs[3]:+.3f}  p75={qs[4]:+.3f}  p95={qs[5]:+.3f}  max={qs[6]:+.3f}")
    print(f"\n  GLOBAL centroid cos(A,D) [report's cos_to_D] = {g:+.3f}")
    print(f"  (mean of per-prompt values = {vals.mean():+.3f})")
    print("\n  distribution:")
    print(ascii_hist(vals))


if __name__ == "__main__":
    for name, sub in [("EXP1 (tail attack)", "exp1_llama31_8b"),
                      ("EXP2 (position-distributed)", "exp2_llama31_8b")]:
        out_dir = RESULTS / sub
        if not (out_dir / "features.npz").exists():
            print(f"skip {name}: no features.npz")
            continue
        summarize(name, out_dir, layer=None, pos_offset=0)
        print()
