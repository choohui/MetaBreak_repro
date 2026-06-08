"""Shared loaders for experiments_hc_7 — consume the validated hc_4_claude
artifacts (``cfg.source_results``) so the steering vector is identical to the
detector and evaluation reuses the SAME held-out split.

What we read from ``source_results``:
  * ``pos{off}/operating_points.json`` — the clean operating point (scalarizer,
    hidden-layer L, threshold, direction).
  * ``pos{off}/scalarizer_fit.npz`` — ``dir__cos_to_attack`` [33, dim] raw TRAIN
    attack centroid (the cos_to_attack geometry).
  * ``pos{off}/scalar_scores.npz`` + ``scalar_scores_meta.json`` — the EXACT
    saved train/held-out split (``is_train``/``is_test`` per balanced token row,
    with ``sample_index`` + per-row ``letters``). Using the saved split sidesteps
    any "did hc_4 use seed=0?" ambiguity — it is bit-identical by construction.
  * ``tokens.jsonl`` — per-token ``position`` (index into the chat-template ids,
    aligned with ``apply_chat_template`` used at generation time) for the
    attack-slot mask + ``features.npz['hidden']`` for benign centroids / rho.
  * ``prompts.jsonl`` — per-sample ``text`` + ``variant`` to (re)generate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import csv as _csv

from experiments_hc_7.core import io  # noqa: E402

ATTACK_LETTERS = ("B", "D", "F")        # attack-variant slot letters (B/D/F prompts)
BENIGN_LETTERS = ("C", "E", "F", "G")   # detection negative class (for contrastive vec)
ATTACK_VARIANTS = ("malicious_mimicry", "malicious_special", "positioned_regular")


# --------------------------------------------------------------------------- #
# operating point + layer
# --------------------------------------------------------------------------- #
def read_operating_point(cfg, off: int) -> dict:
    """The clean (non-classifier) operating point selected by hc_4 stage 05."""
    op = io.read_json(cfg.src_pos_dir(off) / "operating_points.json")
    clean = op.get("clean") or {}
    return clean


def resolve_layer(cfg, off: int, op: dict) -> int:
    """Hidden-layer index L to steer at: --steer_layers override else the op layer."""
    if cfg.steer_layers and off in cfg.steer_layers:
        return int(cfg.steer_layers[off])
    return int(op["layer"])


def block_index(layer: int) -> int:
    """Decoder-block index whose OUTPUT is hidden_states[layer]  (hidden[L] = block L-1)."""
    return int(layer) - 1


# --------------------------------------------------------------------------- #
# saved split + per-token tables
# --------------------------------------------------------------------------- #
def load_source_split(cfg, off: int) -> dict:
    """The exact balanced token rows + saved is_train/is_test for one pos_offset.

    Returns arrays aligned row-for-row: row_id, sample_index, letter, is_train,
    is_test (all length n_rows)."""
    pdir = cfg.src_pos_dir(off)
    meta = io.read_json(pdir / "scalar_scores_meta.json")
    z = np.load(pdir / "scalar_scores.npz")
    letters = np.array(meta["letters"])
    return {
        "row_id": z["row_id"].astype(int),
        "sample_index": z["sample_index"].astype(int),
        "letter": letters,
        "is_train": z["is_train"].astype(bool),
        "is_test": z["is_test"].astype(bool),
        "n_rows": int(meta["n_rows"]),
        "n_train": int(meta.get("n_train", int(z["is_train"].sum()))),
        "n_test": int(meta.get("n_test", int(z["is_test"].sum()))),
    }


def load_hidden(cfg) -> np.ndarray:
    """The full hidden cube [N_rows_global, n_layers, dim] indexed by row_id."""
    return np.load(cfg.source_results / "features.npz")["hidden"]


def load_prompts(cfg) -> dict[int, dict]:
    """{int sample_index: {text, variant, idx}} from the consumed prompts.jsonl."""
    out: dict[int, dict] = {}
    for r in io.read_jsonl(cfg.source_results / "prompts.jsonl"):
        out[int(r["sample_index"])] = {
            "text": r["text"], "variant": r["variant"], "idx": r.get("idx"),
        }
    return out


def held_out_attack_samples(cfg, off: int) -> list[dict]:
    """Unique held-out ATTACK prompts (variant B/D/F) for one pos_offset.

    A prompt is held-out iff its rows are in the saved is_test split. Returns
    [{sample_index, text, variant, letter}] using the consumed prompts.jsonl."""
    split = load_source_split(cfg, off)
    prompts = load_prompts(cfg)
    seen: dict[int, str] = {}
    for si, lt, te in zip(split["sample_index"], split["letter"], split["is_test"]):
        if te and lt in ATTACK_LETTERS and int(si) not in seen:
            seen[int(si)] = lt
    rows = []
    for si, lt in sorted(seen.items()):
        p = prompts.get(si)
        if p is None or p["variant"] not in ATTACK_VARIANTS:
            continue
        rows.append({"sample_index": si, "text": p["text"],
                     "variant": p["variant"], "letter": lt})
    if cfg.limit:
        rows = rows[: cfg.limit]
    return rows


def held_out_benign_samples(cfg, off: int, n: int | None = None) -> list[dict]:
    """Unique held-out BENIGN prompts (variant benign_*/ordinary) for utility/amplify."""
    split = load_source_split(cfg, off)
    prompts = load_prompts(cfg)
    test_samples = sorted({int(si) for si, te in zip(split["sample_index"], split["is_test"]) if te})
    rows = []
    for si in test_samples:
        p = prompts.get(si)
        if p is None or p["variant"] in ATTACK_VARIANTS:
            continue
        rows.append({"sample_index": si, "text": p["text"], "variant": p["variant"]})
    if n is not None:
        rows = rows[:n]
    elif cfg.limit:
        rows = rows[: cfg.limit]
    return rows


def attack_slot_positions(cfg, off: int) -> dict[int, list[int]]:
    """{sample_index: [token positions]} of attack-slot tokens (B/D/F) at this offset.

    ``position`` is the index into the chat-template ids (same template used by
    src.attack.generate_once), so the mask lines up with the generation prompt."""
    out: dict[int, list[int]] = {}
    for r in io.read_jsonl(cfg.source_results / "tokens.jsonl"):
        if int(r.get("pos_offset", -1)) != off:
            continue
        if r.get("letter") in ATTACK_LETTERS:
            out.setdefault(int(r["sample_index"]), []).append(int(r["position"]))
    return out


# --------------------------------------------------------------------------- #
# centroids + rho (model-free, from the hidden cube)
# --------------------------------------------------------------------------- #
def attack_centroid(cfg, off: int, layer: int) -> np.ndarray:
    """Raw TRAIN attack centroid at hidden-layer ``layer`` = dir__cos_to_attack[L]."""
    z = np.load(cfg.src_pos_dir(off) / "scalarizer_fit.npz")
    return z["dir__cos_to_attack"][layer].astype(np.float64)


def benign_centroid(cfg, off: int, layer: int, hidden: np.ndarray) -> np.ndarray:
    """TRAIN benign (negative-class C/E/F/G) centroid at ``layer`` — mirrors the
    detector's negative class, fit on the saved TRAIN rows only."""
    split = load_source_split(cfg, off)
    sel = np.array([(lt in BENIGN_LETTERS) for lt in split["letter"]]) & split["is_train"]
    rids = split["row_id"][sel]
    if len(rids) == 0:
        raise ValueError(f"No TRAIN benign rows for pos{off} layer {layer}.")
    return hidden[rids][:, layer, :].astype(np.float64).mean(axis=0)


def rho_at_layer(cfg, off: int, layer: int, hidden: np.ndarray) -> float:
    """Mean residual-stream norm over held-out attack-slot tokens at ``layer`` —
    the scale that makes alpha interpretable (alpha = fraction of typical norm)."""
    split = load_source_split(cfg, off)
    sel = np.array([(lt in ATTACK_LETTERS) for lt in split["letter"]]) & split["is_test"]
    rids = split["row_id"][sel]
    if len(rids) == 0:
        sel = split["is_test"]
        rids = split["row_id"][sel]
    norms = np.linalg.norm(hidden[rids][:, layer, :].astype(np.float64), axis=1)
    return float(norms.mean()) if len(norms) else 1.0


# --------------------------------------------------------------------------- #
# vectors built by stage 00
# --------------------------------------------------------------------------- #
def load_vectors(cfg, off: int) -> dict:
    """Load stage-00 steer_vectors.npz + build_vectors.json for one pos_offset."""
    pdir = cfg.pos_dir(off)
    z = np.load(pdir / "steer_vectors.npz")
    meta = io.read_json(pdir / "build_vectors.json")
    return {"v": {k: z[k] for k in z.files}, "meta": meta}


# --------------------------------------------------------------------------- #
# CSV reader + alpha* selection (shared by stages 06 and 07)
# --------------------------------------------------------------------------- #
def read_csv(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(_csv.DictReader(f))


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def select_alpha_star(cfg, off: int) -> dict:
    """Pick alpha* on the HEADLINE arm (vector_type=attack, token_mode=all):
    the alpha giving the largest ASR reduction whose benign over-refusal increase
    stays within ``cfg.over_refusal_budget``. ``cfg.alpha_star`` overrides.

    Reads stage-03 ``steer_sweep_asr.csv`` + stage-04 ``steer_utility.csv``."""
    asr = [r for r in read_csv(cfg.pos_dir(off) / "steer_sweep_asr.csv")
           if r.get("vector_type") == "attack" and r.get("token_mode") == "all"]
    util = [r for r in read_csv(cfg.pos_dir(off) / "steer_utility.csv")
            if r.get("vector_type") == "attack"]
    asr_by_a = {_to_float(r["alpha"]): _to_float(r["asr"]) for r in asr}
    or_by_a = {_to_float(r["alpha"]): _to_float(r["over_refusal_rate"]) for r in util}
    base_asr = asr_by_a.get(0.0)
    base_or = or_by_a.get(0.0, 0.0) or 0.0

    if cfg.alpha_star is not None:
        a = float(cfg.alpha_star)
        return {"alpha_star": a, "asr_at_star": asr_by_a.get(a),
                "over_refusal_at_star": or_by_a.get(a), "baseline_asr": base_asr,
                "source": "config"}

    cand = []
    for a, asr_a in asr_by_a.items():
        if a is None or a >= 0 or asr_a is None:   # defense side only (a<0)
            continue
        or_a = or_by_a.get(a)
        incr = (or_a - base_or) if or_a is not None else 0.0
        if incr <= cfg.over_refusal_budget:
            cand.append((asr_a, a))
    if not cand:
        return {"alpha_star": None, "asr_at_star": None, "over_refusal_at_star": None,
                "baseline_asr": base_asr, "source": "none_within_budget"}
    best_asr, best_a = min(cand)                   # smallest ASR (max reduction)
    return {"alpha_star": best_a, "asr_at_star": best_asr,
            "over_refusal_at_star": or_by_a.get(best_a), "baseline_asr": base_asr,
            "source": "selected"}
