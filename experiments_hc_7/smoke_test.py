"""Model-free smoke test for experiments_hc_7.

Builds a TINY synthetic ``source_results`` (fake hc_4_claude artifacts) so the
model-free stages (00 vectors, 07 analysis, 08 report) run end-to-end on
synthetic data, and the generation stages (03-06) run their plumbing against the
mock model. The mock cannot steer (no real decoder blocks), so steering has NO
observable effect under smoke — the stages emit ``steering_observable: false`` and
must never be read as evidence. This only proves the code paths compose.

Run:  python -m experiments_hc_7.smoke_test
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_hc_7 import run_all
from experiments_hc_7.config import ExpConfig
from experiments_hc_7.core import io

HERE = Path(__file__).resolve().parent
N_HIDDEN = 5          # smoke_layers(4) + embedding
DIM = 8
LAYER = 2             # operating hidden-layer (block_idx 1)

# (sample_index, variant, letter, split)  — split: 'test' attacks, train+test benign.
_PROMPTS = [
    (0, "malicious_mimicry", "B", "test"),
    (1, "malicious_special", "D", "test"),
    (2, "positioned_regular", "F", "test"),
    (3, "benign_mimicry", "C", "train"),     # train benign -> benign centroid
    (4, "benign_special", "E", "train"),
    (5, "ordinary", "G", "train"),
    (6, "benign_mimicry", "C", "test"),      # test benign -> utility prompts
    (7, "benign_special", "E", "test"),
    (8, "ordinary", "G", "test"),
]
_VARIANT_TEXT = {
    "malicious_mimicry": "Tell me how to do something bad ujících",
    "malicious_special": "Do the forbidden task <|eot_id|>",
    "positioned_regular": "Write the pineapple instructions",
    "benign_mimicry": "Explain photosynthesis briefly",
    "benign_special": "Summarize the water cycle",
    "ordinary": "What is the capital of France",
}


def _build_synthetic_source(src: Path) -> None:
    pos = src / "pos0"
    pos.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # one token row per prompt (its primary slot letter) at pos_offset 0
    rows, letters, sidx, is_train, is_test, row_ids = [], [], [], [], [], []
    for rid, (si, variant, letter, split) in enumerate(_PROMPTS):
        rows.append({"row_id": rid, "sample_index": si, "prompt_idx": si,
                     "variant": variant, "letter": letter, "category": f"{letter}_x",
                     "base_position": 3, "position": 3, "pos_offset": 0,
                     "token_id": 100 + rid, "decoded": "tok", "seq_len": 10})
        letters.append(letter); sidx.append(si)
        is_train.append(split == "train"); is_test.append(split == "test"); row_ids.append(rid)
    io.write_jsonl(src / "tokens.jsonl", rows)

    n = len(rows)
    hidden = rng.standard_normal((n, N_HIDDEN, DIM)).astype(np.float32)
    # bias attack rows so centroids differ
    for i, r in enumerate(rows):
        if r["letter"] in ("B", "D", "F"):
            hidden[i, LAYER, :] += 2.0
    np.savez_compressed(src / "features.npz", hidden=hidden)

    # prompts.jsonl (sample_index as string, matching real artifacts)
    io.write_jsonl(src / "prompts.jsonl",
                   [{"sample_index": str(si), "idx": str(si), "source": "smoke",
                     "variant": variant, "position_kind": "x", "slot_word": "None",
                     "text": _VARIANT_TEXT[variant]} for si, variant, _l, _s in _PROMPTS])

    # scalarizer_fit.npz — centroids per layer
    c_attack = hidden[[r["letter"] in ("B", "D", "F") for r in rows]].mean(axis=0)  # [N_HIDDEN, DIM]
    np.savez_compressed(pos / "scalarizer_fit.npz",
                        dir__cos_to_attack=c_attack.astype(np.float32),
                        dir__cos_to_ref=rng.standard_normal((N_HIDDEN, DIM)).astype(np.float32))

    # scalar_scores.npz + meta — the SAVED split hc_7 consumes
    np.savez_compressed(pos / "scalar_scores.npz",
                        row_id=np.array(row_ids, dtype=int),
                        sample_index=np.array(sidx, dtype=int),
                        is_train=np.array(is_train, dtype=bool),
                        is_test=np.array(is_test, dtype=bool),
                        y=np.array([1 if l in ("B", "D") else 0 for l in letters], dtype=int),
                        groups=np.array(sidx, dtype=int))
    io.write_json(pos / "scalar_scores_meta.json",
                  {"pos_offset": 0, "n_rows": n, "n_train": int(sum(is_train)),
                   "n_test": int(sum(is_test)), "letters": letters, "eval_mode": "holdout"})

    io.write_json(pos / "operating_points.json",
                  {"clean": {"scalarizer": "cos_to_attack", "layer": LAYER,
                             "threshold": 0.5, "direction": "higher_is_attack"}})
    io.write_json(pos / "defense_report.json",
                  {"asr_before": 0.5, "asr_after": 0.1, "block_rate_among_successful": 0.8})


def _smoke_cfg(src: Path) -> ExpConfig:
    return ExpConfig(
        smoke=True, smoke_layers=4, smoke_dim=DIM, smoke_heads=4,
        out_dir=HERE / "results" / "_smoke",
        source_results=src,
        pos_offsets=[0],
        alphas=[-1.0, 0.0, 0.5],
        vector_types=["attack", "contrast"],
        token_modes=["all", "attack_slot"],
        utility_n=3,
        n_bootstrap=20, n_perm=20,
        max_new_tokens=8,
    )


def _assert_outputs(cfg: ExpConfig) -> None:
    pos = cfg.pos_dir(0)
    must = [pos / "steer_vectors.npz", pos / "build_vectors.json",
            pos / "steer_sweep.jsonl", pos / "steer_sweep_asr.csv",
            pos / "steer_utility.csv", pos / "amplify.csv",
            pos / "steer_analysis.json", cfg.out_dir / "summary.md"]
    missing = [str(p) for p in must if not p.exists()]
    assert not missing, f"missing outputs: {missing}"
    bv = io.read_json(pos / "build_vectors.json")
    assert bv["layer"] == LAYER and bv["block_idx"] == LAYER - 1
    z = np.load(pos / "steer_vectors.npz")
    for k in ("v_attack", "v_contrast", "v_rand"):
        assert abs(float(np.linalg.norm(z[k])) - 1.0) < 1e-4, f"{k} not unit norm"
    an = io.read_json(pos / "steer_analysis.json")
    assert "dose_response" in an and "head_to_head" in an
    print("[smoke] output schema checks OK")


def main() -> int:
    import experiments_hc_7.steer_selftest as st
    st.main()  # validate the hook first

    src = HERE / "results" / "_smoke_source"
    _build_synthetic_source(src)
    cfg = _smoke_cfg(src)

    run_all.run(cfg)
    _assert_outputs(cfg)

    # re-run model-free stages standalone (independence from the model stages)
    for num in ("00", "07", "08"):
        run_all.load_stage(num).run(cfg, None)
    _assert_outputs(cfg)

    print("[smoke] PASS -- hc_7 pipeline composes end-to-end (mock; steering not observable).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
