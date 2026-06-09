# experiments_attack_attr

A single, structurally-clean reproduction of **`choan.md` §0 → §3.4** — MetaBreak
semantic-mimicry **attack attribution + token-level defense** on
Llama-3.1-8B-Instruct.

Core claim, end to end: the mimicry attack tokens are invisible in the token
*embedding* (§1), but the victim model's *internal representation* separates them
(§2.1); a single internal-rep signal — `cos_to_attack` (clean headline) and
`diff_means` (the token detector) — flags them on held-out prompts (§2.2); and
**dropping each flagged token ±1** sanitizes the prompt, dropping ASR while keeping
utility, with no classifier and no second inference (§3.1–§3.4).

## Self-containment (hard requirement)

This folder has **zero dependency on any sibling `experiments_*` folder**. It uses
only shared, non-experiment code under `repro_mb/`:

- `src/` — the canonical MetaBreak attack (`model_configs`, `mimicry`, `embedding`,
  `attack`, `evaluate`).
- `prompts/` — `Q.txt` (malicious) and `Q_TM-1_Llama.txt` (mimicked).

Everything else is local: datasets in `data/` (`benign_*.jsonl`,
`positioned_regular_words.txt`) and the mimicry signature `data/replacement.json`
are **vendored**; stage 00 regenerates `replacement.json` into `out_dir` for the
actual `--model` via `src.embedding` (so a non-Llama model also works). Verify:

```bash
grep -rE "experiments_(hc|hwichan|yeonseok)" experiments_attack_attr/   # -> empty
```

## The 7 token types (choan.md §2.0)

| | type | role |
|---|---|---|
| A | system special | reference centroid (`cos_to_ref`) |
| **B** | malicious mimicry regular | **attack (positive)** |
| C | benign mimicry regular | negative (token-identity control) |
| **D** | malicious special | **attack (positive)** |
| E | benign special | negative |
| F | positioned regular | negative (position control) |
| G | ordinary regular | negative (body baseline) |

Detection label: positive = B∪D, negative = C∪E∪F∪G, A = reference. All seven types
are count-matched per (type, pos_offset).

## §-to-stage map (the acceptance contract)

| choan.md | stage | output (per `out_dir` or `pos{off}/`) |
|---|---|---|
| §1 embedding not separable | `00_embedding_analysis` | `embedding_analysis.json` (AUC ≈ 0.5) + regenerated `replacement.json` |
| §0 + §2.0 dataset (A-G) | `01_build_prompts` | `prompts.jsonl` |
| baseline ASR | `02_run_asr` | `asr.jsonl`, `asr_summary.json` |
| §2 capture internal rep | `03_capture` | `tokens.jsonl`, `features.npz`, `extract_summary.json` (7-way balanced) |
| §2.1 logistic-probe separable | `04_separability` | `separability.json/.csv` (per-layer probe AUC) |
| §2.2 signals clean①+borderline② | `05_scalars` | `scalarizer_auc.json`, `scalar_scores.npz`, `scalarizer_fit.npz` |
| §2.2 detector (held-out) | `06_detect` | `operating_points.json`, `holdout_eval.json`, `detect_summary.json`, curves, permutation |
| §3.1 masking | `07_defense_mask` | `defense_mask.json` |
| §3.2 steering (partial) | `08_defense_steer` | `defense_steer.json` |
| §3.3 drop±1 (HEADLINE) | `09_defense_drop` | `defense_drop_token_pm1.json` |
| §3.4 conclusion | `10_report` | `final_report.md/.json` |

§2.2 signals — **clean①**: `hidden_norm, value_norm, output_norm, sink,
active_value, active_output, cos_to_ref, cos_to_attack` (headline);
**borderline②**: `diff_means` (= attack−benign projection), `pca_sep_proj`,
`lda_1d`. The §3 defenses flag tokens with `diff_means` by default
(`--defense_family borderline`); `--defense_family clean` uses `cos_to_attack`.

## Running

Model-free smoke test (no GPU / no weights — exercises all of 00–10):

```bash
python -m experiments_attack_attr.smoke_test        # exit 0 = OK
```

Real run (needs a local Llama-3.1-8B-Instruct snapshot + GPU):

```bash
# detection only (fast): §1, dataset, capture, §2.1, §2.2 detector
python -m experiments_attack_attr.run_all --model /path/to/Llama-3.1-8B-Instruct \
    --n 150 --stages 00,01,02,03,04,05,06

# full pipeline with REAL re-generated ASR for the three defenses (§3.1–§3.3)
python -m experiments_attack_attr.run_all --model /path/to/Llama-3.1-8B-Instruct \
    --n 150 --real_intervention
```

Useful flags: `--pos_offsets 0,1` (slot token / its +1), `--defense_family
borderline|clean`, `--mask_mode neutral|unk|eos` (§3.1 control),
`--steer_alphas 2,4,8`, `--asr_judge keyword|guard|both` (`guard` needs
`--guard_model`), `--scalarizer_set all|clean|borderline`.

## ASR measurement: proxy vs real

- **proxy** (always, model-free): held-out block-rate — an attack prompt is blocked
  if any of its B/D tokens is flagged; the drop-semantics lower bound.
- **real** (`--real_intervention` + real model): apply the action (mask / steer /
  drop±1) to the prompt tokens, **re-generate**, and re-judge. This is the only
  honest number for masking (which can *raise* ASR for unk/eos) and steering.

## Rigor

Thresholds are fit on a prompt-level TRAIN split and reported on a disjoint
held-out TEST split (`--holdout_frac`, default 1/3, grouped by prompt so no prompt
straddles the split). Fitted signals (`cos_to_attack`, `diff_means`, …) use their
TRAIN geometry only; stage 05 reports out-of-fold train AUC + bootstrap CIs and
stage 06 adds a prompt-grouped permutation p-value.

## Out of scope

`choan.md` §4 (multi-model Qwen/Mistral/Gemma + Llama-Guard / JBShield / GUARD-SLM
baselines) is **not** implemented here — this folder stops at §3.4 on Llama-3.1.
