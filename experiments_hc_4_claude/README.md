# experiments_hc_4_claude

A **self-contained** experiment: a per-token defense against MetaBreak
semantic-mimicry prompt-injection that reduces an internal representation to **one
scalar per token** and applies a **threshold** to flag and exclude malicious
tokens — **no logistic-regression classifier**. It tests whether such a defense
generalises to held-out prompts and reduces ASR, where hc_2's single-threshold
cascade collapsed to 0% block-rate on held-out. See [Main.md](Main.md) for the spec.

## Layout

```
experiments_hc_4_claude/
├── Main.md / README.md
├── config.py            # ExpConfig dataclass + argparse (all knobs)
├── run_all.py           # orchestrator (stages 00-09); loads model once
├── smoke_test.py        # mock-model full pipeline + re-run 04-09 from disk
├── core/                # copied hc_2 utilities + NEW scalar-defense modules
│   ├── labels, io, metrics, model, capture, template, labeling, features, mock, splits   (copied)
│   ├── scalarizers.py   # NEW: clean + borderline scalarizers + per-prompt normalisation
│   ├── thresholds.py    # NEW: TRAIN-fit selectors + threshold_stability
│   ├── stats.py         # NEW: prompt-grouped bootstrap CI + permutation test
│   ├── curves.py        # NEW: ROC/DET/PR/calibration points
│   └── intervene.py     # NEW: optional real token-exclusion re-generation
├── stages/
│   ├── 00..03           # copied from hc_2 (03 modified: 7-way balance incl. A)
│   ├── scalar_common.py # NEW: shared loader / split / honest-AUC / persistence
│   └── 04..09           # NEW: scalarize, threshold, holdout, counterfactual, defense, robustness
├── data/                # copied seeds (benign_*_prompts.jsonl, positioned_regular_words.txt)
└── results/             # outputs (results/_smoke/ for the smoke run)
```

**Dependency rule.** No imports from any other `experiments_*` folder. Allowed:
this package's own `core/`, and files directly under `repro_mb/` (`src/*`,
`prompts/Q*.txt`, `results/llama/replacement.json`).

## Run

```bash
# from repro_mb/ (the directory above this package)
python -m experiments_hc_4_claude.smoke_test                      # model-free, exit 0 = OK

python -m experiments_hc_4_claude.run_all \
    --model /path/to/Llama-3.1-8B-Instruct --n 150                # full run

python -m experiments_hc_4_claude.run_all --model ... --stages 04,05,06   # subset
python -m experiments_hc_4_claude.run_all --model ... --real_intervention  # real ASR re-gen
```

## Key knobs (see `config.py`)

| knob | meaning |
|---|---|
| `--balance_a/--no-balance_a` | include A in the 7-way equal-count cap (default on) |
| `--scalarizer_set clean\|borderline\|all` | which family is the headline (default clean) |
| `--scalarizers a,b,c` | explicit scalarizer list (overrides the set) |
| `--normalize none\|zscore\|rank\|robust` | per-prompt normalisation wrapper |
| `--threshold_methods youden,fpr@1,...` | TRAIN-fit threshold selectors |
| `--holdout_frac 0.33 --seed 0` | prompt-level held-out test split |
| `--cv_folds 5 --n_bootstrap 1000 --n_perm 1000` | rigor budget |
| `--sink_gate_pct 30` | optional 1st-stage sink gate (ablation arm; 100 = off) |
| `--real_intervention` | stage 08 drops flagged tokens and re-generates (needs `--model`) |

## Outputs (per `results/<run>/`)

- `extract_summary.json` — census with **equal A–G counts** (`cap_mode=balanced7`).
- `pos{off}/scalarizer_auc.json` — per-scalarizer per-layer AUC (out-of-fold for fitted ones), best layer, bootstrap CI, borderline tag.
- `pos{off}/threshold_stability.json`, `operating_points.json` — TRAIN-selected op-point (`selected_on:"train"`) with `threshold_cv`; clean + borderline separately.
- `pos{off}/holdout_eval.json`, `curves.json`, `permutation_test.json` — **the honest held-out numbers** (the hc_2 scenario).
- `pos{off}/counterfactual_*` — paired deltas B−C, B−F, D−E, D−F, F−G.
- `pos{off}/defense_report.json` (+ `real_asr.json`) — ASR before/after (proxy; real if requested).
- `pos{off}/ablations.json`, `robustness_report.md`, `summary.md` — ablation arms + aggregate.

## Notes

- **Headline vs borderline.** `diff_means`/`lda_1d`/`pca_sep_proj` fit a 1-D direction
  on train and are tagged `borderline`; they are reported in their own operating point
  and never enter the clean (`--scalarizer_set clean`) claim.
- **Counterfactual `B−C` / `D−E` may show `paired_auc=null`** when the benign-control
  prompts (C/E) do not share a `prompt_idx` with the attack prompts; `B−F`, `D−F`,
  `F−G` always pair (same source prompt). All five pairs are always listed.
- **Cost on the real model.** The covariance/PCA scalarizers (`mahalanobis_benign`,
  `energy_lse`, `pca_resid`, `lda_1d`) and the out-of-fold AUC do per-layer linear
  algebra at the model's hidden dim (4096) × layers. For a fast pass, subset with
  `--scalarizers` or restrict `--pos_offsets`.
- **Smoke** runs with `pos_offsets=[0]` so every category is present at the analysed
  offset and the 7-way balance is exactly equal.
