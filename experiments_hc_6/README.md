# experiments_hc_6 - MetaBreak Mask and Hidden-State Steering

Standalone experiment for testing two defenses beyond plain unknown/eos masking:

- search for replacement mask tokens that suppress MetaBreak behavior while keeping benign C/E/F/G semantics stable;
- directly steer hidden states at detected prefill token positions.

## Run

```powershell
cd repro_mb
python -m experiments_hc_6.run_all --smoke --n 3 --skip_generation --run_name smoke_schema
python -m experiments_hc_6.smoke_test
python -m experiments_hc_6.run_all --model <Llama-3.1-8B-Instruct path> --model_type llama --n 150 --run_name llama31_hc6_n150
```

## Stages

- `00_prepare_data`: copies/validates local data inputs and writes replacement token candidates.
- `01_build_prompts`: builds A-G prompt/control rows.
- `02_capture_representations`: captures hidden states and token-level signals.
- `03_split_balance`: performs prompt-level group split and balances `(letter, pos_offset)` cells.
- `04_fit_detectors`: builds scalar detector features and selects validation thresholds.
- `05_threshold_stability`: estimates threshold and AUC stability.
- `06_counterfactual_eval`: compares paired attack/control detector scores.
- `07_mask_candidate_search`: builds non-unk mask candidates and cheap validation metrics.
- `08_mask_eval`: evaluates baseline masking/drop/block actions plus top replacement candidates.
- `09_fit_steering_vectors`: fits layer-wise B/D -> C/E/F/G steering directions from train hidden states.
- `10_steering_eval`: applies hidden-state hooks at flagged prefill positions and evaluates generation.
- `11_stress_controls`: writes additional benign and non-L2 stress controls.
- `12_report`: renders `report.md`, `metrics.json`, and compact CSV summaries.

## Dependency Rule

This package must not import older `experiments_*` packages. Runtime imports are
local to `experiments_hc_6` plus generic repository helpers under `repro_mb/src`
and standard Python dependencies. `smoke_test.py` checks this rule with `rg`.

## Key Knobs

- `--mask_candidate_k`: maximum replacement-token candidates to search.
- `--mask_top_n`: top candidates to carry into real generation.
- `--steer_layers`: `auto` or comma-separated hidden layer ids.
- `--steer_alphas`: comma-separated steering strengths.
- `--steer_modes`: `add,project_out,pull_to_benign`.
- `--semantic_eval_n`: cap for first-token KL/top1 semantic checks.
