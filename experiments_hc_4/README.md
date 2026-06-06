# experiments_hc_4 - Scalar Threshold Defense

This package is a standalone scalar-threshold defense experiment.

It builds A-G prompt types, captures internal signals, balances token counts by
`(letter, pos_offset)` inside each train/val/test split, discovers scalar
features, selects threshold-only rules, and evaluates token-removal defenses.

## Run

```powershell
cd repro_mb
python -m experiments_hc_4.run_all --smoke --n 3 --run_name smoke
python -m experiments_hc_4.run_all --model <Llama-3.1-8B-Instruct path> --model_type llama --n 150 --run_name llama31_scalar_n150
```

## Stages

- `00_embedding`: writes replacement tokens.
- `01_build_prompts`: builds A-G prompt rows.
- `02_capture`: captures sink, value, output, hidden norm, hidden vectors, and baseline responses.
- `03_balance_split`: splits by prompt group and balances every `(A-G, pos0/pos1)` cell.
- `04_scalar_discovery`: creates raw, rank, trajectory, robust-tail, and centroid scalar features.
- `05_threshold_rules`: selects threshold-only rules without logistic regression.
- `06_apply_defense`: removes or masks flagged user-content tokens and re-runs generation.
- `07_counterfactual`: compares attack/control paired prompt scores.
- `08_stress_controls`: writes local safe, mimicry, and non-L2 structure controls.
- `09_report`: renders `final_report.md`.

## Dependency Rule

Runtime imports are local to this package plus generic helpers under `repro_mb/src`.
The package does not import older experiment packages.

