# experiments_hc_4 - Active Sink % Sweep

This experiment is self-contained. Runtime code does not import other
`experiments_*` packages and does not read existing `results` artifacts.

Allowed runtime inputs:

- `experiments_hc_4/data/*`
- generic helpers under `repro_mb/src`
- the model path passed with `--model`

The copied seed files in `data/` come from earlier prompt/control seeds, but the
code reads only the local copies.

## Run

```bash
python experiments_hc_4/run_all.py \
  --model_type llama \
  --model /path/to/Llama-3.1-8B-Instruct \
  --n 50 \
  --keep_pcts 5,10,20,30,50,100 \
  --fpr 0.01 \
  --run_name llama_n50_active_value
```

Smoke test without a real model:

```bash
python experiments_hc_4/run_all.py --smoke --n 3 --run_name smoke
```

## Stages

- `00_embedding`: writes `replacement.json`.
- `01_build_prompts`: builds A-G prompt rows from local `data/`.
- `02_capture`: captures sink, value norm, output norm, hidden norm, and
  `active_value = sink * value_norm` per analyzed token. It also writes
  `responses.jsonl` for ASR unless `--skip_generation` is set.
- `03_active_pct_threshold`: sweeps top active-value keep percentages and fits
  a high-score threshold at target benign FPR.
- `04_report`: renders `pct_threshold_report.md`.

## Outputs

Outputs are written to `experiments_hc_4/results/<run_name>/`, which is ignored
by `.gitignore` via `*results`.

- `replacement.json`
- `prompts.jsonl`
- `active_value_rows.jsonl`
- `responses.jsonl`
- `pct_threshold_report.json`
- `pct_threshold_report.md`
- `sweep_summary.csv`

## Dependency Checks

```bash
rg "experiments_hc_[123]|experiments_hwichan|experiments_yeonseok" experiments_hc_4
rg "classifier_results|embedding_results|results/hc" experiments_hc_4
```

