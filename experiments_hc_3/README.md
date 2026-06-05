# experiments_hc_3

hc_3 tests the recommended attention-sink defense ideas in this order:

1. Active SinkProbe
2. Prompt-Level Aggregation
3. Two-Branch Cascade
4. Counterfactual paired-control validation

By default, hc_3 is model-free. It reads compatible artifacts from:

```powershell
repro_mb\experiments_hc_2\results\hc2_llama31_8b
```

and copies the core artifacts into:

```powershell
repro_mb\experiments_hc_3\results\hc3_active_sink
```

## Run

From `repro_mb`:

```powershell
python -m experiments_hc_3.run_all
```

Custom source/output:

```powershell
python -m experiments_hc_3.run_all `
  --source_out_dir .\experiments_hc_2\results\hc2_llama31_8b `
  --out_dir .\experiments_hc_3\results\hc3_active_sink `
  --asr_judge both `
  --fpr 0.01 `
  --token_recall 0.95
```

Run one stage:

```powershell
python -m experiments_hc_3.run_all --stages 08
```

## Outputs

Each `pos{offset}` directory contains:

- `active_sinkprobe_report.md/json`
- `active_sinkprobe_features.npz`
- `active_sinkprobe_scores.jsonl`
- `prompt_aggregation_report.md/json`
- `two_branch_cascade_report.md/json`
- `counterfactual_validation_report.md/json`
- `counterfactual_paired_deltas.csv`
- `counterfactual_manifest.jsonl`

## Method Notes

The current hc_2 artifacts store mean-over-head sink scores for labeled tokens.
Therefore hc_3 implements a labeled-token SinkProbe: sink ranks/top-k flags are
computed among analyzed rows in the same prompt and `pos_offset`. A future hc_3
extractor can improve this by saving all-token and per-head sink tensors.

