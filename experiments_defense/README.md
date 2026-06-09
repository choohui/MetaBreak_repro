# experiments_defense — choan.md §4 multi-model defense comparison

Answers **§4 of [choan.md](../choan.md)**: *is the "diff-means + ±1-token-drop"
sanitizing defense applicable across models, and how does it compare to existing
defenses?* It evaluates **4 defenses** across up to **3 models** on **2 prompt
sets**, producing a models×defenses comparison (`results/<run>/REPORT.md`).

The headline (choan.md "main contribution"): **token-level sanitizing preserves
utility**, unlike prompt-level detect-and-refuse defenses that reject the whole
prompt — the GSM8k(+mimicry-header) table is what makes that visible.

## The 4 defenses

| name | family | mechanism | file |
|---|---|---|---|
| `ours` | token-level **sanitize** | diff-means flags malicious tokens → drop flagged ±1 → regenerate | [defenses/ours.py](defenses/ours.py) |
| `llama_guard` | prompt-level **refuse** | Llama-Guard-3 classifies the user prompt; unsafe → refuse | [defenses/llama_guard.py](defenses/llama_guard.py) |
| `jbshield` | prompt-level **refuse** | JBShield-D: input activates BOTH toxic & jailbreak concept directions → refuse | [defenses/jbshield.py](defenses/jbshield.py) |
| `guard_slm` | prompt-level **refuse** | GUARD-SLM: per-layer last-token SVM → malicious → refuse | [defenses/guard_slm.py](defenses/guard_slm.py) |

`jbshield` and `guard_slm` are **reimplemented from their papers**
([NISPLab/JBShield](https://github.com/NISPLab/JBShield),
[solidlabnetwork/GUARD-SLM](https://github.com/solidlabnetwork/GUARD-SLM)) in this
repo's style and calibrated on our data, because the official repos don't support
our target models and use incompatible data formats. Only **JBShield-D**
(detection→refuse) is implemented; JBShield-M steering is future work.

## Models & data

- **Models**: `llama` (Llama-3.1-8B-Instruct), `qwen` (Qwen2.5-7B-Instruct),
  `gemma` (Gemma-2-9B-it). Family chat-templates come from `src.model_configs`.
- **Attack set**: `prompts/MetaBreak_data/Q_TM-1_<Model>.txt` (MetaBreak TM-1
  mimicry). Re-mimicked with `results/<model>/replacement.json` when present
  (Llama has it today; Qwen/Gemma fall back to the literal-special form until you
  generate theirs via the existing `src` attack pipeline).
- **Benign**: `experiments_yeonseok/benign_prompts.jsonl` (utility + FPR controls).
- **GSM8k**: `openai/gsm8k` (downloaded via `datasets`), plus a header-injected
  variant — the MetaBreak mimicry header prepended to a benign math question so it
  looks structurally like an attack. Scored by final-integer exact match.

## What it imports

Self-contained per repo convention: depends ONLY on `repro_mb/src`
(`src.model_configs`, `src.evaluate.matches_refusal`, `src.evaluate.GuardJudge`).
The diff-means direction fit, ±1 drop, assistant-header span detection, and mock
model are **copied** into `core/` (from experiments_hc_4 / experiments_hc_4_claude),
not cross-imported.

## Layout

```
config.py            ExpConfig + argparse
run_all.py           orchestrator (load model once per family → prepare → eval → report)
smoke_test.py        model-free validation on the mock model (exit 0 == OK)
core/                model, capture (hidden-only), mock, template, data, judge,
                     defense_base (Defense protocol + GuardResult + eval_one), stats, io
defenses/            ours, llama_guard, jbshield, guard_slm  (+ REGISTRY)
stages/              01_build_data  02_prepare_defenses  03_evaluate  04_report
results/<run>/       data/<model>/*.jsonl, <model>/{prepare,metrics}.json,
                     <model>/eval_*_*.jsonl, REPORT.md, summary.json
```

Each `stages/NN_*.py` exposes `run(...)`; `run_all.py` holds the prepared defense
objects in-memory between stage 02 and 03 (their SVM / direction state is not
round-tripped through disk).

## Metrics (in `REPORT.md`)

Per model × defense:
- **Attack**: `asr_after` and `block_rate_among_successful` (vs no-defense `asr_before`).
- **GSM8k+header**: `acc_after` (utility) vs no-defense `plain` / `hdr_nodef`.
- **Benign**: false-positive `refuse_rate` (prompt-level) / `flag_rate` (`ours`).

## Run

```bash
# model-free smoke test (no weights / GPU / network) — the primary gate
python experiments_defense/smoke_test.py

# real single-model run (Llama-3.1 + Llama-Guard-3 already local)
python experiments_defense/run_all.py --models llama \
    --model_path llama=/path/to/Llama-3.1-8B-Instruct \
    --guard_model /path/to/Llama-Guard-3-8B --n_gsm8k 50
```

### Running several models at once

`run_all` loops the models **sequentially** in one command, freeing each model's
GPU memory (victim + any guard model) before loading the next — so a single GPU
can cover all three. Pass them comma-separated, or `--models all`:

```bash
python experiments_defense/run_all.py --models all \
    --model_path llama=/m/Llama-3.1-8B-Instruct \
    --model_path qwen=/m/Qwen2.5-7B-Instruct \
    --model_path gemma=/m/gemma-2-9b-it \
    --guard_model /m/Llama-Guard-3-8B
```

To avoid retyping paths, copy `models.json.example` → `models.json` and fill in
the checkpoints; `run_all` auto-loads it (CLI flags still override):

```bash
python experiments_defense/run_all.py --models all          # paths from models.json
```

A model that fails to load (e.g. not downloaded) is logged and **skipped**; the
run continues and the report covers whatever finished. If a single GPU can't hold
the victim **and** Llama-Guard together, run `--defenses ours,jbshield,guard_slm`
and `--defenses llama_guard` as two passes.

Useful flags: `--defenses ours,llama_guard` (subset), `--n_attack`/`--n_benign`/
`--n_gsm8k` (caps), `--no_mimicry` (keep literal-special attacks), `--dtype`,
`--device`, `--run_name`.

## Design notes / caveats

- `jbshield` and `guard_slm` pool the prompt with the **last-token** hidden state
  (standard practice); our capture appends the generation-prompt header, so "last
  token" is the header end — context-dependent and consistent across prompts.
- `ours` only flags/drops tokens in the **user-content region** (between the
  template prefix and the final assistant header) so the chat wrapper is never
  corrupted, and expands each flag to ±1 neighbours.
- Defense thresholds are fit on the **TRAIN** split only (`ours`/`jbshield` at a
  low FPR, `guard_slm` SVM by best train AUC).
- `llama_guard` uses a deterministic stub under `--smoke` / mock / when no
  `--guard_model` is given.
