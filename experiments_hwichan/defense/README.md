# defense — how much does each guard lower **ASR**?

The sibling studies (`defense_thresholds.py`, `cascade_defense.py`) measure
**token-level TPR/FPR/AUC** — *how separable* attack tokens are from benign ones.
That answers "is there a signal?" but **not** "if I actually deploy this detector
as a guard, how far does the attack success rate drop, and at what utility cost?".

This module measures exactly that: for each defense, the **attack success rate
(ASR) before vs. after** deployment, plus the benign over-refusal it causes.

## What it does

```
build attack + benign prompts
        │
        ├─ generate victim responses for attack prompts → judge jailbreak (refusal-kw / Llama-Guard)  → ASR_no_defense
        │
        ├─ forward_capture per prompt → per-content-token internal signals
        │
        ├─ calibrate every guard on a held-out CALIBRATION split
        │
        └─ on the EVAL split, block-on-detect → ASR_with_defense, recall, benign block-rate
```

**Block-on-detect semantics.** A blocked prompt is replaced by a fixed refusal, so
a blocked attack *cannot* succeed. Therefore

```
ASR_with_defense = mean( jailbroken  AND  NOT blocked_by_guard )   over the eval attacks
```

## Defenses compared (all on one ASR axis)

| guard | signal | where |
|---|---|---|
| `internal:<feature>@L<layer>` | `sink` / `hidden_norm` / `value_norm` / `output_norm` / `cos_to_D`, **max over content tokens**, single threshold | inside the model |
| `cascade:sink->...` | cheap `sink` recall-gate, then a hidden feature on survivors | inside the model |
| `input_l2_guard` | L2 embedding distance of regular tokens to chat specials (experiments_yeonseok) | tokenization boundary |
| `none` | — (baseline ASR) | — |

Each internal/cascade guard is reported at several **operating points**:
`fpr0.01 / fpr0.05 / fpr0.1` (threshold calibrated to that benign block-rate) and
`youden` (max TPR−FPR on the calibration split).

## Why thresholds are re-calibrated here (not reused from `defense_report.json`)

Those reports calibrate at the **token** level. A deployed guard blocks a whole
**prompt**, and we flag a prompt if *any* of its content tokens trips the threshold —
so the prompt-level FPR is strictly higher than the token-level FPR. We therefore
re-calibrate each threshold to a target **prompt-level** benign block-rate, on a
held-out calibration split that the eval ASR never sees.

## Threat model / honesty notes

- **Attack prompts** = MetaBreak TM-1: `mimicked` (regular-token disguise, the real
  attack) and optionally `malicious` (literal special tokens kept). `mimicked` is the
  default — it's what evades the naive literal-special-id check.
- **Benign control** = clean `Q.txt` queries + `benign_special_prompts.jsonl` (a
  special token used in a harmless context). The benign block-rate on these is the
  utility cost / over-refusal.
- The internal guards require a **forward pass of the victim model** on the input —
  realistic for a self-hosted deployment, not for a black-box API caller.
- Calibration and evaluation use **disjoint** prompt splits (`--calib_frac`).

## Run (GPU server, Llama-3.1-8B-Instruct)

```bash
cd repro_mb
pip install -r requirements.txt -r experiments_hwichan/requirements_extra.txt

# smoke test (verify shapes / hooks) — tiny n
python experiments_hwichan/defense/run_defense_asr.py \
  --model <local-llama31-8b-snapshot> --n_attack 6 --n_benign 12 --include_l2_guard

# full run, both judges
python experiments_hwichan/defense/run_defense_asr.py \
  --model <local-llama31-8b-snapshot> \
  --replacement experiments_yeonseok/results/l2_guard_llama31_8b_n450/common/replacement.json \
  --n_attack 60 --n_benign 120 \
  --attack_variants mimicked,malicious \
  --include_l2_guard \
  --guard_model <local-Llama-Guard-3-8B>      # optional; default judge = refusal keyword
```

Key flags: `--features` (which internal features to calibrate as guards),
`--fpr_targets`, `--recall_targets` (cascade gate), `--calib_frac` (split),
`--temperature` (0 = greedy). See `run_defense_asr.py --help`.

## Outputs (under `--out_dir`, default `results/asr_llama31_8b/`)

- `asr_defense_report.json` / `.md` — the headline table:
  `ASR no-def → ASR w/def`, absolute (pp) and relative (%) drop, recall over the
  attacks that *actually succeeded*, and the benign block-rate, **sorted by lowest
  surviving ASR** (ties broken by lower over-refusal). Plus a per-variant ASR
  breakdown and the calibrated layer/orientation per feature.
- `attack_eval_judged.json` — per-prompt jailbreak judgements (for inspection).

## How to read it

1. Find the guard with the **lowest `ASR w/def %`** whose **`benign block %` is
   acceptable** (e.g. ≤ a few %). That's the deployable operating point.
2. Compare the internal-signal guards against `input_l2_guard`: does looking *inside*
   the model catch attacks the input-side guard misses (and vice-versa)? If their
   strengths differ, a combined guard is motivated.
3. `recall on success %` tells you what fraction of the *successful* jailbreaks the
   guard would have stopped — the number that matters most for ASR.
4. Check the per-variant table: `malicious` (literal specials) should be trivially
   caught by `input_l2_guard`; the interesting question is `mimicked`.

## Files

| file | role |
|---|---|
| `detectors.py` | per-prompt feature extraction, threshold/layer calibration, `InternalDetector` / `CascadeDetector` / `L2GuardDetector` |
| `run_defense_asr.py` | orchestrator: build prompts → generate+judge → calibrate → ASR report |

Relies on `experiments_hwichan/common.py` (forward capture, sink score, token
labelling), `src/` (mimicry, generation, refusal/Guard judge) and
`experiments_yeonseok/metabreak_l2_guard.py` (input-side baseline).
