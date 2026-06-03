# experiments_hwichan — Internal representations of mimicry vs. special tokens

Defense research for MetaBreak's **Semantic Mimicry Attack (TM-1)**. We measure
whether the LLM's *internal representation* distinguishes four kinds of tokens,
and whether that difference supports a **threshold-based defense**.

## The four token categories

| code | name | where it comes from |
|------|------|---------------------|
| **A** | mimicry regular token | regular token that L2-replaced a special token in a mimicked prompt (e.g. `ujících`, `?`) |
| **B** | malicious special token | literal special token the attacker injected into the user content (TM-1 assistant header) |
| **C** | benign special token | a special token appearing in a benign, non-attack context (single/isolated) |
| **D** | system special token | the real special tokens the chat template inserts (BOS, user header, the final generation-prompt assistant header) |
| **E** | ordinary regular token | plain text tokens (negative baseline) |

Defense framing: **positive = A ∪ B (attack)**, **negative = C ∪ E (benign)**.
`D` is the reference "genuine special token" representation that A is suspected
to mimic.

## Two experiments

- **Experiment 1** — uses `prompts/Q_TM-1_Llama.txt` as-is. The attack headers sit
  at the prompt tail, so A/B tokens cluster in the last positions.
- **Experiment 2** — injects the attack payload (full header, mimicked header, or a
  single special token) into clean `prompts/Q.txt` carriers at **start / middle /
  end / scattered** positions, removing the "last-few-tokens" confound.

## Signals measured (per token, per layer, at the token position **and** the immediately following position `pos_offset=1`)

- hidden-state vector + L2 norm (massive-activation signal)
- **attention sink score** — Def. 2.1 of *Attention Sinks as Internal Signals*
  (Gu et al. 2025): `s_i^(l,h) = (1/(T−i)) · Σ_{u≥i} A_{u,i}^(l,h)`, averaged over heads
- value-vector norm `‖V‖` (the paper's *computationally-active sink* signal)
- attention-output norm `‖O‖`
- `cos_to_D` — cosine of the token's hidden state to the system-special (D)
  centroid at that layer ("does this token look like a real special token inside?")

## Pipeline / files

| file | role |
|------|------|
| `common.py` | model loading (eager attn), `forward_capture` (hidden states + attentions + `‖V‖`/`‖O‖` hooks), `sink_scores`, A/B/C/D token labelling |
| `benign_special_prompts.jsonl` | category-C prompts (special token used benignly) |
| `build_category_prompts.py` | builds `exp1_prompts.jsonl` / `exp2_prompts.jsonl` |
| `extract_representations.py` | forward pass → `tokens.jsonl` (per-layer scalars) + `features.npz` (hidden cube) |
| `analyze_representations.py` | per-layer norms, centroid cos/L2 (A-D, B-D, A-B, …), A→D convergence, separability probe, PCA coords |
| `defense_thresholds.py` | per-(feature, layer) ROC-AUC, Youden threshold, TPR@FPR for the threshold defense |
| `run_experiment.py` | orchestrates build→extract→analyze→defense, loading the model once |

## Run (GPU server, Llama-3.1-8B-Instruct)

```bash
cd repro_mb
pip install -r requirements.txt -r experiments_hwichan/requirements_extra.txt

python experiments_hwichan/run_experiment.py \
  --model <local-llama31-8b-snapshot> \
  --replacement experiments_yeonseok/results/l2_guard_llama31_8b_n450/common/replacement.json \
  --exp both --n 50
```

Smoke test first with `--n 5` to verify tensor shapes and hook behaviour, then
scale up. `--model` defaults to the README id `meta-llama/Llama-3.1-8B-Instruct`
is **not** assumed — always pass your local snapshot path.

Stages can also be run standalone (each reads/writes one `--out_dir`):
`build_category_prompts.py`, `extract_representations.py`,
`analyze_representations.py`, `defense_thresholds.py`.

## Outputs (under `results/exp{1,2}_llama31_8b/`)

- `exp{1,2}_prompts.jsonl`, `tokens.jsonl`, `features.npz`, `extract_summary.json`
- `pos{0,1}/representation_metrics.{json,csv}`, `pos{0,1}/pca_coords.npz`
- `pos{0,1}/defense_report.{json,md}`

## How to read the results

1. `extract_summary.json` → `category_census` confirms each category has enough
   tokens; A tokens should carry no special-token ids (the mimicry is regular text).
2. `representation_metrics.json`:
   - `A_to_D_convergence` — if `cos(A, D)` rises with depth, the mimicked regular
     tokens are *becoming* system-special-like internally (why the attack works).
   - `cos__*`, `l2__*` and `*_mean_norm` — where A/B diverge from C and from D.
   - `attack_vs_negative_probe` + `best_probe_layer` — the layer where attack is
     most linearly separable from benign.
3. `defense_report.md` — the headline: which `(feature, layer)` separates attack
   from benign with the highest AUC, and whether the attention-only `sink` score
   beats the hidden-state features (`hidden_norm`, `value_norm`, `cos_to_D`).
   `TPR@FPR=1%/5%` indicates a deployable operating point for a threshold guard.

## Relationship to `experiments_yeonseok`

`experiments_yeonseok` detects mimicry at the **input/tokenization boundary**
(L2 embedding distance). This study looks for a complementary signal **inside the
model** (sink score + hidden-state geometry), which could catch variants that the
input-side guard misses and motivates an internal-state threshold defense.
