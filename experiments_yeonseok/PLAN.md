# MetaBreak L2 Mimicry Defense Plan

## Repository reading

The repository is a compact reproduction of MetaBreak TM-1 for Llama-3.x chat
templates.

- `embedding.py` loads the model input embedding table and searches regular
  token triples that are close in L2 distance to the Llama chat special tokens
  `<|eot_id|>`, `<|start_header_id|>`, and `<|end_header_id|>`.
- `mimicry.py` rewrites `Q_TM-1_Llama.txt` by replacing those literal special
  token strings with the decoded regular-token triple from `replacement.json`.
- `attack.py` wraps the resulting user content with `tokenizer.apply_chat_template`
  and generates responses.
- `evaluate.py` measures attack success by refusal-keyword absence, optionally
  with Llama Guard if a local guard model is provided.
- `run.py` orchestrates the original four-stage no-defense attack pipeline.

The attack works because naive string-level special-token sanitization is no
longer enough after `mimicry.py`: the prompt contains regular tokens, but those
tokens are embedding-nearest neighbors of the special tokens and are arranged as
the Llama assistant-header pattern:

```text
near(<|eot_id|>) near(<|start_header_id|>) assistant near(<|end_header_id|>) \n\n
```

## Defense method

I implemented a token-level L2 structural guard:

1. Load the same tokenizer and input embedding table used by the model.
2. Calibrate an L2 threshold for each target special token by taking the
   `neighbor_rank` nearest regular-token distance to that special token.
3. Tokenize user content before chat-template wrapping.
4. Block if literal chat special IDs appear in user content.
5. Block if any 5-token span has this structure:

```text
token_i token_j assistant token_k \n\n
```

where `token_i`, `token_j`, and `token_k` are regular tokens but their L2
distances to `<|eot_id|>`, `<|start_header_id|>`, and `<|end_header_id|>` are
within the calibrated thresholds.
6. Also block repeated regular-token assistant-header skeletons in user content.
   This catches retokenization artifacts such as decoded replacement characters
   that are not themselves the final token IDs used in the prompt, while still
   requiring the suspicious chat-control structure to appear repeatedly.

This targets the attack mechanism rather than the exact replacement strings. A
different L2-neighbor triple generated with the same attack family should still
match if it lies inside the calibrated nearest-neighbor bands.

## Experiment plan

Use the README-matched Llama-3.1-8B-Instruct snapshot and Llama-Guard-3-8B
snapshot downloaded from Hugging Face using `tokenlist.txt`. The resolved paths
are recorded in `experiments_yeonseok/results/readme_model_paths.json` and summarized in
`MODEL_AVAILABILITY.md`.

Run:

```bash
sr 1 48 python experiments_yeonseok/run_defense_experiment.py \
  --model /gallery_tate/yeonseok.jung/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --guard_model /gallery_tate/yeonseok.jung/.cache/huggingface/hub/models--meta-llama--Llama-Guard-3-8B/snapshots/7327bd9f6efbbe6101dc6cc4736302b3cbb6e425 \
  --out_dir experiments_yeonseok/results/l2_guard_llama31_8b_n10 \
  --n 10 \
  --topk 200 \
  --neighbor_rank 256 \
  --structural_min_spans 2 \
  --max_new_tokens 256 \
  --also_baseline
```

The runner saves:

- `common/replacement.json`
- `common/prompt_mimicked.jsonl`
- `no_defense/responses.jsonl`
- `no_defense/eval_report.json`
- `defended/responses.jsonl`
- `defended/eval_report.json`
- `summary.json`
- `RESULTS.md`

Additional benchmark:

- Download `Paul/XSTest` from Hugging Face.
- Use the 250 `safe` prompts to measure benign pass-through.
- Convert the 200 `unsafe` contrast prompts into MetaBreak-mimicked attacks by
  appending the TM-1 suffix and applying `replacement.json`.
- Measure guard decisions only, without a paid judge.
- Save results under `experiments_yeonseok/results/l2_guard_llama31_8b_n10/xstest/`.

Primary success criteria:

- The guard detects the mimicked structural spans in attacked prompts.
- Defended ASR by refusal-keyword judge is lower than no-defense ASR.
- Baseline literal-special-token prompts are also blocked by the literal special
  token branch.
- The guard passes a broad safe benchmark rather than becoming an unconditional
  barrier.
