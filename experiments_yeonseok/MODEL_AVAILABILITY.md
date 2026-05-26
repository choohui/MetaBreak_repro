# Model Availability Check

This note records the local model state for the README-matched experiment.

## Searched targets

- README target attack model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Optional judge model: `meta-llama/Llama-Guard-3-8B` or compatible Llama Guard

## Current findings

- `meta-llama/Llama-3.1-8B-Instruct` was downloaded from Hugging Face with the
  token in `tokenlist.txt`.
- `meta-llama/Llama-Guard-3-8B` was also downloaded from Hugging Face with the
  same token.
- The resolved local snapshot paths are stored in:

```text
experiments_yeonseok/results/readme_model_paths.json
```

Resolved paths:

```text
model_path=/gallery_tate/yeonseok.jung/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
guard_path=/gallery_tate/yeonseok.jung/.cache/huggingface/hub/models--meta-llama--Llama-Guard-3-8B/snapshots/7327bd9f6efbbe6101dc6cc4736302b3cbb6e425
```

## Tokenizer check

The tokenizer special-token range and IDs required by the reproduction were
verified for the downloaded Llama-3.1 model:

```text
128009 -> <|eot_id|>
128006 -> <|start_header_id|>
128007 -> <|end_header_id|>
vocab_size=128000
len(tokenizer)=128256
```

The README-matched experiment result is stored under:

```text
experiments_yeonseok/results/l2_guard_llama31_8b_n10
```
