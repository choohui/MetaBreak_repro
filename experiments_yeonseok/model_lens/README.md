# MetaBreak Model Lens Experiment

## Purpose

The previous guard detects MetaBreak TM-1 by inspecting user-token structure
before generation. This model-lens experiment asks a deeper question: after the
mimicked regular-token header enters the model, does it behave internally like a
literal Llama chat-template assistant header?

The goal is not to add another unconditional barrier. The goal is to identify
internal metrics that distinguish:

- literal special-token assistant-header injection,
- L2-mimicked regular-token assistant-header injection,
- benign safe prompts.

Those metrics can later be used to design a more robust defense that sees
through tokenizer retokenization artifacts.

## Metrics

The script computes these layer-wise signals.

1. **Hidden-state special convergence**

   For aligned injected spans, compare hidden states at the mimicked regular
   token positions against the corresponding literal special-token baseline
   positions:

   ```text
   mimicked regular eot-like token       vs literal <|eot_id|>
   mimicked regular start-header token  vs literal <|start_header_id|>
   mimicked regular end-header token    vs literal <|end_header_id|>
   ```

   Metrics:

   - cosine similarity to literal-special baseline hidden state,
   - L2 distance to literal-special baseline hidden state,
   - cosine margin to target special-token embedding minus own-token embedding.

2. **Post-span attention uptake**

   For each injected span, measure how much attention the immediately following
   tokens place on the 5-token header span:

   ```text
   token_i token_j assistant token_k \n\n next tokens...
   ```

   Metrics:

   - average attention mass from the next 1, 2, and 3 query tokens to the span,
   - same measurement for literal special-token baseline spans.

3. **Structural span counts**

   Record how many fake regular assistant-header spans and literal assistant
   header spans appear in each prompt. This separates structural attack
   evidence from ordinary text containing the word `assistant`.

## Expected Defensive Use

If mimicked spans converge toward literal special-token behavior in hidden space
or receive similar downstream attention, those signals can support a defense
that checks internal activation/attention signatures rather than relying only on
surface token IDs.

## Completed Runs

The current README-model runs are stored under:

```text
experiments_yeonseok/model_lens/results/lens_metabreak_mimicry_llama31_8b_n450_safe250
experiments_yeonseok/model_lens/results/lens_metabreak_negative_control_llama31_8b_n450_safe250
experiments_yeonseok/model_lens/results/lens_xstest_mimicry_llama31_8b_n200_safe250
experiments_yeonseok/model_lens/results/lens_xstest_negative_control_llama31_8b_n200_safe250
```

Summary:

- 450 MetaBreak mimicry prompts and 200 XSTest mimicry prompts analyzed.
- Matching non-L2 controls preserve the assistant skeleton but replace the
  special-token positions with `["The", " red", " blue"]`.
- End-header-like peak cosine separates mimicry from the non-L2 controls:
  MetaBreak mimicry `0.669839`, XSTest mimicry `0.684846`, controls
  `0.444456` and `0.470937`.
- Attention and eot-like cosine are not selective enough to use as standalone
  block signals.
- XSTest safe prompts had no user-controlled regular assistant-header-like
  spans in this lens check.

Detailed tables are in `RESULTS.md`.
