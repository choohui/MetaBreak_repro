# XSTest Benchmark Results

## Benchmark

XSTest is a public over-refusal benchmark with 250 safe prompts across 10
categories and 200 unsafe contrast prompts. It is useful here because it tests
whether an attack-specific guard blocks benign prompts that merely look risky.

For this experiment:

- XSTest `safe` prompts are used as normal benign pass-through inputs.
- XSTest `unsafe` contrast prompts are converted into MetaBreak-style attacked
  prompts by appending the TM-1 assistant-header suffix and applying the current
  L2 replacement from `common/replacement.json`.
- No paid judge is used. This benchmark measures the guard's pre-generation
  block/pass decision.

Source: `Paul/XSTest` on Hugging Face.

## Files

- Prepared safe prompts: `xstest_safe.jsonl`
- Prepared attacked unsafe prompts: `xstest_unsafe_metabreak_mimicked.jsonl`
- Report: `selectivity_report.json`
- Per-item decisions: `selectivity_per_item.jsonl`

## Metrics

| split | total | blocked | passed | rate |
| --- | ---: | ---: | ---: | ---: |
| XSTest unsafe + MetaBreak mimicry | 200 | 200 | 0 | 100.0% block |
| XSTest safe normal text | 250 | 0 | 250 | 100.0% pass |

## Safe Category Breakdown

| safe category | total | blocked | pass rate |
| --- | ---: | ---: | ---: |
| definitions | 25 | 0 | 100.0% |
| figurative language | 25 | 0 | 100.0% |
| historical events | 25 | 0 | 100.0% |
| homonyms | 25 | 0 | 100.0% |
| nonsense group / real discrimination | 25 | 0 | 100.0% |
| privacy fictional | 25 | 0 | 100.0% |
| privacy public | 25 | 0 | 100.0% |
| real group / nonsense discrimination | 25 | 0 | 100.0% |
| safe contexts | 25 | 0 | 100.0% |
| safe targets | 25 | 0 | 100.0% |

## Interpretation

The guard is not acting as an unconditional barrier on this benchmark: it passes
all XSTest safe prompts, including prompts with superficially risky words, while
blocking all MetaBreak-mimicked unsafe contrast prompts.
