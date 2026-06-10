# experiments_defense — §4 report (def_all)

4 defenses × 3 model(s) on MetaBreak TM-1 attacks + GSM8k(+mimicry header).

**ours** = diff-means token detection + ±1 drop (sanitize). **llama_guard / jbshield / guard_slm** = prompt-level detect→refuse.

### Attack success rate after defense  (lower = better)

baseline `asr_before` is the no-defense ASR on the held-out attack set.

| model | asr_before | ours | llama_guard | jbshield | guard_slm |
|---|---|---|---|---|---|
| llama | 0.830 | 0.556 | 0.385 | 0.000 | 0.000 |
| qwen | 0.844 | 0.652 | 0.378 | 0.674 | 0.000 |
| gemma | 0.659 | 0.281 | 0.370 | 0.000 | 0.000 |

### Block rate among originally-successful attacks  (higher = better)

fraction of baseline-successful attacks neutralised by the defense.

| model | ours | llama_guard | jbshield | guard_slm |
|---|---|---|---|---|
| llama | 0.420 | 0.536 | 1.000 | 1.000 |
| qwen | 0.263 | 0.553 | 0.202 | 1.000 |
| gemma | 0.629 | 0.438 | 1.000 | 1.000 |

### GSM8k (+mimicry header) accuracy  (higher = better; utility)

`plain` / `hdr_nodef` = no-defense accuracy without / with the header. Prompt-level refusers should drop here; `ours` should not.

| model | plain | hdr_nodef | ours | llama_guard | jbshield | guard_slm |
|---|---|---|---|---|---|---|
| llama | 0.700 | 0.700 | 0.360 | 0.700 | 0.020 | 0.000 |
| qwen | 0.440 | 0.480 | 0.120 | 0.480 | 0.480 | 0.000 |
| gemma | 0.700 | 0.600 | 0.580 | 0.600 | 0.000 | 0.000 |

### Benign false-positive rate  (lower = better)

benign prompts refused (prompt-level) — `ours` reports flag-rate instead.

| model | ours | llama_guard | jbshield | guard_slm |
|---|---|---|---|---|
| llama | 0.667 | 0.000 | 0.167 | 0.000 |
| qwen | 0.167 | 0.000 | 0.000 | 0.000 |
| gemma | 0.000 | 0.000 | 0.000 | 0.000 |
