# repro_mb — MetaBreak TM-1 reproduction

MetaBreak (Wu et al., 2025) §5.3의 **TM-1 (Semantic Mimicry, 1-token replacement)**
공격을 Llama-3.1-8B-Instruct 환경에서 end-to-end로 재현한다.

## Pipeline

```
embedding.py  ─►  mimicry.py  ─►  attack.py  ─►  evaluate.py
   (replace.json)   (prompt.jsonl) (resp.jsonl)   (report.json)
```

1. `embedding.py` — Llama-3.1 chat-template special token 3개
   (`<|eot_id|>`, `<|start_header_id|>`, `<|end_header_id|>`) 각각에 대해
   L2 distance 기준 top-`k` 후보를 뽑고, 3-tuple `(i, j, k)` brute-force로
   `decode(i)+decode(j)+'assistant'+decode(k)+'\n\n'` 가 5-token으로
   re-tokenize 되는 조합 중 최소 L2-sum을 선택.
2. `mimicry.py` — `MetaBreak/prompts/Q_TM-1_Llama.txt`의 각 prompt에서
   special-token 문자열을 best replacement decoded string으로 치환.
3. `attack.py` — HF transformers로 Llama-3.1-8B-Instruct 로드, mimicked prompt를
   `apply_chat_template`으로 user role에 넣어 `model.generate`. baseline (원본
   special-token 유지) 도 `--also_baseline` 으로 함께 측정 가능.
4. `evaluate.py` —
   (A) refusal-keyword matching (Zou et al. 2023, GCG 논문의 표준 list),
   (B) `--guard_model` 지정 시 Llama Guard 3 분류기로 unsafe 판정.
   ASR(attack success rate) 두 가지 모두 보고.

## Quickstart

```bash
# end-to-end (10개 prompt, refusal keyword 만)
python run.py --model /path/to/Llama-3.1-8B-Instruct --n 10

# Llama Guard 까지 같이 돌리고, baseline 비교 포함
python run.py \
    --model /path/to/Llama-3.1-8B-Instruct \
    --guard_model /path/to/Llama-Guard-3-8B \
    --n 50 \
    --also_baseline
```

individual stage 도 가능:

```bash
python embedding.py --model /path/to/Llama-3.1-8B-Instruct --topk 200
python mimicry.py   --model /path/to/Llama-3.1-8B-Instruct --n 10
python attack.py    --model /path/to/Llama-3.1-8B-Instruct --also_baseline
python evaluate.py  --guard_model /path/to/Llama-Guard-3-8B
```

## CLI 주요 인자

- `--model` : Llama-3.1-8B-Instruct local HF 디렉토리.
- `--guard_model` : Llama-Guard-3-8B local HF 디렉토리 (optional).
- `--n` : Q_TM-1_Llama.txt에서 앞 N개 prompt만 사용 (default 10).
- `--topk` : embedding search 후보 풀 크기 (default 200).
- `--also_baseline` : 원본(특수 token 그대로) prompt로도 generate.
- `--temperature` : 0.0 = greedy decoding (default).
- `--dtype` : bfloat16 / float16 / float32.
- `--skip_embedding` : 기존 `replacement.json` 재사용 (re-run 빠름).

## 출력물 (`repro_mb_out/` 기본)

| 파일 | 내용 |
| --- | --- |
| `replacement.json`     | best (i, j, k) triple + L2-sum + 메타 |
| `prompt_mimicked.jsonl`| 원본/치환된 prompt + token id |
| `responses.jsonl`      | mimicked/baseline response 텍스트 + 통계 |
| `eval_per_item.jsonl`  | 항목별 refusal/guard 판정 |
| `eval_report.json`     | 전체 ASR 요약 |

## 의존성

- `torch`, `transformers` (HF), `numpy`
- Llama Guard 까지 쓰려면 `meta-llama/Llama-Guard-3-8B` 가 로컬에 다운로드 되어
  있어야 한다.

## 알려진 한계 / 추가하면 좋은 것

- TM-1 외 다른 변형 (TM-2, role-swap 등) 은 미구현. 필요 시 `mimicry.py`의
  치환 로직과 `Q.txt` 사용으로 확장.
- 평가는 두 judge만. GPT-4 judge / custom rubric 등은 미포함.
- batched generation 없음 — 8B 1 prompt씩 순차 처리 (450 prompt = 수십 분 단위).
  vLLM/accelerate batching 으로 가속 여지.
- baseline (특수 token 유지) 의 효과는 `apply_chat_template` 동작에 의존:
  HF tokenizer가 user content 내 `<|eot_id|>` 같은 문자열을 default로 special
  token ID로 파싱하기 때문에, 원본 MetaBreak Ollama chat 경로와 동일한 효과가
  난다. tokenizer 버전이 달라 동작이 바뀌면 명시적 `add_special_tokens`
  옵션을 점검할 것.

## References

- Wu et al., *MetaBreak: Jailbreaking Online LLM Services via Special Token
  Manipulation*, 2025. (project repo: `../MetaBreak/`)
- Zou et al., *Universal and Transferable Adversarial Attacks on Aligned
  LLMs*, 2023. (refusal-string judge)
- Inan et al., *Llama Guard: LLM-based Input-Output Safeguard for
  Human-AI Conversations*, 2023. (safety classifier)
