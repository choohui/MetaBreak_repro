# repro_mb — MetaBreak TM-1 재현

MetaBreak (Wu et al., 2025) §5.3의 **TM-1 (Semantic Mimicry, 1-token replacement)**
공격을 HuggingFace Transformers 환경에서 end-to-end로 재현한다.

Llama-3.1 외에 Qwen2.5, Gemma-2, Phi-4도 `--model_type` 인자 하나로 지원한다.

## Pipeline

```
src/embedding.py  →  src/mimicry.py  →  src/attack.py  →  src/evaluate.py
  replacement.json   prompt_mimicked.jsonl  responses.jsonl  eval_report.json
```

1. **embedding** — 모델 입력 임베딩 테이블에서 각 특수 토큰(`<|eot_id|>` 등)의
   top-`k` L2-nearest 일반 토큰 후보를 수집하고, 결합 문자열이 동일한 N-token
   구조로 re-tokenize되는 조합 중 L2-sum 최소를 선택.
2. **mimicry** — `Q_TM-1_<Model>.txt`의 각 프롬프트에서 특수 토큰 문자열을
   best replacement 디코딩 결과로 치환.
3. **attack** — HF transformers로 victim 모델 로드, mimicked prompt를
   `apply_chat_template`으로 user role에 넣어 `model.generate`.
   `--also_baseline`으로 원본(특수 토큰 유지) 응답도 함께 수집 가능.
4. **evaluate** — (A) refusal-keyword matching (Zou et al. 2023, GCG 논문 표준 리스트),
   (B) `--guard_model` 지정 시 Llama Guard 3 분류기. 두 기준의 ASR 모두 보고.

## Quickstart

```bash
# Llama-3.1 end-to-end (10개 prompt, refusal keyword만)
python run.py \
    --model_type llama \
    --model /path/to/Llama-3.1-8B-Instruct \
    --n 10

# Llama Guard까지 포함, baseline 비교
python run.py \
    --model_type llama \
    --model /path/to/Llama-3.1-8B-Instruct \
    --guard_model /path/to/Llama-Guard-3-8B \
    --n 50 \
    --also_baseline

# Qwen2.5 예시
python run.py \
    --model_type qwen \
    --model /path/to/Qwen2.5-7B-Instruct \
    --n 10
```

### 개별 stage 실행 (다중 패밀리)

```bash
python -m src.embedding --model_type llama --model /path/to/Llama-3.1-8B-Instruct
python -m src.mimicry   --model_type llama --model /path/to/Llama-3.1-8B-Instruct --n 10
python -m src.attack    --model_type llama --model /path/to/Llama-3.1-8B-Instruct --also_baseline
python -m src.evaluate  --model_type llama --guard_model /path/to/Llama-Guard-3-8B
```

### Llama-only 레거시 entry point (backward compatible)

```bash
# --model_type llama 자동 주입. 인터페이스는 src/ 모듈과 동일.
python embedding.py --model /path/to/Llama-3.1-8B-Instruct --topk 200
python mimicry.py   --model /path/to/Llama-3.1-8B-Instruct --n 10
python attack.py    --model /path/to/Llama-3.1-8B-Instruct --also_baseline
python evaluate.py  --guard_model /path/to/Llama-Guard-3-8B
```

## CLI 주요 인자

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--model_type` | `llama` | 모델 패밀리 슬러그. llama / qwen / gemma / phi 또는 미등록 패밀리(auto-detect). |
| `--model` | (필수) | victim 모델 로컬 HF 디렉토리. |
| `--guard_model` | None | Llama-Guard-3-8B 로컬 HF 디렉토리 (optional). |
| `--n` | 10 | 사용할 프롬프트 수. |
| `--topk` | 200 | embedding search 후보 풀 크기. |
| `--also_baseline` | False | 원본 특수 토큰 prompt로도 generate. |
| `--temperature` | 0.0 | 0.0 = greedy decoding. |
| `--dtype` | bfloat16 | bfloat16 / float16 / float32. |
| `--device` | auto | cuda / cpu. |
| `--skip_embedding` | False | 기존 `replacement.json` 재사용 (re-run 빠름). |
| `--out_dir` | `results/<model_type>/` | 모든 중간/최종 파일 저장 경로. |

## 출력물 (`results/<model_type>/` 기본)

| 파일 | 내용 |
| --- | --- |
| `replacement.json` | best tuple IDs + L2-sum + 메타 |
| `prompt_mimicked.jsonl` | 원본/치환된 prompt + token ids |
| `responses.jsonl` | mimicked/baseline response 텍스트 + 통계 |
| `eval_per_item.jsonl` | 항목별 refusal/guard 판정 |
| `eval_report.json` | 전체 ASR 요약 |

## 디렉토리 구조

```
repro_mb/
├── run.py                    # 단일 명령 orchestrator (src/ 모듈 직접 호출)
├── src/
│   ├── model_configs.py      # 패밀리별 assistant_header + auto-detect
│   ├── embedding.py          # Stage 1: L2 embedding search
│   ├── mimicry.py            # Stage 2: special token → regular token 치환
│   ├── attack.py             # Stage 3: LLM generate
│   ├── evaluate.py           # Stage 4: ASR 계산
│   └── build_prompts.py      # Q.txt → Q_TM-1_<Model>.txt 생성 (전처리)
├── embedding.py              # Llama-only legacy wrapper → src.embedding
├── mimicry.py                # Llama-only legacy wrapper → src.mimicry
├── attack.py                 # Llama-only legacy wrapper → src.attack
├── evaluate.py               # Llama-only legacy wrapper → src.evaluate
├── prompts/
│   ├── Q.txt                 # 원본 450개 쿼리 (MetaBreak 저장소에서 복사)
│   └── Q_TM-1_Llama.txt      # build_prompts로 생성된 Llama용 프롬프트
├── results/<model_type>/     # run.py 출력 디렉토리
└── experiments_yeonseok/     # L2-guard 방어 실험 (별도 README 참조)
```

## 의존성

```
pip install -r requirements.txt
```

- `torch`, `transformers`, `accelerate`, `numpy`, `safetensors`, `sentencepiece`
- Llama Guard 사용 시 `meta-llama/Llama-Guard-3-8B` 로컬 다운로드 필요.

## 새 모델 패밀리 추가

`src/model_configs.py`의 `KNOWN_HEADERS` 딕셔너리에 `assistant_header`와 `user_header`를
추가하고, `prompts/Q_TM-1_<Family>.txt`를 `build_prompts.py`로 생성한다.
등록하지 않은 패밀리는 HF chat template에서 auto-detect된다.

## 알려진 한계 / 확장 포인트

- TM-1 외 변형(TM-2, role-swap 등)은 미구현. `src/mimicry.py` 치환 로직과
  `prompts/Q.txt` 기반으로 확장 가능.
- 평가 judge 2종(refusal keyword, Llama Guard)만 포함. GPT-4 judge 등은 미포함.
- batched generation 없음 — 1 prompt씩 순차 처리. vLLM/accelerate batching으로 가속 가능.
- tokenizer 버전이 달라 HF tokenizer의 special-token 파싱 동작이 바뀌면
  `--also_baseline` 결과가 달라질 수 있음. `add_special_tokens` 옵션 점검 요.

## References

- Wu et al., *MetaBreak: Jailbreaking Online LLM Services via Special Token
  Manipulation*, 2025.
- Zou et al., *Universal and Transferable Adversarial Attacks on Aligned
  LLMs*, 2023. (refusal-string judge)
- Inan et al., *Llama Guard: LLM-based Input-Output Safeguard for
  Human-AI Conversations*, 2023. (safety classifier)
