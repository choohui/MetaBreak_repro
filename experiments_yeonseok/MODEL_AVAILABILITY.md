# 모델 가용성 확인

README에 명시된 실험에 사용할 로컬 모델 상태를 기록한다.

## 탐색 대상

- README 기준 공격 모델: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- 선택적 판정 모델: `meta-llama/Llama-Guard-3-8B` 또는 호환 가능한 Llama Guard

## 확인 결과

- `meta-llama/Llama-3.1-8B-Instruct`는 `tokenlist.txt`의 토큰을 사용하여
  Hugging Face에서 다운로드 완료.
- `meta-llama/Llama-Guard-3-8B`도 동일한 토큰으로 다운로드 완료.
- 확인된 로컬 스냅샷 경로는 아래 파일에 기록되어 있다.

```text
experiments_yeonseok/results/readme_model_paths.json
```

확인된 경로:

```text
model_path=/gallery_tate/yeonseok.jung/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
guard_path=/gallery_tate/yeonseok.jung/.cache/huggingface/hub/models--meta-llama--Llama-Guard-3-8B/snapshots/7327bd9f6efbbe6101dc6cc4736302b3cbb6e425
```

## 토크나이저 확인

다운로드된 Llama-3.1 모델에 대해 재현에 필요한 토크나이저의 특수 토큰 범위 및 ID를 검증하였다.

```text
128009 -> <|eot_id|>
128006 -> <|start_header_id|>
128007 -> <|end_header_id|>
vocab_size=128000
len(tokenizer)=128256
```

README에 명시된 실험 결과는 다음 경로에 저장되어 있다.

```text
experiments_yeonseok/results/l2_guard_llama31_8b_n10
```
