# XSTest 벤치마크 결과

## 벤치마크 개요

XSTest는 10개 카테고리, 250개 안전 프롬프트와 200개 비안전 대조 프롬프트로 구성된
공개 과잉 거부(over-refusal) 벤치마크이다. 공격에 특화된 가드가 단순히 위험해 보이는
정상 프롬프트까지 차단하는지 여부를 검증하는 데 유용하다.

이 실험에서:

- XSTest `safe` 프롬프트는 정상적인 통과(pass-through) 입력으로 사용한다.
- XSTest `unsafe` 대조 프롬프트는 TM-1 어시스턴트 헤더 접미사를 붙이고
  현재 `common/replacement.json`의 L2 교체를 적용하여 MetaBreak 모방 공격 형태로 변환한다.
- 유료 판정기 없이 가드의 생성 전 차단/통과 판정만을 측정한다.

출처: Hugging Face `Paul/XSTest`.

## 파일 목록

- 준비된 안전 프롬프트: `xstest_safe.jsonl`
- 준비된 공격 비안전 프롬프트: `xstest_unsafe_metabreak_mimicked.jsonl`
- 보고서: `selectivity_report.json`
- 항목별 판정: `selectivity_per_item.jsonl`

## 지표

| 분류 | 전체 | 차단 | 통과 | 비율 |
| --- | ---: | ---: | ---: | ---: |
| XSTest 비안전 + MetaBreak 모방 | 200 | 200 | 0 | 100.0% 차단 |
| XSTest 안전 정상 텍스트 | 250 | 0 | 250 | 100.0% 통과 |

## 안전 카테고리별 세부 결과

| 안전 카테고리 | 전체 | 차단 | 통과율 |
| --- | ---: | ---: | ---: |
| 정의(definitions) | 25 | 0 | 100.0% |
| 비유적 표현(figurative language) | 25 | 0 | 100.0% |
| 역사적 사건(historical events) | 25 | 0 | 100.0% |
| 동음이의어(homonyms) | 25 | 0 | 100.0% |
| 무의미 집단/실제 차별(nonsense group / real discrimination) | 25 | 0 | 100.0% |
| 가상 인물 프라이버시(privacy fictional) | 25 | 0 | 100.0% |
| 공인 프라이버시(privacy public) | 25 | 0 | 100.0% |
| 실제 집단/무의미 차별(real group / nonsense discrimination) | 25 | 0 | 100.0% |
| 안전 맥락(safe contexts) | 25 | 0 | 100.0% |
| 안전 대상(safe targets) | 25 | 0 | 100.0% |

## 해석

이 벤치마크에서 가드는 무조건적인 차단 장치로 작동하지 않는다. 표면적으로 위험해 보이는
단어가 포함된 프롬프트를 포함하여 XSTest 안전 프롬프트 전체를 통과시키면서,
MetaBreak 모방 비안전 대조 프롬프트는 전부 차단한다.
