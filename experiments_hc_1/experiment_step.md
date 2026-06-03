# experiment_step.md — 실험 단계 분해 (선택형)

`Main.md`의 전체 연구를 실행 가능한 step으로 분해한 문서입니다. 각 step은 독립적으로
실행하거나 `run_all.py --stages ...`로 묶어 실행할 수 있습니다. ✅ = 구현 완료,
🔜 = 명세만 있고 코드는 추후 생성(이 문서를 기준으로 선택 생성).

---

## Step 0 — 토큰 임베딩 분석 (Main.md §1) ✅
**질문:** special token과 regular token이 입력 임베딩 테이블에서 cosine/L2-norm 경향으로
구분되는가?
- 코드: `00_embedding_analysis.py`
- 산출물: `embedding_analysis.{json,md}` (norm·cosine·중심점 거리 통계, special-vs-regular
  분리 AUC)
- 예상 결론: 분리되지 않음(AUC≈0.5) → 내부 표현을 봐야 함.
- 실행: `python 00_embedding_analysis.py --model <LLAMA>`

## Step 1 — 7종류 prompt 생성 + ASR labeling (Main.md §2.1) ✅
**작업:** 7 타입(A~G) prompt를 생성·저장하고, 공격 변형을 모델에 입력해 공격 성공 여부(ASR)를
prompt별로 라벨링.
- 코드: `01_build_prompts.py`(생성), `02_run_asr.py`(생성+판정)
- 산출물: `prompts.jsonl`, `asr.{jsonl,csv}`, `asr_summary.json`
- ASR 대상: B(malicious_mimicry)·D(malicious_special)·F(positioned_regular)
- 판정: refusal-keyword(`src.evaluate.matches_refusal`) + 선택적 Llama Guard
- 실행: `python 01_build_prompts.py --n 50` → `python 02_run_asr.py --model <LLAMA>`

## Step 2 — 내부 표현 추출 + 분석 (Main.md §2.2 / §2.3) ✅
**작업:** prompt별 1회 forward로 5신호 + hidden cube 저장, 그 위에서 logistic regression과
cosine similarity, single-threshold 방어 가능성 분석.
- 코드: `03_extract_representations.py`, `04_analyze_cosine_logreg.py`, `05_threshold_defense.py`
- 산출물: `tokens.jsonl`, `features.npz`, `pos{k}/representation_metrics.*`,
  `cosine_pairs.json`, `threshold_defense.*`, `threshold_per_type.csv`, `threshold_asr.json`
- 분석:
  - logreg: 전체 hidden 벡터로 공격 vs 정상 레이어별 분리 (ROC-AUC, balanced acc, 5-fold CV)
  - cosine: (A,B)(A,D)(A,G)(B,C)(B,D)(B,F) 레이어별, 중심점 + prompt별 분포 두 방식
  - threshold: 5신호 × 레이어 ROC-AUC/Youden/TPR@FPR, 타입별 분해 + ASR 기반
- 실행: `python run_all.py --model <LLAMA> --stages 03,04,05`

## Step 3 — sink 범위 축소 분석 (Main.md §3) ✅
**작업:** sink score로 봐야 하는 token 범위를 최대한 축소한 뒤, 축소된 집합에서 신호를
threshold로 보는 방법의 TPR/FPR + ASR 기반 분석을 수행. **Step 2와 함께 실행됨**(동일 산출물
재사용).
- 코드: `06_sink_range.py`
- 산출물: `pos{k}/sink_range_report.{json,md}`
- 축소 모드:
  - `header_slots`(기본): 공격 헤더 슬롯 위치 {B,D,F}만 → 공격(B,D) vs 슬롯 내 benign(F)
  - `topk`: prompt별 sink 상위 k개만 → 공격 vs 가장 sink가 큰 benign 토큰
- 실행: `python 06_sink_range.py --sink_range_mode header_slots`
  또는 `--sink_range_mode topk --sink_range_topk 8`

---

## Step 4 — cascade 방어기법 생성 + 평가 (Main.md §4) 🔜 (미구현 명세)
**목표:** 지금까지의 분석을 바탕으로 실제 방어기법을 만들고, 실제 공격을 얼마나 막는지 평가.

> 이 step은 Step 0~3의 분석 결과(어느 신호·레이어·threshold가 가장 잘 분리하는지)를 확인한
> 뒤 생성하는 것이 정확합니다. 따라서 코드는 아직 만들지 않고, 아래 설계만 고정해 둡니다.

**설계 (의도한 `07_cascade_defense.py`)**
- 입력: Step 2/3의 산출물(`tokens.jsonl`, `features.npz`, `asr.jsonl`) — **모델 재실행 불필요**
  (사후 분석). 필요 시 held-out 평가를 위해 prompt를 train/eval로 grouped split.
- 1차 거름 (cheap gate): **sink score** 임계값으로 의심 토큰만 통과
  (recall 목표, 예: 0.99에서 `t1 = quantile(공격 sink, 1-recall)`).
- 2차 거름 (strong filter): 1차 통과분에 대해 가장 분리력이 높았던 신호
  (예: `hidden_norm`/`value_norm`/`output_norm`/`cos_to_ref`) 임계값 적용.
- 평가 지표:
  - 전체 풀에서의 TPR@FPR{1%,5%} (1차+2차 결합)
  - **ASR 기반 차단율**: 실제로 성공한 공격(B∪D, ASR=success)을 얼마나 차단하는가
  - 정상(C∪E∪F∪G) 오차단율(false block rate), 1차/2차 신호 간 상관(Spearman)
- 산출물(예정): `cascade_report_pos{k}.{json,md}` — (1차 신호 × 2차 신호 × recall × FPR) 격자.
- 확장 옵션: held-out 평가(`cascade_tier2`), 입력단 L2-guard 비교 등.

**생성 트리거:** Step 0~3 결과를 검토 후, "Step 4 / §4 / cascade 방어 구현"을 요청하면
위 설계대로 `07_cascade_defense.py`를 생성합니다.
