# experiment_hc_2 분석 리포트 — sink 1차 필터 + threshold 2차 cascade 방어

> **상태**: 📝 골격(skeleton). `< >`로 표시된 수치/판단은 **실제 모델 실행 후**
> 산출 JSON/CSV에서 채운다. 실행: `python -m experiments_hc_2.run_all --model <...>
> --n 150 --asr_judge both` → `python -m experiments_hc_2.result_report.make_figures`.
>
> **대상 모델**: Llama-3.1-8B-Instruct · **분석 산출물**: `results/hc2_llama31_8b/`
> **hc_1 대비 보완**: C 통합 · type 균형(raw+balanced) · prompt-level GroupKFold ·
> Llama-Guard ASR 옵션 · **§3 sink-filter sweep + §4 cascade 방어 구현·평가**.

> **방법론적 타당성 안전장치 (review 반영)**
> - **§2(probe/threshold)는 balanced 부분집합**, **§3/§4(sink 게이트)는 raw 전체
>   토큰셋**을 사용 — 게이트의 "본문(G) 제거" 효과를 균형화로 미리 없애지 않도록
>   stage 03이 full set + `balanced_row_ids` 인덱스를 함께 저장.
> - **§4 cascade는 prompt 단위 hold-out**: threshold와 (signal,layer) 선택은
>   **train**에서, 모든 차단율/FPR/ASR는 **held-out test**에서 측정(`eval_mode`).
> - **reference A(템플릿 special)는 게이트 후보에서 제외**(H1) — 게이트 예산 점유 방지.
> - **probe**: naive(per-token) vs **grouped(prompt-level)** AUC 병기, best layer는
>   grouped 기준.
> - `gate_only`의 C/E "FPR"은 게이트 생존율(coverage)이지 detector FPR이 아님(주의).

---

## 0. 한 페이지 요약 (실행 후 작성)

이 실험의 최종 질문: **"semantic-mimicry 공격을 내부 신호로 어떻게 방어해야 하며,
sink 1차 거름 + threshold 2차 거름의 cascade가 실제로 효과적인가?"**

- (§1) 임베딩만으로는 못 가른다 — `< by_l2_norm AUC = ___ 는 reserved-token 착시 >`.
- (§2) 내부표현에서 공격은 또렷하다 — 전체 hidden probe AUC `< naive ___ / grouped ___ >`.
  GroupKFold로 누수 보정 후에도 `< 견고/하락 >`.
- (§2) 단일 threshold는 가능하나 "special 탐지기" 함정 — value_norm/cos_to_ref가
  정상 special(E)을 `< ___ >` 비율로 오탐.
- **(§3) sink 1차 필터가 신호를 선명하게 한다** — keep `<__>%`로 줄이면 best 신호
  AUROC `<0.xx → 0.xx>`, E·C FPR `<__ → __>`, 공격 recall `<__ 유지>`.
- **(§4) cascade가 실제로 막는다** — 공격 차단율 B `<__>`/D `<__>`, benign FPR
  C `<__>`/E `<__>`, **ASR `<before → after>`**. 1-stage·gate-only 대비 `<우위>`.

---

## 1. §1 — 임베딩 단계 (stage 00)

`embedding_analysis.json`. special vs regular 분리 `< AUC=___ >` → reserved 토큰
착시 여부 판단. **결론: 내부표현을 봐야 한다.** (Fig 1)

## 2. §2 — 내부표현·신호 (stages 01–05)

### 2.1 데이터 규모: raw vs balanced (stage 03)
`extract_summary.json`. **hc_1의 핵심 결함(C=0)을 해결** — C가 census에 정상 포함.

| 타입 | raw_census | census(balanced) |
|---|---|---|
| A | `<__>` | `<__>` |
| B | `<__>` | `<__>` |
| C | `<__>` (≠0) | `<__>` |
| D/E/F/G | `<__>` | `<__>` |

`cap_mode=<balanced>`, `cap_applied=<__>`. 균형 데이터셋을 주 분석에 사용.

### 2.2 ASR (stage 02) — keyword vs guard
`asr_summary.json`. D `< __% >` · B `< __% >` · F(정상 통제) keyword `< __% >`
vs guard `< __% >`. **F 위양성이 guard에서 사라지는가**가 판정 신뢰도의 핵심. (Fig 8)

### 2.3 probe·cosine·single-threshold (stages 04–05)
- 전체 hidden probe: naive `< __ >` vs **grouped(GroupKFold) `< __ >`** (Fig 4).
  누수 보정 후에도 분리력 `< 견고/감소 >`.
- cosine: cos(B,D) `< 0.0x→0.xx 수렴 >`, **cos(B,C) `< __ >`** (이제 측정 가능). (Fig 3)
- 단일 threshold best-layer AUROC (Fig 5) + per-type flagged (Fig 6):

| 신호 | per-type AUC | E flagged | C flagged | 성격 |
|---|---|---|---|---|
| hidden_norm | `<__>` | `<__>` | `<__>` | `<맥락/특수>` |
| sink | `<__>` | `<__>` | `<__>` | `<__>` |
| value_norm | `<__>` | `<__>` | `<__>` | `<__>` |
| output_norm | `<__>` | `<__>` | `<__>` | `<__>` |
| cos_to_ref | `<__>` | `<__>` | `<__>` | `<__>` |

`operating_points.json`: best 신호=`<__>`@layer`<__>`, FPR1% threshold=`<__>`.

---

## 3. §3 — sink 1차 필터링이 효과적인가 (stage 06) ★

`pos0/sink_filter_report.json`. keep-% sweep으로 **"볼 토큰 범위를 줄이면 신호가
선명해지는가"**에 답한다. (Fig 7a/7b)

| keep_% | n_kept | 비율 | B recall | D recall | best AUROC | E FPR | C FPR |
|---|---|---|---|---|---|---|---|
| 100(무게이트) | `<__>` | 1.0 | – | – | `<__>` | `<__>` | `<__>` |
| 50 | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` |
| 30 | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` |
| 20 | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` |
| 10 | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` |
| 5  | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` |

**판정**: `< 게이트가 공격(B/D) recall은 ___ 유지하면서 본문(G)을 대량 제거 →
best 신호 AUROC ___→___, benign FPR ___→___ 로 개선되었다 >`. 권장 게이트
keep_pct=`<__>` (`sink_filter_report.json › recommended`). **→ §4의 1차 거름 근거.**

---

## 4. §4 — cascade 방어 구현·평가 (stage 07) ★★

`pos0/cascade_report.json`. **1차 sink 게이트(keep `<__>%`) + 2차 threshold(`<신호>`
@layer`<__>`, FPR=`<__>` 운영점)**. `eval_mode=<holdout>` (n_train=`<__>`, n_test=`<__>`).
**모든 수치는 held-out test, threshold·신호 선택은 train.** (Fig 9a/9b)

| 전략 | B 차단 | D 차단 | C FPR | E FPR | F FPR | G FPR | ASR before | ASR after | blk@succ | 게이트통과 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-stage threshold | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | 1.0 |
| gate-only* | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` |
| **cascade** | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | `<__>` | **`<__>`** | **`<__>`** | `<__>` |

\* gate-only의 C/E열은 게이트 생존율(coverage)이지 detector FPR이 아님.
`blk@succ` = 성공했을 공격 중 실제 차단 비율(honest efficacy; ASR↓ 자체는 자명).

**판정**: `< cascade는 held-out에서 B/D 차단율을 ___ 유지하면서 C/E FPR을 1-stage
대비 ___ 낮추고(special-detector 함정 완화), ASR을 ___ → ___ (blk@succ ___)로
감소시켰다. 게이트로 2차 연산을 ___% 로 절감. → 최종 방어 기법으로 ___ 채택/권장 >`.

---

## 5. 종합 — 최종 방어 기법 결론 (실행 후 작성)

`< sink 1차 + threshold 2차 cascade가 단일 신호 대비 FPR/연산에서 우월하며,
운영점 ___ 에서 ASR ___% 감소·benign FPR ___% 를 달성 → 권장 구성: keep_pct=___,
신호=___@layer___, FPR=___. 한계: ___ (단일 모델/공격 템플릿, guard 표본 등). >`

## 6. 한계
- 단일 모델·단일 공격 템플릿(TM-1) — 일반화 추가 검증 필요.
- E(benign special) 표본은 본질적으로 적음 (균형 cap으로 통제하되 명시).
- **§2.3 단일-threshold(stage 05) AUROC·Youden·TPR@FPR는 in-sample 관례치**(전체
  데이터에서 임계 선택). §4 cascade는 hold-out으로 보정했으나, §2.3 표는 분리 신호의
  *상대* 비교용으로 해석. (best signal/layer 선택의 selection bias도 §2.3엔 잔존.)
- **ASR judge 라벨 교차확인**: `asr_judge=guard/both`로 돌렸어도 `asr_summary.json`의
  `guard_model_used=false`이면 실제로는 keyword로 폴백된 것 — 리포트 수치 인용 시 확인.
- guard 미사용 시 ASR 절대값은 keyword 휴리스틱 한계 잔존(상대 비교 위주).

## 부록 — 산출 파일
`extract_summary.json`(raw+balanced) · `asr_summary.json` ·
`pos{0,1}/{representation_metrics,cosine_pairs,operating_points,threshold_defense,threshold_asr,sink_filter_report,cascade_report}.{json,csv,md}` ·
그림 `result_report/figures/fig0{1..9}_*.png` (`make_figures.py`).
