# experiments_hc_4_claude — Non-Logistic Scalar-Signal + Threshold Defense

대주제: MetaBreak semantic-mimicry attack에 대한 **per-token 방어** 기법 연구 —
internal representation에서 신호를 찾아 **scalar 한 값**으로 변환하고, **threshold**로
악의적 토큰을 특정·배제한다. **logistic regression classifier는 사용하지 않는다.**

## 연구 질문 (왜 이 실험인가)

- **hc_2**: single-signal threshold는 in-sample AUC ~0.9이지만, sink-gate +
  single-threshold cascade가 **held-out에서 block-rate 0%** 로 붕괴했다 (train→test
  분포 이동에 보수적 threshold가 전이되지 못함). → 이것이 이겨야 할 실패 모드.
- **hc_3**: engineered feature 위 sparse **logistic regression**으로 held-out AUC≈1.0,
  ASR 47.6%→0%. 그러나 이는 사용자가 원치 않는 classifier.
- **hc_4_claude의 질문**: classifier 없이 **scalar+threshold** 만으로 held-out 일반화 +
  ASR 감소가 가능한가? (해석 가능성을 유지하면서 hc_3의 성공을 재현)

## §1. 토큰 7분류 (A–G) — 모든 type을 **동일 개수**로 균형

| | 정의 | 방어 라벨 |
|---|---|---|
| A | system special (chat-template) | reference |
| B | malicious mimicry (L2 치환 공격 토큰) | **positive (공격)** |
| C | benign mimicry (정상 맥락의 치환 토큰) | negative / 정체성 통제 |
| D | malicious special (공격용 special 토큰) | **positive (공격)** |
| E | benign special (정상 맥락의 special) | negative |
| F | positioned regular (공격 위치의 평범한 토큰) | negative / 위치 통제 |
| G | ordinary regular (본문 토큰) | negative / baseline |

positive = B∪D, negative = C∪E∪F∪G, A = reference. hc_2와 달리 **A를 포함한 7종
모두**를 같은 개수로 cap한다 (`balance_a=True`, stage 03의 `balanced7`).

## §2. Scalarizer 메뉴 — internal rep → 1 scalar/token/layer (classifier 없음)

모든 fit (중심점/공분산/방향)은 **TRAIN row에서만** 수행한다.

**Clean set (headline)** — 순수 측정 또는 one-class OOD 거리, 2-class 경계 학습 없음:
`hidden_norm, value_norm, output_norm, sink` (raw) ·
`cos_to_ref` (A 중심점 코사인) · `cos_to_attack` (B∪D 중심점 코사인) ·
`mahalanobis_benign` (benign Gaussian까지 거리, shrinkage) ·
`pca_resid` (benign PCA 부분공간 재구성 잔차) ·
`energy_lse` (whitened 좌표의 logsumexp OOD energy) ·
`active_value`, `active_output` (sink×norm).

**Borderline set (별도 블록)** — TRAIN에서 1-D 방향을 fit → linear 경계에 근접:
`diff_means` (μ_attack−μ_benign 투영) · `lda_1d` (Fisher 방향) ·
`pca_sep_proj` (분리도 최고 PCA 축 투영). `--scalarizer_set clean|borderline|all`로
선택하며, headline claim은 clean만 사용한다.

**Per-prompt 정규화 래퍼** (`--normalize none|zscore|rank|robust`): 각 prompt 내부에서
정규화 → train→test 스케일 이동을 제거 (hc_2 붕괴에 대한 핵심 대응). prompt 내부 통계만
쓰므로 leakage 없음.

## §3. Threshold 선택 (모두 TRAIN fit, oriented score 위)
`youden`, `fpr@{1,5,10}`, `eer`, `pct_benign@{95,99}`, `cost`(`--fn_fp_cost`).
**Threshold 안정성**(CV fold + bootstrap 재적합의 mean/std/cv/CI)을 *선택 기준*으로
사용 — 단순 최고 in-sample AUC가 아니라 **안정적 threshold**를 우선 (hc_2 교훈).

## §4. 엄밀성 방법론 → 산출물
prompt-level GroupKFold (토큰 leakage 차단, fitted scalarizer는 out-of-fold AUC) ·
train/held-out split (hc_2 실패 시나리오) · bootstrap CI (AUC·threshold) ·
permutation test (선택된 op-point) · counterfactual paired control
(B−C, B−F, D−E, D−F, F−G) · type별 TPR/FPR · ROC/DET/PR + calibration ·
layer sweep (best layer는 **TRAIN에서만** 선택) · ablation
(정규화 / scalarizer family / sink-gate) · end-to-end ASR (block-rate proxy +
선택적 실제 token 배제 재생성).

## Stage 파이프라인 (00–09)
00 embedding · 01 build 7-type prompts · 02 ASR(B/D/F) · 03 extract + 7-way balance ·
04 scalarize (TRAIN fit, honest train AUC) · 05 threshold + stability + op-point(train) ·
06 held-out 평가 · 07 counterfactual · 08 token-exclusion 방어 + ASR(proxy/real) ·
09 robustness/ablation. 00/02/03만 모델 필요 (08은 `--real_intervention` 시).

## 엄밀성 리스크와 완화
1. test에서 선택 = leakage → 04/05는 train에서만 fit·선택(`selected_on:"train"`), 06만 test 판독.
2. 불균형/A 과수집 → 7-way `balanced7` cap, geometry는 train-only.
3. hc_2 threshold 붕괴 → per-prompt 정규화 + threshold-cv 선택 기준 + 정직한 held-out 보고.
4. mock smoke의 형상 미검증 → mock이 [N,L+1,dim] 큐브 생성, 공분산 scalarizer는 shrinkage로 finite 보장.
5. "classifier 아님" 경계 → borderline 태그·분리, clean이 headline.
6. 다중비교 → 최종 op-point에만 permutation test, 모든 headline AUC에 bootstrap CI.
