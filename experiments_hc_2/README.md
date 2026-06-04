# experiments_hc_2 — MetaBreak semantic-mimicry 방어 연구 (hc_1 개선·확장판)

victim 모델(Llama-3.1-8B)의 **내부표현 신호**만으로 MetaBreak의 semantic-mimicry
jailbreak를 탐지·방어할 수 있는가? 그리고 **최종적으로 어떤 방어 기법을 쓸 수
있는가?** 이 폴더는 그 질문에 답하기 위한 전체 실험 캠페인이다 (Main.md §1–§4).

`experiments_hc_1`의 개선·확장판으로, 아래 한계를 보완하고 **§3 sink 필터링**과
**§4 2단 cascade 방어**를 1급(stage)으로 구현·평가한다.

| hc_1 한계 | hc_2 보완 |
|---|---|
| C(benign mimicry) census=0 (치환문자열이 재토큰화 안 됨) | **stage 03에 token-splice C 추출을 처음부터 통합** (`core/benign_inject.py`) |
| type별 토큰 수 불균형 (A3600…C0) | **`--balanced` 자동 동일-cap** + `extract_summary.json`에 `raw_census`/`census` 병기 |
| logreg AUC≈1.0 누수 우려 | **prompt-level GroupKFold** (`probe_auc` naive vs `probe_auc_grouped` honest) |
| ASR이 거부-키워드 휴리스틱 (F 14% 위양성) | **`--asr_judge {keyword,guard,both}`** (Llama-Guard 옵션) |
| §4 cascade 미구현 | **stage 07에서 실제 detector 구현 + 차단율/FPR/ASR end-to-end 평가** |

## 디렉터리 구조

```
experiments_hc_2/
  Main.md                 실험 스펙 (§1–§4)
  config.py               ExpConfig + CLI (모든 stage 공용 knobs)
  run_all.py              오케스트레이터 — 모델 1회 로드 후 stages 00–07
  smoke_test.py           모델 없이 전구간 무결성 검증
  stages/                 번호식 파이프라인 단계 + 분석 헬퍼
    analysis_common.py    신호/라벨/ASR 공용 헬퍼 (04–07 재사용)
    00_embedding_analysis.py   §1
    01_build_prompts.py        §2.1  (C carrier 포함)
    02_run_asr.py              §2.1  (keyword/guard 판정)
    03_extract_representations.py §2.2 (C 통합 + balanced + raw census)
    04_probe_cosine.py         §2.3  (naive + grouped 확률 AUC, cosine)
    05_threshold_defense.py    §2.3  (+ operating_points.json)
    06_sink_filter.py          §3    (sink 게이트 keep-% sweep)
    07_cascade_defense.py      §4    (2단 cascade, 차단율/FPR/ASR)
  core/                   재사용 라이브러리 (모델·캡처·신호·라벨·메트릭·mock)
    cascade.py            2단 detector 프리미티브 (sink_gate + threshold + 집계)
  data/                   통제군 seed (benign_special / benign_mimicry / positioned)
  result_report/          분석 리포트 + make_figures.py
  results/                모든 출력 (gitignore: *results)
```

- **`core/`** = 순서 없는 라이브러리, **`stages/`** = 순서 있는 파이프라인,
  루트 = 진입점. `run_all.py`가 `stages/`의 번호식 스크립트를 경로 로드한다.

## 7 토큰 타입과 방어 라벨 (Main.md §2.1)

A=system special(reference) · **B=malicious mimicry(공격)** · C=benign mimicry(정상,
토큰정체성 통제) · **D=malicious special(공격)** · E=benign special(정상) ·
F=positioned regular(정상, 위치 통제) · G=ordinary(정상, baseline).
**positive=B∪D, negative=C∪E∪F∪G, A=reference.**

5개 측정 신호: `hidden_norm`, `sink`, `value_norm`, `output_norm`, `cos_to_ref`.

## 실행

`repro_mb`(이 폴더의 상위) 디렉터리에서 **모듈로** 실행한다.

```powershell
# 모델 없이 전구간 검증 (mock)
python -m experiments_hc_2.smoke_test

# 실제 모델 전체 실행
python -m experiments_hc_2.run_all --model <Llama-3.1-8B-Instruct 경로> --n 150
# ASR을 Llama-Guard로도 판정 (F 위양성 비교)
python -m experiments_hc_2.run_all --model <...> --n 150 --asr_judge both --guard_model <Llama-Guard 경로>
# 일부 stage만
python -m experiments_hc_2.run_all --model <...> --stages 03,04,05,06,07

# 리포트 그림 생성 (모델 불필요)
python -m experiments_hc_2.result_report.make_figures
```

개별 stage는 경로로도 실행 가능: `python experiments_hc_2/stages/06_sink_filter.py --smoke`.

주요 knobs(`config.py`): `--balanced/--no-balanced`, `--cap_per_type`,
`--asr_judge`, `--sink_filter_pcts`, `--cascade_keep_pct`, `--cascade_signal`,
`--cascade_layer`, `--cascade_fpr`, `--pos_offsets`.

## 방법론적 타당성 설계 (balanced vs raw, hold-out)

실험 결론의 타당성을 위해 데이터 사용을 단계별로 분리한다:

- **stage 03은 raw(전체) 토큰셋을 저장**하고, 균형화는 `balanced_row_ids`
  **인덱스**로만 기록한다. 토큰을 버리지 않는다.
- **§2 (stage 04 probe / 05 threshold)**: `balanced=True` 부분집합 사용 — type
  개수가 같아야 AUROC 비교가 공정.
- **§3/§4 (stage 06 sink-filter / 07 cascade)**: `balanced=False` **raw 전체** 사용 —
  게이트의 "본문(G) 제거" 효과는 현실적 per-prompt 분포에서만 의미가 있으므로,
  균형화로 G를 미리 솎아내면 안 된다.
- **§4 cascade는 prompt 단위 hold-out**: threshold와 (signal, layer) 선택은 train,
  차단율·FPR·ASR은 held-out test에서 측정(`eval_mode`; 프롬프트가 적으면 in_sample
  폴백·표기). reference **A는 게이트 후보에서 제외**.
- **probe**: naive(per-token) vs grouped(prompt-level GroupKFold) AUC 병기.

## 의존성 규칙 (엄수)

- **허용**: `repro_mb/src/*`, `repro_mb/prompts/*`(단 `MetaBreak_data/` 제외),
  `repro_mb/results/llama/replacement.json`.
- **금지**: 다른 `experiments_*` 폴더 import, `.gitignore`의 `*results`에 걸리는
  경로(자기 출력 포함 — 타 실험 results를 읽지 않음). `core/`는 자체 복제본이다.

## 핵심 산출물 (results/hc2_llama31_8b/)

`extract_summary.json`(raw+balanced census) · `asr_summary.json`(judge별 ASR) ·
`pos{0,1}/representation_metrics.json`(naive+grouped probe, cosine) ·
`pos{0,1}/operating_points.json`(threshold 운영점) ·
`pos{0,1}/sink_filter_report.{json,md}`(§3 게이트 sweep) ·
`pos{0,1}/cascade_report.{json,md}`(§4 차단율/FPR/ASR).
