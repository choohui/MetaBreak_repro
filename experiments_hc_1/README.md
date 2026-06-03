# experiments_hc_1 — MetaBreak Semantic-Mimicry 방어 연구

`Main.md`의 §1~§3을 구현한 **독립 실험 폴더**입니다. victim 모델의 **internal
representation**을 분석하여 MetaBreak semantic-mimicry 공격을 탐지/방어할 수 있는지
연구합니다. (§4 cascade 방어기법 생성은 `experiment_step.md`에 명세만 되어 있고 코드는
추후 생성합니다.)

## 의존성 규칙
- 의존 허용: `repro_mb/src/*`, `repro_mb/prompts/*`, `repro_mb/results/llama/replacement.json`
- 의존 금지: 다른 `experiments_*` 폴더 (forward-capture/sink/labeling 로직은 `core/`에 자체 구현)

## 빠른 시작

```bash
# (1) 모델 없이 전체 파이프라인 스모크 검증 (transformers 불필요, torch/numpy/sklearn만)
python experiments_hc_1/smoke_test.py

# (2) 실제 모델로 전체 실행 (Llama-3.1-8B-Instruct 로컬 경로 필요)
#     --n 이 모든 타입(B/D/F/G 및 C/E)의 prompt 수를 결정합니다. 타입별 ~150개 권장.
python experiments_hc_1/run_all.py --model <LLAMA_PATH> --n 150

# (3) 일부 단계만
python experiments_hc_1/run_all.py --model <LLAMA_PATH> --stages 03,04,05

# (4) 개별 단계 단독 실행 (모델 불필요 단계는 디스크 산출물만 사용)
python experiments_hc_1/01_build_prompts.py --n 50
python experiments_hc_1/03_extract_representations.py --model <LLAMA_PATH> --n 50
python experiments_hc_1/05_threshold_defense.py
```

선택: `--n_benign 150` 로 C/E 개수를 B/D/F/G와 별도로 지정(기본 = `--n`),
`--guard_model <LLAMA_GUARD_PATH>` 로 Llama Guard 판정 추가, `--ordinary -1` 로 모든
본문 토큰을 G로 사용, `--pos_offsets 0` 로 슬롯 토큰만 분석, `--smoke` 로 가짜 모델 사용.

> **표본 수**: 각 타입의 prompt 수는 `--n`(C/E는 `--n_benign`)으로 정해집니다. C/E는 curated
> seed(약 20·25개) 먼저 사용 후 템플릿 자동 생성으로 목표치까지 보강합니다
> (`core/benign_gen.py`). B/D는 prompt당 9개, F는 3개, C/E는 1개, G는 prompt당 `--ordinary`개
> 토큰을 만들어내므로 **토큰 수준** 표본은 타입별로 다릅니다(prompt 수는 동일).
>
> **A·G 과수집 제어**: A(템플릿 special)와 G(본문)는 모든 prompt에서 누적되어 토큰 수가
> 매우 커집니다. 이를 제한하는 두 knob이 있습니다.
> - `--max_a_per_prompt N` (기본 2, -1=전부): prompt당 수집할 A 토큰 수.
> - `--cap_per_type N` (기본 없음): (category, pos_offset)별 전역 상한으로 균등 다운샘플링 →
>   타입별 토큰 수를 ~N으로 맞춰 분석(logreg/threshold)의 클래스 균형을 잡습니다.
>   예: `--n 150 --cap_per_type 300`.

## 7종류 토큰 타입 (Main.md §2.1)

| 문자 | 타입 | 방어 라벨 | 생성 방법 |
|---|---|---|---|
| A | system special | reference | 모든 prompt의 chat-template special 토큰 |
| B | malicious mimicry | **positive(공격)** | TM-1에 mimicry 적용(`ujících`/`�`) |
| C | benign mimicry | negative | `ujících` 포함 정상 문장 (토큰 정체성 통제) |
| D | malicious special | **positive(공격)** | TM-1 원본 (literal special) |
| E | benign special | negative | special 토큰 언급 정상 문장 |
| F | positioned regular | negative | TM-1 골격의 공격 슬롯에 benign 단어 (위치 통제) |
| G | ordinary regular | negative | 정상 본문 토큰 baseline |

`positive = B∪D`, `negative = C∪E∪F∪G`, `A = reference`.

## 5가지 측정 신호 (Main.md §2.2)
`hidden_norm`(hidden L2) · `sink`(attention sink score) · `value_norm`(‖v_proj‖) ·
`output_norm`(‖o_proj 입력‖) · `cos_to_ref`(hidden ↔ A 중심점 cosine).

## 단계 (stage)

| stage | Main.md | 모델 | 산출물 |
|---|---|---|---|
| `00_embedding_analysis` | §1 | ✓ | `embedding_analysis.{json,md}` |
| `01_build_prompts` | §2.1 | ✗ | `prompts.jsonl` |
| `02_run_asr` | §2.1 | ✓ | `asr.{jsonl,csv}`, `asr_summary.json` |
| `03_extract_representations` | §2.2/2.3 | ✓ | `tokens.jsonl`, `features.npz`, `extract_summary.json` |
| `04_analyze_cosine_logreg` | §2.3 | ✗ | `pos{k}/representation_metrics.{json,csv}`, `cosine_pairs.json`, `ref_centroids.npz`, `pca_coords.npz` |
| `05_threshold_defense` | §2.3 | ✗ | `pos{k}/threshold_defense.{json,md}`, `threshold_per_type.csv`, `threshold_asr.json` |
| `06_sink_range` | §3 | ✗ | `pos{k}/sink_range_report.{json,md}` |

§3(stage 06)은 §2와 **동일한 산출물**을 재사용하므로 `run_all` 한 번으로 §2·§3가 함께
수행됩니다.

## 분석 내용
- **logistic regression**: 전체 hidden 벡터로 공격(B∪D) vs 정상(C∪E∪F∪G)을 레이어별 5-fold
  CV로 분리 (sklearn, 없으면 nearest-centroid fallback). → `representation_metrics.*`
- **cosine similarity**: 쌍 (A,B)(A,D)(A,G)(B,C)(B,D)(B,F)을 레이어별로, 중심점-대-중심점 +
  prompt별 분포 두 방식 모두. → `cosine_pairs.json`
- **single-threshold 방어**: 5신호 × 레이어 ROC-AUC/Youden/TPR@FPR, 타입별 분해 + ASR 기반.
  → `threshold_defense.*`, `threshold_asr.json`

## 디렉터리
```
core/          모델·캡처·sink·labeling·신호·지표·mock (자체 구현)
data/          C/E/F 통제군 seed 데이터
NN_*.py        각 단계 스크립트 (run(cfg, lm=None) 노출)
config.py      ExpConfig + 공통 CLI
analysis_common.py  stage 05/06 공용 분석 로직
run_all.py     모델 1회 로드 후 선택 단계 일괄 실행
smoke_test.py  모델 없이 전 경로 검증
experiment_step.md  §1~§4 step 분해 (선택형, §4는 미구현 명세)
```

## 요구 패키지
`torch`, `numpy`, `tqdm` (필수). `transformers`(실제 모델 단계), `scikit-learn`(logreg; 없으면
fallback). 스모크 테스트는 `transformers` 없이 동작합니다.
