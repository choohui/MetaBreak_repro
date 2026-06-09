# Metabreak Defense Experiment Plan

> **대주제** : Metabreak의 semantic mimicry attack의 defense 기법 연구

## 0. 데이터셋 준비

**metabreak** 에서 사용한 프롬프트를 그대로 가져오되, 실제로 malicious한지를 확인하여 malicious한 프롬프트, benign한 프롬프트만을 수집한다.

---

## 1. 토큰 임베딩만으로 구분 가능한가?

> Metabreak 공격에 사용되는 malicious mimicry token과 special token, regular token은 토큰 임베딩만으로 구분 가능한가?

- **근거** : regular token들의 분포 및 L2 distance, Cosine similarity(vector 방향)은 special token과 구분되어 있다.

**→ 그렇지 않다.** special token과 regular token은 토큰 임베딩만으로 구분되지 않는다.

---

## 2. Internal Representation을 보자

### 2.0. 구분 — token type 분류

token들을 7가지 type으로 분류한다.

| Type | 설명 | 역할 |
|---|---|---|
| **A** | 시스템이 주입한 chat template 용의 special token | reference 기준점 |
| **B** | malicious mimicry regular token — 공격용 l2치환 토큰 | positive (공격) |
| **C** | benign mimicry regular token — 정상 맥락에서 등장한 치환 토큰 | negative (정상) |
| **D** | malicious special token — 공격용 special token | positive (공격) |
| **E** | benign special token — 정상 맥락에서 등장한 special token | negative (정상) |
| **F** | positioned regular token — 공격용 자리에 들어간 benign regular token | negative (정상) / 위치 통제 |
| **G** | ordinary regular token — 일반 regular token | negative (정상) / 본문 baseline |

> dataset도 이에 맞춰 추가하였고, 실험 과정에서는 token수를 맞추어 엄밀성 확보함

### 2.1. 구분이 되긴 하는가?

- **→** internal representation을 logistice regression을 하였더니 잘 나왔다.
- **→** special token과 regular token을 나누는 무언가가 있을 것이다.

### 2.2. 그 무언가가 무엇인가?

internal representation으로 다음 signal을 확인하고, tpr, fpr과 ASR을 확인한다.

#### ① clean — 순수 측정 / one-class OOD (headline 주장)

| 신호 | 계산 내용 |
|---|---|
| `hidden_norm` | hidden-state 벡터의 L2 norm (layer별) |
| `value_norm` | value 투영(v_proj)의 L2 norm |
| `output_norm` | attention 출력(o_proj)의 L2 norm |
| `sink` | attention sink score (head 평균) |
| `active_value` | `sink × value_norm` |
| `active_output` | `sink × output_norm` |
| `cos_to_ref` | hidden과 **A(시스템 special) 중심점**의 cosine |
| `cos_to_attack` | hidden과 **공격(B∪D) 중심점**의 cosine ← **헤드라인 성능(held-out AUC ~0.96)** |

> 단일 임계값으로 구분하고자 하였으나, train set에서만 효과적인 모습을 보임.
> 분류기 및 안정화 필요성.

#### ② borderline — 학습된 1-D 선형 방향 (분류기에 근접)

| 신호 | 계산 내용 |
|---|---|
| `diff_means` | **= attack_minus_benign**. 단위 (μ_attack − μ_benign) 방향으로의 부호 있는 투영 |
| `pca_sep_proj` | TRAIN-PCA 성분 중 train에서 \|AUC\| 분리력이 가장 큰 1개로의 투영 |

- **→** sink score는 frp이 높고, tpr이 낮았다.
- **→** hidden state를 pca하고 threshold를 두는 것도 은근 괜찮았다.
- **→** token 수준으로 malicious token을 잘 잡아내고(tpr이 높고), benign token을 그대로 두는(fpr이 낮은) 기법은 diff means이다. 또한, ASR도 굉장히 많이 낮아진다.

---

## 3. 어떻게 방어할 수 있는가?

탐지 프롬프트를 전부 거부하는 것이 가장 좋으나 utility 관점에서 좋지 않다.

### 3.1. masking

- **→** 단순히 unk token, eos token으로의 치환은 ASR을 오히려 증가시킨다.
- token 등 중립 단어로 치환할 경우 0.647 -> 0.474

### 3.2. steering

hidden state를 benign 방향으로 미는 방법으로는 충분한 steering이 되지 않는다.(부분적인 효과)

### 3.3. drop

- **→** special token +_1 token을 제거한 경우 효과적이다.
해당 토큰 직접 제거도 어느정도 효과 있음
> drop을 위주로 defense 진행

### 3.4 결론

지금까지 해온 것으로 보아, diff-means를 통해 hidden state를 확인하여 malicious한 token을 탐지하고 그 token 앞뒤 1토큰을 drop하는 방어 기법이 효과를 보이는 것으로 확인된다.

## 4. 모든 모델에 적용가능한가?

확인을 위해 Llama-3.1-8B-Instruct에 추가로
Qwen2.5-7B-Instruct
Mistral-7B-Instruct-v0.3
Gemma-2-9B-it
3가지 모델에 대하여

ours, 
Llama-guard
JBShield: Defending LLMs from Jailbreak Attacks through Activated Concept Analysis and Manipulation
GUARD-SLM : Token Activation-Based Defense Against Jailbreak Attacks for Small Language Models
4가지 방어기법을 비교한다.

이 때, 사용하는 프롬프트는
Metabreak에서 제시한 공격 프롬프트(mimicry 적용된)
GSM8k + Mimicy attack 헤더 추가한 버전
2가지 이고,

Metabreak에 대해 잘 탐지되고 막고 있는지,
GSM8k의 정답을 잘 말하는지
를 확인한다.

# main cotribution

- llama guard의 prompt 단위의 거절이 아닌 token 단위의 detect를 통한 sanitizing으로 utility를 확보함.
> agent, prompt 등등 외부에서 악성 prompt가 와도 방어 가능

- 분류기 등 학습이나 2차 추론 없이 defense가 가능하여 overhead를 줄임

