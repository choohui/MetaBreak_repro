Metabreak Defense Experiment Plan

대주제 : Metabreak의 semantic mimicry attack의 defense 기법 연구

1. special token과 regular token은 토큰임베딩에서의 cosine similarity 혹은 l2norm의 경향에서 차이를 가질 것이다.
-> 그렇지 않다. 실제 모델에서 사용되는 special token이 6개 내외로 적어 완벽한 분석은 어려우나, Regular token과 유사한 분포를 보이고 있다.
-> 또한, special token이 regular token이 분포하는 깔대기의 바깥에 산재하여 있다는 연구가 있기에 이 부분을 확인해 보았으나 실제 모델들에서는 그러한 부분을 확인할 수 없었다.
-> 그렇기에 internal representation을 보아야 한다.

2. 그렇다면 special token과 regular token은 model의 internal representation에서 어떠한 차이를 가지는가.
	2.1. 확인을 위해 token을 7종류로 분류하였다.
		A : 시스템이 주입한 chat template 용의 special token        [reference 기준점]
		B : malicious mimicry regular token - 공격용 l2치환 토큰     [positive(공격)]
		C : benign mimicry regular token - 정상 맥락에서 등장한 치환 토큰   [negative(정상)]
		D : malicious special token - 공격용 special token          [positive(공격)]
		E : benign special token - 정상 맥락에서 등장한 special token   [negative(정상)]
		F : positioned regular token - 공격용 자리에 들어간 benign regular token   [negative(정상) / 위치 통제]
		G : ordinary regular token - 일반 regular token             [negative(정상) / 본문 baseline]
	-> 방어 라벨 정의: **positive(공격) = B ∪ D**, **negative(정상) = C ∪ E ∪ F ∪ G**, A는 reference(기준점).
	-> (기존 experiments_hwichan 대비 신규 통제군) C = B와 같은 '치환 토큰'을 정상 맥락에 둔 것(토큰 정체성 통제), F = 공격 위치에 들어간 평범한 토큰(위치 효과 통제). 이 둘로 "공격성이 토큰 정체성에서 오는가 / 위치에서 오는가 / 맥락에서 오는가"를 분리한다.
	그리고 각 분류별로 prompt를 생성하여 저장해두고, 모델에 직접 입력하여 실제로 공격에 성공하는지 여부를 각 prompt별로 labeling(ASR)하여 추후 분석에 사용하여야 한다.
	2.2. 측정 신호는 5가지이다.
		hidden norm : hidden-state 벡터의 L2 norm
		sink : sink score
		value norm : value 벡터 norm
		output norm : attention layer의 출력 norm
		cos to d : hidden state와 special token 중심점의 cosine similarity
	2.3. 토큰은 해당 위치만을 확인한다.
-> 이를 통해서 아래 결론을 내고자 한다.
	-> 전체 hidden state 벡터를 활용하여 logistic regression 결과를 확인한다.
	-> cos similarity를 계산한다. cos(A,B), cos(A, D), cos(A,G), cos(B,C), cos(B,D), cos(B,F) 를 계산한다. 이 때, 대하여 평균 벡터를 계산하고 cos similarity를 계산하는 방식과 각각의 프롬프트에 대해 Cos similarity를 구하고 분포를 구하는 기법 모두 수행한다. layer별로 수행한다.
	internal representation vector에서 어떠한 차이를 보이는지 확인하기 위함.
	-> single threshold 방어 가능성 판별 : 각 측정신호별 분석을 수행한다. 각 type별 tpr fpr 분석, ASR 기반 분석 두가지를 수행한다.
	**이후 defense에 사용한다.**

3. sink score으로 봐야하는 token의 범위를 최대한 축소한 후 신호를 threshold로 보는 방법에 대하여 tpr fpr 분석, ASR 기반 분석을 수행한다.

4. defense 기법 생성
지금까지의 결과를 바탕으로 sink score를 이용한 1차 거름, 측정 신호를 threshold로 사용하여 2차 거름.이라는 defense 부분을 실제로 생성하고 이 기법이 실제 공격을 얼마나 잘 막는지를 실험.

