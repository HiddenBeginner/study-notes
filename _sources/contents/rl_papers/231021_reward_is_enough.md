# Reward Is Enough

- 제목: Reward Is Enough
- 저자: Silver, David, Satinder Singh, Doina Precup, and Richard S. Sutton, *DeepMind*
- 연도: 2021년
- 학술대회: Artificial Intelligence
- 링크: [https://www.sciencedirect.com/science/article/pii/S0004370221000862?via%3Dihub](https://www.sciencedirect.com/science/article/pii/S0004370221000862?via%3Dihub)

<br>

---

DeepMind의 보상 (reward)에 대한 철학을 담은 논문.
주어진 문제를 해결하기 위해 요구되는 지능과 능력이 있다고 하자. 
이 지능과 능력을 어떻게 획득할 수 있을까?
한 가지 방법은 각 능력마다 개별적으로 특수한 알고리즘을 고안할 수 있을 것이다.
예를 들어, 감각 기관을 통하여 대상을 인식하는 지각 (perception)은 객체 분할과 인식 (object segmentation and recognition) 능력과 관련되어 있는데, 
이 능력을 얻기 위하여 segmentation 모델을 만들면 될 것이다.

하지만 이 논문에서는 각 능력을 얻기 위해 개별 알고리즘을 개발할 필요 없이,
보상 최대화라는 다소 포괄적인 목적을 수행함으로써 지능과 이와 관련된 능력들을 획득할 수 있다는 가설을 제시한다. 내가 DeepMind의 철학을 완전히 이해하지 못할 것이고, 설사 이해했더라도 이를 쉽게 설명할 수 있는 능력이 없다. 이 포스팅은 그냥 내가 이 논문을 읽고 흥미로웠던 구절들을 정리해놓은 것 뿐이다.

<br>

**Reward hypothesis**은 모든 목표 (goal)를 보상으로 표현할 수 있다는 가설이다.
그리고 누적 보상을 최대화함으로써 지능 (intelligence)과 이와 관련된 능력들 (abilities)을 유도할 수 있다는 가설이 **reward-is-enough hypothesis**이다.
이때 지능이란 주어진 목표를 달성하기 위한 능력에서 연산을 수행하는 (computational) 부분이다:

> "Intelligence is the computational part of the ability to achieve in the world."  *John McCarthy*

<br>

단순하게 생각하면, 원하는 행동을 할 경우 양의 보상을 부여하고, 잘못된 행동에는 음의 보상을 부여하면 내가 원하는 능력을 수행하는 지능을 개발할 수 있겠다. 하지만 이 가설은 이런 당연한 인과 관계를 말하는 것이 아니다. 어떤 능력을 유도할 수 있는 보상 신호는 하나로 유일한 것이 아니라, 다양한 보상 신호가 능력을 유도할 수 있다. 뿐만 아니라, 아주 간단한 보상 신호 하나를 최대화함으로써 다양한 지능과 이와 관련된 능력들을 획득할 수도 있다. 이 가설이 성립하는 가장 대표적인 예는 AlphaZero Go이다. 바둑을 이기면 +1, 지면 -1을 부여하는 보상을 최대화함으로써 바둑을 두는데 필요한 지능과 능력들을 획득할 수 있었다.

<br>

그럼 보상을 최대화할 수 있는 방법론이 무엇이 있는지 궁금할 것이다. 
이 논문에서는 가장 일반적이고 확장성 (scalable) 있는 방법론은 trial-and-error로 직접 환경과 상호작용하며 보상을 관측하고 점점 더 많은 보상을 받도록 행동하는 것이라고 한다.
그리고 현실 세계와 같은 엄청 풍부한 (rich) 환경에서 보상을 최대화한다면, general intelligence을 정교하게 유도할 수 있을 것이라고 추측한다.

<br>

보상을 최대화하는 방법론 중에는 강화학습도 있다. 강화학습은 주어진 문제를 두 개의 시스템 (에이전트와 환경)으로 나눈다.
에이전트는 의사 결정을 하는 시스템이고, 환경은 의사 결정의 영향을 받는 시스템이다.
강화학습은 주어진 목표를 누적 보상으로 표현해낸다. 그렇다면 보상은 목표를 달성한 정도를 측정한 것으로 이해할 수 있다.

