# QR-DQN

- 제목: Distributional Reinforcement Learning With Quantile Regression
- 저자: Dabney, Will, Mark Rowland, Marc Bellemare, and Rémi Munos, *DeepMind*
- 연도: 2018년
- 학술대회: AAAI
- 링크: [https://openreview.net/forum?id=S1lOTC4tDS](https://openreview.net/forum?id=S1lOTC4tDS)
- 키워드: Distributional RL


일반적인 RL 알고리즘이 return의 기댓값인 가치 함수에 관심이 있는 반면 Distributional RL은 return의 분포에 관심이 있다.

## 핵심 아이디어

주어진 정책 $\pi$의 상태 $s$와 행동 $a$에 대한 return의 확률 분포를 $Z^{\pi}(s, a)$라 하자.
Distributional RL은 $Z^{\pi}(s, a)$를 찾는 것이 목표이다.


## Quantile regression
확률 변수 $X$라 하자. 그리고 확률 변수 $X$의 누적분포함수 (CDF)를 $F_X$라고 하자. $F_X(x)$는 확률 변수가 $X$가 $x$보다 작을 확률으로 정의된다. 즉, $F_X(x):=\operatorname{Pr}[X \le x]$ 이다. 한편, 0과 1사이의 값 $\tau$에 대해 확률 변수 $X$의 $\tau$-quantile은 CDF 값이 $\tau$이 되는 $x$ 값을 의미한다. 즉, $x=F^{-1}_X(\tau)$이다. 물론, CDF의 역함수가 정의되지 않을 수 있기 때문에 여기서 $F_X^{-1}$는 조금 다르게 정의된다. 

$$F_X^{-1}(p) := \inf \left\{ x: F_X(x) \ge p \right\}.$$

CDF에서 확률이 유지되는 평평한 부분 때문에 역함수가 정의되지 않는 것인데, 평평한 부분의 시작점을 선택하여 해결한다는 의미이다.  Quantile regression은  $\tau \in [0, 1]$가 주어졌을 때, $\tau$-quantile 값인 $F^{-1}_X(\tau)$를 찾는 문제이다. 이를 찾기 위해 먼저 다음과 같은 함수를 사용할 것이다.

$$
\rho_{\tau}(m) = m(\tau - \mathbb{I}_{m < 0}),
$$
여기서 $\mathbb{I}_{m < 0}$는 입력 $m$이 0보다 작으면 1이고 0보다 같거나 크면 0인 함수이다. $\rho_{\tau}(m)$는 입력 $m$이 0보다 작으면 $\tau m$, 0보다 같거나 크면 $(\tau - 1) m$이된다. $\tau$-quantile은 다음과 같은 최적화 문제를 풀어서 구할 수 있다. 

$$
\begin{matrix}
q_X (\tau) & = & \operatorname*{argmin}_u \mathbb{E}_{X}\left[\rho_\tau(X - u) \right] \\
& = & \operatorname*{argmin}_u \left\{ (\tau -1)\int_{-\infty}^{u} (x - u) \, dF_X(x) + \tau \int_{u}^{\infty} (x - u) \, dF_X(x) \right\} 
\end{matrix}
 $$

위 방법에 대한 직관적인 설명을 고민해보았는데, 마땅히 떠오르지 않았다. 증명은 간단하다. 
