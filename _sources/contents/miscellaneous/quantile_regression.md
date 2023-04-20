# Quantile Regression

## Quantile이란?

확률 변수 $X$가 있다고 하자. 그리고 확률 변수 $X$의 누적분포함수 (CDF)를 $F_X$라고 하자. 
$F_X(x)$는 확률 변수가 $X$가 $x$보다 작을 확률으로 정의된다. 즉, $F_X(x):=\operatorname{Pr}[X \le x]$ 이다. 
한편, 0과 1사이의 값 $\tau$에 대하여, 확률 변수 $X$의 $\tau$-quantile은 CDF 값이 $\tau$가 되는 $x$ 값을 의미한다. 
즉, $x=F^{-1}_X(\tau)$이다. 물론, CDF의 역함수가 정의되지 않을 수 있기 때문에 여기서 $F_X^{-1}$는 조금 다르게 정의된다. 

$$F_X^{-1}(p) := \inf \left\{ x: F_X(x) \ge p \right\}.$$
CDF에서 확률이 유지되는 평평한 부분 때문에 역함수가 정의되지 않는 것인데, 평평한 부분의 시작점을 선택하여 해결한다는 의미이다.  

<br>

## Quantile Regression

Quantile regression은  $\tau \in [0, 1]$가 주어졌을 때, $\tau$-quantile 값인 $F^{-1}_X(\tau)$를 찾는 문제이다. 

먼저, $\tau$-quantile에 대한 추정값을 $m$이라고 하자. 
만약 $m$이 정확하다면, 확률 변수 $X$의 분포에서 데이터 여러 개를 샘플링했을 때,
$\tau$ 비율만큼의 데이터는 $m$보다 작을 것이고, $1-\tau$ 비율만큼의 데이터는 $m$보다 클 것이다.

$m$이 정확하면 다음과 같은 수식이 성립할 것이다.

$$
\begin{matrix}
& F_X(m) & = & \tau \\ 
\iff & \displaystyle\int_{-\infty}^{m} f_X(x) \, dx & = & \tau \cdot 1 \\
\iff & \displaystyle\int_{-\infty}^{m} f_X(x) \, dx & = & \tau \displaystyle\int_{-\infty}^{\infty} f_X(x) \, dx \\
\iff & \displaystyle\int_{-\infty}^{m} f_X(x) \, dx & = & \tau \left( \displaystyle\int_{-\infty}^{m} f_X(x) \, dx + \displaystyle\int_{m}^{\infty} f_X(x) \, dx \right), \\
\end{matrix}
$$
마지막 식을 다시 한번 정리하면 다음과 같다.

$$ (1 - \tau) \displaystyle\int_{-\infty}^{m} f_X(x) \, dx - \tau \displaystyle\int_{m}^{\infty} f_X(x) \, dx =0.$$ 

위 식은 quantile 추정값 $m$이 정확할 때 성립한다. 위 식의 좌변이 중요하기 때문에 $m$에 대한 함수로 정의해놓자.

$$g(m) := (1 - \tau) \displaystyle\int_{-\infty}^{m} f_X(x) \, dx - \tau \displaystyle\int_{m}^{\infty} f_X(x) \, dx$$

만약 추정값 $m$이 틀렸다면 $g(m)$은 어떻게 될까? 예를 들어, 추정값 $m$에 대한 CDF 값을 $\tau'$라고 하면, $g(m)$은 다음과 같이 계산된다.

$$g(m)=(1-\tau)\tau' - \tau (1-\tau')=\tau' - \tau.$$

위 식으로 알 수 있는 점.
- 만약 $m$이 실제 $\tau$-quantile보다 작으면 $\tau'$이 $\tau$보다 작게 되고 $g(m)$이 음수가 된다. 
- 만약 $m$이 실제 $\tau$-quantile보다 크면 $\tau'$이 $\tau$보다 크게 되고 $g(m)$이 양수가 된다.

$g(m)$을 어떤 $m$에 대한 함수 $G(m)$의 도함수라고 생각해보면, $G(m)$은 $m$이 $\tau$-quantile보다 작을 때는 함수가 감소하고, $m$이 $\tau$-quantile보다 클 때는 증가하는 함수이다. 그리고 $m$이 $\tau$-quantile과 같을 때 최소값을 갖는다. $G(m)$은 어렵게 생겼지만 다음과 같다.

$$
\begin{matrix}
G(m) & = & (\tau -1)\displaystyle\int_{-\infty}^{m} (x - m) f_X(x) \, dx + \tau \int_{m}^{\infty} (x - m) f_X(x) \, dx \\
& = &\mathbb{E}_{X}\left[\rho_\tau(X - m) \right]
\end{matrix}
$$

where

$$
\rho_{\tau}(x) = x(\tau - \mathbb{I}_{x < 0}).
$$
여기서 $\mathbb{I}_{x < 0}$는 입력 $x$가 0보다 작으면 1이고 0보다 같거나 크면 0인 함수이다. $\rho_{\tau}(x)$는 입력 $x$가 0보다 작으면 $\tau x$, 0보다 같거나 크면 $(\tau - 1) x$이된다. 지금까지 내용을 바탕으로 정리하자면, $\tau$-quantile은 다음과 같은 최적화 문제를 풀어서 구할 수 있다. 

$$
q_X (\tau)  =  \operatorname*{argmin}_m \mathbb{E}_{X}\left[\rho_\tau(X - m) \right]
 $$


## 참고문헌

[1] https://en.wikipedia.org/wiki/Quantile_function