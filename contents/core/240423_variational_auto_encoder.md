# Variational Autoencoder (VAE)

맨날 헷갈리는 VAE 정리 =ㅅ=. Variational inference에 대한 개념 정리 위주이기 때문에 그림 없음 주의!

<br>

## 문제 정의

$\mathbf{X}=\left\{ x_i \in \mathbb{R}^{n} \right\}_{i=1}^{N}$을 $n$차원 데이터를 $N$개 모아놓은 집합이라고 하자. 
그리고 이 $N$개의 데이터는 어떤 알 수 없는 확률분포 $p$에서 독립적으로 샘플링되었다고 가정하자.
우리의 목표는 뉴럴 네트워크를 사용하여 데이터를 생성시킨 확률분포 $p$를 근사하는 것이다.
따라서 뉴럴 네트워크로 근사시킨 확률분포를 $p_\theta$로 표기할 것이다.
우리의 maximum likelihood estimation (MLE) 목적함수는 다음과 같다.

$$\operatorname*{maximize}_\theta \frac{1}{N}\sum_{i=1}^{N} \log p_\theta(x_i).$$

<br>

---

## Latent Variable Model

여기서 우리는 latent variable model을 사용할 것이다. 
데이터 $x_i$를 생성하는 데 필요한 실질적인 정보가 어떤 latent variable $z$에 담겨져 있다고 가정하는 것이다.
예컨데, 용의자의 몽타주를 그리기 위하여 용의자의 얼굴을 묘사할 때 "얼굴은 길고, 눈은 작으며 쌍커풀이 없다. 안경을 썼고, 코는 크며, 입은 얍실하게 생겼다." (필자 얼굴 절대 아님) 등으로 묘사할 것이지 얼굴의 좌표마다 색상값을 알려주진 않을 것이다. 
얼굴의 좌표마다 색상값으로 나타난 것이 원래 이미지 $x_i$라고 생각하면 되고, 몽타주 묘사가 latent variable $z$라고 생각하면 된다.
또한 같은 몽타주 묘사에서도 다양한 결과 사진이 생성되는 특징도 있을 것이다.
실질적인 정보 $z$를 바탕으로 $x_i$를 생성하는 것이므로 우리가 근사할 확률 분포는 $p_{\theta}(x_i | z)$이 되며, VAE에서는 이를 디코더라고 부른다. 일반적으로 $z \in \mathbb{R}^m$를 $m$차원 벡터로 사용하며, $m$은 $n$보다 작게 설정하는 것이 국룰이다 $(m << n)$.

<br>

Latent variable을 도입하면 좋은 점 중 하나는 $p_{\theta}(x_i | z)$를 가우시안 분포 등 비교적 단순한 확률분포로 모델링하여도 최종적인 $p_{\theta}(x_i)$는 직접 계산이 불가능할 정도로 복잡한 확률분포가 된다는 것이다. Latent variable $z$를 도입하면 $p_\theta(x_i)$를 다음과 같이 다시 적어줄 수 있다.

$$
\begin{matrix}
p_\theta(x_i) 
& = & \displaystyle \int_{z} p(x_i, z) \, dz \\[1.0em]
& = & \displaystyle \int_{z} p_\theta(x_i | z) p(z) \, dz.
\end{matrix}
$$

<br>

첫 번째 등식은 joint distribution $p(x_i, z)$를 $z$에 대해 marginalization 해준 것이고, 두 번째 등식은 조건부 확률의 정의를 사용한 것이다. $p_\theta(x_i | z)$가 단순한 분포라 하더라고 $p_\theta(x_i)$는 적분으로 표현되기 때문에 계산할 수 없는 대상이 된다. VAE에서는 $p_\theta(x_i | z)$을 가우시안 분포로 모델링한다. 

$$
p_\theta(x_i | z) = \mathcal{N}\left(\mu_\theta(z), \sigma^2_\theta(z)\right),
$$

<br>

where $\mu_\theta(z) \in \mathbb{R}^n$ and $\sigma_\theta(z) \in \mathbb{R}^n$. 
참고로 이는 $z$를 입력 받아서 2개의 벡터 $\mu_\theta(z) \in \mathbb{R}^n, \sigma_\theta(z)\in \mathbb{R}^n$를 출력하는 뉴럴 네트워크로 구현할 수 있다.
그리고 엄밀히 말하면 공분산 행렬은 $\sigma^2_\theta(z)$를 대각성분으로 갖는 대각행렬인데, 표기의 간결함을 위하여 저렇게 적었다. 다시 돌아가서 위의 적분을 조금 더 변형해보자.

$$
\begin{matrix}
p_\theta(x_i) 
& = & \displaystyle \int_{z} p_\theta(x_i | z) p(z) \, dz \\[1.0em]
& = & \displaystyle \int_{z} \frac{p(z | x_i)}{p(z | x_i)} p_\theta(x_i | z) p(z) \, dz \\[1.0em]
& = & \displaystyle \mathbb{E}_{z\sim p(z|x_i)} \left[ \frac{p_\theta(x_i | z) p(z)}{p(z|x_i)}\right]. \\[1.0em]
\end{matrix}
$$

<br>

두 번째 등식은 $p(z | x_i)$이 0이 아니라는 가정 하에 적분 안에 1을 곱해준 것 뿐이고, 세 번째 등식은 기댓값의 정의를 사용한 것이다. 양변에 로그를 취해보자.

$$
\begin{matrix}
\log p_\theta(x_i) 
& = & \displaystyle \log \mathbb{E}_{z\sim p(z|x_i)} \left[ \frac{p_\theta(x_i | z) p(z)}{p(z|x_i)}\right] \\[1.0em]
& \ge & \displaystyle \mathbb{E}_{z\sim p(z|x_i)} \left[ \log  \frac{p_\theta(x_i | z) p(z)}{p(z|x_i)}\right]. \\[1.0em]
\end{matrix}
$$

<br>

두 번째 부등식은 로그 함수가 위로 볼록하다는 성질 때문에 Jensen's inequality 만족하기 때문에 성립한다. 이제 겉보기에는 적분 계산이 사라졌고, $\log p_\theta(x_i)$가 기댓값으로 표현된다. 기댓값으로 표현되면 좋은 점은 다음과 같이 $z$를 $p(z|x_i)$에서 샘플링하여 표본평균으로 기댓값을 근사할 수 있다는 점이다.

$$
\begin{matrix}
\log p_\theta(x_i) 
& \ge & \displaystyle \mathbb{E}_{z\sim p(z|x_i)} \left[ \log  \frac{p_\theta(x_i | z) p(z)}{p(z|x_i)}\right] \\[1.0em]
& \approx & \displaystyle \frac{1}{K} \sum_{k=1}^{K} \log \frac{p_\theta(x_i | z_k) p(z_k)}{p(z_k|x_i)},
\end{matrix}
$$
where $z_k \sim p(z|x_i)$ for $k=1, 2, \ldots, K$. 

<br>

---

## Variational Inference 

여기서 문제는 posterior인 $p(z|x_i)$를 계산하는 것과 샘플링하는 것 모두 불가능하다. 따라서 우리는 $p(z|x_i)$을 보다 더 쉬운 분포로 근사를 할 것이다. 예를 들어, $x_i$에 dependent한 가우시안 분포로 근사할 수 있을 것이다. 이는 데이터 $x_i$를 입력 받아 가우시안 분포의 평균과 분산을 출력하는 뉴럴 네트워크로 구현 가능하다.

$$q_{\phi}(z|x_i)=\mathcal{N}\left(\mu_\phi(x_i), \sigma_\phi^2(x_i) \right),$$

<br>

where $\mu_\phi(x_i) \in \mathbb{R}^m$ and $\sigma_\phi(x_i) \in \mathbb{R}^m$. 이렇게 Bayesian inference를 할 때 계산이 불가능한 분포를 보다 더 쉬운 분포로 근사하여 inference를 하는 방법론을 variational inference라고 부르며, 근사에 사용되는 더 쉬운 분포인 $q_{\phi}(z|x_i)$를 variational distribution이라고 부른다. 그리고 VAE에서는 $q_{\phi}(z|x_i)$를 인코더라고 부른다.

$$
\begin{matrix}
\log p_\theta(x_i) 
& \ge & \displaystyle \mathbb{E}_{z\sim q_\phi(z|x_i)} \left[ \log  \frac{p_\theta(x_i | z) p(z)}{q_\phi(z|x_i)}\right]. \\[1.0em]
\end{matrix}
$$

<br>

이렇게 바로 $p(z|x_i)$ 대신 $q_{\phi}(z|x_i)$으로 갈아 끼울 수 있는 이유는 기댓값으로 유도할 때 적분 안에 $1=\frac{p(z|x_i)}{p(z|x_i)}$을 곱해주는 대신 $1=\frac{q_{\phi}(z|x_i)}{q_{\phi}(z|x_i)}$을 곱해주면 되기 때문이다. 위 부등식을 조금 더 분해해보자.

$$
\begin{matrix}
\log p_\theta(x_i) 
& \ge & \displaystyle \mathbb{E}_{z\sim q_\phi(z|x_i)} \left[ \log  \frac{p_\theta(x_i | z) p(z)}{q_\phi(z|x_i)}\right] \\[1.0em]
& = & \mathbb{E}_{z\sim q_\phi(z|x_i)} \left[ \log  p_\theta(x_i | z)\right] - \mathbb{E}_{z\sim q_\phi(z|x_i)} \left[ \log \frac{q_\phi(z|x_i)}{p(z)} \right]  \\[1.0em]
& = & \mathbb{E}_{z\sim q_\phi(z|x_i)} \left[ \log  p_\theta(x_i | z)\right] - D_{\text{KL}}\left[ q_\phi(z|x_i) \| p(z)\right].
\end{matrix}
$$

<br>

두 번째 항의 경우 $p(z)$를 가우시안 분포로 가정할 경우 $q_{\phi}(z|x_i)$도 가우시안 분포이기 때문에 KL divergence 공식이 존재하여 쉽게 계산할 수 있다. 첫 번째 항의 경우 $q_{\phi}(z|x_i)$에서 $z$를 샘플링하여 표본평균으로 기댓값을 근사시킨다.

$$
\mathbb{E}_{z\sim q_\phi(z|x_i)} \left[ \log  p_\theta(x_i | z)\right] \approx \frac{1}{K} \sum_{k=1}^{K} \log  p_\theta(x_i | z_k),
$$
where $z_k \sim q_\phi(z|x_i)$ for $k=1, 2, \ldots, K$.

<br>

---

## Variational Lower Bound

지금까지 우리는 다음과 같은 부등식을 얻었다. 부등식을 얻어서 어디에 사용할 것인가? 우리의 목표는 주어진 데이터셋 $\mathbf{X}$에 대해서 좌변 값들의 평균을 최대화하는 $\theta$를 찾는 것이다. 문제는 latent variable을 도입하여 $p_\theta(x_i | z)$를 간단한 분포로 가져가는 대신 좌변의 $p_\theta(x_i)$는 계산하지 못하게 되었다는 것이다.

$$
\log p_\theta(x_i) \ge \mathbb{E}_{z\sim q_\phi(z|x_i)} \left[ \log  p_\theta(x_i | z)\right] - D_{\text{KL}}\left[ q_\phi(z|x_i) \| p(z)\right].
$$

<br>

따라서 우리의 전략은 좌변을 최대화하는 대신 부등식의 우변인 lower bound를 최대화하는 것이다. Variational inference에서 이 lower bound를 variational lower bound라고 부른다. 이때, 의심 많은 사람이라면 다음과 같은 질문을 제기할 수 있다 (필자는 절대 의심하지 않았다).

<center>
<i>"Lower bound를 최대화한다고 해서 좌변이 최대화된다는 보장이 있어?"</i>
</center>

<br>

질문에 대한 답변을 하자면, 특정 조건에서 그렇다고 말할 수 있다. 그 특정 조건에서는 부등식이 아닌 등호가 성립하며, 등호가 성립하기 때문에 우변을 최대화하면 좌변도 같이 최대화된다. 그 특정 조건이 무엇이냐면, 바로 우리의 variational distribution $q_\phi(z|x_i)$와 실제 posterior $p(z|x_i)$ 사이의 KL divergence가 0이 될 때이다.

<br>

위에서는 부등식을 조금 더 쉽게 유도하기 위하여 Jensen's inequality를 사용했지만,  다시 한 번 등식만 사용하여 $\log p_\theta(x_i)$를 적어보자.

$$
\begin{matrix}
\log p_\theta(x_i)
 & = & \displaystyle \log p_\theta(x_i) \cdot \int_z q_{\phi}(z|x_i)  \, dz \\[1.0em]
& = & \displaystyle \int_z q_{\phi}(z|x_i) \log p_\theta(x_i)  \, dz \\[1.0em]
& = & \displaystyle \int_z q_{\phi}(z|x_i) \log \frac{p_\theta(x_i|z)p(z)}{p(z|x_i)} \, dz \\[1.0em]
& = & \displaystyle \int_z q_{\phi}(z|x_i) \log \frac{q_\phi(z|x_i) p_\theta(x_i|z)p(z)}{q_\phi(z|x_i) p(z|x_i)} \, dz \\[1.0em]
& = & \displaystyle \int_z q_{\phi}(z|x_i) \left( \log p_\theta(x_i|z) - \log \frac{q_{\phi}(z|x_i)}{p(z)} + \log\frac{q_{\phi}(z|x_i)}{p(z|x_i)}  \right) \, dz \\[1.0em]
& = & \displaystyle \mathbb{E}_{z\sim q_\phi(z|x_i)} \left[ \log  p_\theta(x_i | z)\right] - D_{\text{KL}}\left[ q_\phi(z|x_i) \| p(z)\right] + D_{\text{KL}}\left[ q_\phi(z|x_i) \| p(z|x_i)\right].
\end{matrix}
$$

<br>

첫 번째 등호는 확률분포의 적분 값은 1인 것을 이용한 것이고, 두 번째 등호는 적분 변수가 $z$이기 때문에 $\log p_\theta(x_i)$는 상수로 간주되어 적분 안으로 들어온 것이다. 세 번째 등호는 Bayes's theorem인 $P(A) = \frac{P(A|B) P(B)}{P(B|A)}$를 사용한 것이다. 네 번째 등호는 로그 안에 $1=\frac{q_\phi(z|x_i)}{q_\phi(z|x_i)}$을 곱해준 것이고, 다섯 번째 등호는 로그의 성질을 사용해서 관심 있는 항들로 분리해놓은 것이다. 마지막 등호는 KL divergence의 정의를 사용한 것이다.

<br>

Variational lower bound를 $\mathcal{L}_i(\theta, \phi)$라고 적어주면, $\log p_\theta (x_i)$는 다음과 같이 표현될 수 있다.

$$
\log p_\theta(x_i) = \mathcal{L}_i(\theta, \phi) + D_{\text{KL}}\left[ q_\phi(z|x_i) \| p(z|x_i)\right].
$$

<br>

KL divergence는 항상 양수 값을 갖는다. 따라서 우리가 이미 확인했던 부등식을 한 번 더 확인할 수 있다.

$$
\log p_\theta(x_i) \ge \mathcal{L}_i(\theta, \phi).
$$

<br>

만약 $D_{\text{KL}}\left[ q_\phi(z|x_i) \| p(z|x_i)\right] = 0$이면, 다음 등식이 성립한다.

$$\log p_\theta(x_i) = \mathcal{L}_i(\theta, \phi)$$

<br>

따라서 이때 우변을 최대화하면 좌변도 최대화가 된다. 그럼 똑똑한 사람이라면 다음과 같이 질문할 수 있다.

<center>
<i> 그럼 variational lower bound만 줄일 것이 아니라 KL divergence도 0이 되도록 파라미터 업데이트를 해야 하는 것 아닌가요? 하지만 실제 posterior는 못 구한다면서요? </i>
</center>

<br>

다시 한 번 등식을 보자.

$$
\log p_\theta(x_i) = \mathcal{L}_i(\theta, \phi) + D_{\text{KL}}\left[ q_\phi(z|x_i) \| p(z|x_i)\right].
$$

<br>

파라미터 $\phi$를 조절하여 variational lower bound $\mathcal{L}_i(\theta, \phi)$를 증가시키면, 등호가 성립하기 위해서 KL divergence 텀이 줄어들어야만 한다 (좌변은 $\phi$가 없기 때문에 값이 유지되기 때문이다). 즉, $\mathcal{L}_i(\theta, \phi)$을 최대화시키는 것만으로도 $D_{\text{KL}}\left[ q_\phi(z|x_i) \| p(z|x_i)\right]$가 줄어드는 효과가 있는 셈이다.

<br>

따라서 $\phi$에 대해서 $\mathcal{L}_i(\theta, \phi)$을 최대화하면 KL divergence가 0에 가까워질테고, KL divergence가 0에 가까우면 가까울수록 $\mathcal{L}_i(\theta, \phi)$이 실제 $\log p_\theta(x_i)$에 가까워지고, 이때 $\theta$에 대해서 $\mathcal{L}_i(\theta, \phi)$을 최대화하면 $\log p_\theta(x_i)$도 같이 커질 수 있다는 것이다.

<br>

---

## Reparameterization Trick

지금까지 내용의 결론은 아래의 variational lower bound를 최대화시킬 것이라는 것이다라는 것이다라는 것이다.

$$
\mathcal{L}_i(\theta, \phi)=
\mathbb{E}_{z\sim q_\phi(z|x_i)} \left[ \log  p_\theta(x_i | z)\right] - D_{\text{KL}}\left[ q_\phi(z|x_i) \| p(z)\right].
$$

<br>

### 디코더 $p_\theta(x_i|z)$ 학습

우선 디코더 $p_\theta(x_i|z)$의 파라미터 $\theta$는 첫 번째 항에만 있다. 기댓값으로 표현되는 첫 번째 항은 $z_k$를 인코더 $q_{\phi}(z|x_i)=\mathcal{N}\left(\mu_\phi(x_i), \sigma_\phi^2(x_i) \right)$로부터 샘플링하여 $\log p_\theta(x_i | z_k)$ 값을 계산하고 $\theta$에 대하여 backward를 진행하여 파라미터를 업데이트시키면 된다. 참고로 확률값은 디코더를 가우시안 분포로 모델링했기 때문에 가우시안의 PDF를 통해 계산할 수 있다. 

$$
J(\theta)=\frac{1}{K} \sum_{k=1}^{K} \log  p_\theta(x_i | z_k),
$$

<br>

where $z_k \sim q_\phi(z|x_i)$ for $k=1, 2, \ldots, K$.

<br>

### 인코더 $q_\phi(z|x_i)$ 학습

인코더 $q_\phi(z|x_i)$의 파라미터 $\phi$는 variational lower bound의 두 항 모두에 들어있다. 우선 두 번째 항의 경우 $q_\phi(z|x_i)$와 $p(z)$ 모두 가우시안 분포로 모델링하기 때문에 KL divergence 계산 공식을 따라 계산할 수 있다. 계산한 KL divergence를 $\phi$에 대해 backward하고 파라미터를 업데이트하면 된다.

<br>

문제는 첫 번째 텀이다. 첫 번째 텀의 기댓값의 확률변수 $z$가 $q_\phi(z|x_i)$에서 샘플링되기 때문이다. 하지만 이 기댓값을 직접 계산하는 것이 불가능하기 때문에 $q_\phi(z|x_i)$에서 $z_k$를 샘플링하여 표본평균으로 첫 번째 항을 근사시켰다:

<br>

$$
J(\phi)=\frac{1}{K} \sum_{k=1}^{K} \log  p_\theta(x_i | z_k),
$$

<br>

where $z_k \sim q_\phi(z|x_i)$ for $k=1, 2, \ldots, K$. 문제는 그냥 샘플링은 말 그대로 확률분포에서 무작위로 추출하기 때문에 $z_k$에 파라미터 $\phi$가 묻어 있지 않다 (연산 그래프가 유지되지 않는다는 의미). 샘플링을 한 순간 그냥 $z_k$는 값 그 자체이다. 따라서 지금 상태에서 $J(\phi)$를 $\phi$에 대해서 미분을 하면 그냥 0이 되어 버린다.

<br>

샘플링한 $z_k$의 계산 과정에 파라미터 $\phi$를 포함시키기 위하여 reparametrization trick이라는 것을 사용한다. 방법은 아주 간단한데,$q_{\phi}(z|x_i)=\mathcal{N}\left(\mu_\phi(x_i), \sigma_\phi^2(x_i) \right)$에서 바로 $z_k$를 샘플링하는 대신 $\mathcal{N}\left( \mathbf{0}, I \right)$에서 $\epsilon_k$를 샘플링하고, $z_k^{\phi}=\sigma_\phi(x_i) * \epsilon_k+ \mu_\phi(x_i)$으로 계산한다. 여기서 $*$는 elementwise곱이다. 이렇게 되면 $\log  p_\theta(x_i | z_k^\phi)$ 값 계산에 파라미터 $\phi$가 포함되게 되어 $J(\phi)$을 $\phi$에 대하여 역전파를 할 수 있게 된다.

<br>

---

## 학습 이후 데이터 생성

우리가 variational lower bound를 최대화하면서 한 것은 주어진 입력 이미지 $x$에 대해서 latent variable $z \sim q_\phi(z|x)$을 샘플링하고, 이를 디코더에 넣었을 때 다시 $x$가 나올 확률의 로그 값을 높여주는 방식이었다. 즉, $x$의 정보를 최대한 $z$에 인코딩하고 이를 다시 디코딩했을 때 원래 이미지 $x$가 나올 확률을 높인 것이다. 따라서 variational lower bound의 첫 번째 텀을 reconstruction term이라고도 부른다.

$$
\mathcal{L}_i(\theta, \phi)=
\mathbb{E}_{z\sim q_\phi(z|x_i)} \left[ \log  p_\theta(x_i | z)\right] - D_{\text{KL}}\left[ q_\phi(z|x_i) \| p(z)\right].
$$

<br>

한편, 데이터를 생성할 때는 $p(z)$에서 $z$를 하나 샘플링하여 이를 디코더 $p_\theta(\cdot|z)$에 입력하여 얻은 가우시안 분포 $\mathcal{N}\left(\mu_\theta(z), \sigma^2_\theta(z)\right)$에서 데이터를 생성한다. 더 이상 인코더는 사용되지 않는다. 이는 두 번째 항을 통해 $q_\phi(z|x_i)$를 $p(z)$와 최대한 비슷하게 만들어주었기 때문에 가능한 일이다. 훈련 데이터와 비슷한 그럴싸한 데이터가 나올 로그확률 값 $\log  p_\theta(x_i | z)$이 높아진 latent variable의 분포는 어디까지나 $q_\phi(z|x_i)$이었다. 따라서 만약  $p(z)$가 $q_\phi(z|x_i)$와 달랐더라면 그럴싸한 이미지가 나올 확률이 높다는 보장이 없었을 것이다.

<br>
