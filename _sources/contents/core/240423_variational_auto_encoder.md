# Variational Autoencoder (VAE)

맨날 헷갈리는 VAE 정리 =ㅅ=

$\mathbf{X}=\left\{ x_i \right\}_{i=1}^{N}$을 주어진 $N$개 데이터의 집합이라고 하자. 그리고 이 $N$개의 데이터는 어떤 알 수 없는 확률분포 $p$에 대한 i.i.d 샘플이라고 가정하자. 
우리의 목표는 뉴럴 네트워크를 사용하여 데이터를 생성시킨 확률분포 $p$를 근사하는 것이다.
따라서 뉴럴 네트워크로 근사시킨 확률분포를 $p_\theta$로 표기할 것이다.
우리의 maximum likelihood estimation (MLE) 목적함수는 다음과 같다.

$$\operatorname*{maximize}_\theta \frac{1}{N}\sum_{i=1}^{N} \log p_\theta(x_i).$$

<br>

---

## Latent variable model

여기서 우리는 latent variable model을 사용할 것이다. 데이터 $x_i$를 생성하는 데 

그럼, 위 목적함수에서 $p_\theta(x_i)$를 다시 적어줄 수 있다.

$$
\begin{matrix}
p_\theta(x_i) 
& = & \displaystyle \int_{z} p(x_i, z) \, dz \\[1.0em]
& = & \displaystyle \int_{z} p_\theta(x_i | z) p(z) \, dz.
\end{matrix}
$$

<br>

첫 번째 등식은 joint distribution $p(x_i, z)$를 $z$에 대해 marginalization 해준 것이고, 두 번째 등식은 Bayes's theorem을 사용한 것이다. 

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

두 번째 부등식은 로그 함수가 위로 볼록하다는 성질 때문에 Jensen's inequality 만족하기 때문이다. 이제 겉보기에는 적분 계산이 사라졌고, $\log p_\theta(x_i)$가 기댓값으로 표현된다. 기댓값으로 표현되면 좋은 점은 다음과 같이 $z$를 $p(z|x_i)$에서 샘플링하여 표본평균으로 기댓값을 근사할 수 있다는 점이다.

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

## Variation inference

여기서 문제는 posterior인 $p(z|x_i)$를 계산하는 것과 샘플링하는 것 모두 불가능하다. 따라서 우리는 $p(z|x_i)$을 보다 더 쉬운 분포로 근사를 할 것이다. 예를 들어, $x_i$에 depend한 가우시안 분포로 근사할 수 있을 것이다. 이는 데이터 $x_i$를 입력 받아 가우시안 분포의 평균과 분산을 출력하는 뉴럴 네트워크로 구현 가능하다.

$$q_{\phi}(z|x_i)=\mathcal{N}\left(\mu_\phi(x_i), \sigma_\phi^2(x_i) \right).$$

<br>

이렇게 Bayesian inference를 할 때 계산이 불가능한 분포를 보다 더 쉬운 분포로 근사하여 inference를 하는 방법론을 variational inference라고 부르며, 근사에 사용하는 더 쉬운 분포인 $q_{\phi}(z|x_i)$를 variational distribution이라고 부른다.

$$
\begin{matrix}
\log p_\theta(x_i) 
& \ge & \displaystyle \mathbb{E}_{z\sim q_\phi(z|x_i)} \left[ \log  \frac{p_\theta(x_i | z) p(z)}{q_\phi(z|x_i)}\right]. \\[1.0em]
\end{matrix}
$$

<br>

이렇게 바로 $p(z|x_i)$ 대신 $q_{\phi}(z|x_i)$으로 갈아 끼울 수 있는 이유는 기댓값으로 유도할 때 적분 안에 $1=\frac{p(z|x_i)}{p(z|x_i)}$을 곱해주는 대신 $1=\frac{q_{\phi}(z|x_i)}{q_{\phi}(z|x_i)}$을 곱해주는 것이기 때문에 위의 부등식이 성립한다. 위 부등식을 조금 더 분해해보자.

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

## Reparameterization trick.
Coming soon!