# AlphaGo Zero

- 제목: Mastering the Game of Go without Human Knowledge
- 저자:Silver, David, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, et al, *DeepMind*
- 링크: [https://doi.org/10.1038/nature24270](https://doi.org/10.1038/nature24270)
- 키워드: AlphaGo Zero

---

AlphaGo Zero에는 크게 두 가지 과정이 있다.
1. 자기 자신과 실제 바둑을 두는 단계인 self-play 단계
2. Self-play 단계에서 각 수를 두기 위해 바둑을 시뮬레이션해보는 단계 MCTS 

Self-play는 에이전트가 자기 자신과 바둑을 한 수씩 두는 단계이다. 한 self-play에서의 상태들의 sequence를 $s_0, s_1, s_2, \ldots, s_T$라고 하자.
각 수를 어디에 둘지는 현재 상태에서 MCTS로 시뮬레이션을 돌려보고 결정된다. 먼저 MCTS를 정리해보자

## Monte Carlo Tree Search (MCTS)

- Self-play의 각 상태 $s_t$에서 MCTS를 이용하여 시뮬레이션이 돌아간다. 한 시뮬레이션에서 1,600번 바둑을 둔다. 이 시뮬레이션으로부터 다음 행동을 샘플링할 분포인 $\mathbf{\pi}_t$가 만들어진다. 취할 수 있는 행동은 $19 \times 19 + 1 = 392$ 가지이다 (+1은 턴 넘기기).
- 시뮬레이션 트리의 루트 노드는 $s_t$이다. 트리의 각 노드 $s$에는 각 행동마다 엣지가 있는 상태이다. 각 엣지마다 4개의 statistics가 저장되어 있다.
    
    $$
    \left\{ Q(s, a), W(s, a), N(s, a), P(s, a) \right\}
    $$
    

### Select

- 각 노드 $s$에서 UCB을 사용하여 행동을 선택한다.
    
    $$
    a=\operatorname{argmax} \left( Q(s,a)+U(s,a) \right), \quad U(s,a)=c_{\text{puct}}P(s,a)\frac{\sqrt{\sum_b N(s,b)}}{(1 + N(s,a))},
    $$
    
    이때, $c_{\text{puct}}$는 탐색 정도를 결정하는 하이퍼 파라미터이다.
    
- 루트 노드에서 시작하여 위의 방법으로 행동을 선택하는 것을 $L$번 반복하자.

### Expand and evaluation

- Select를 통해 만든 하나의 트리의 리프 노드를 $s_L$이라 하자. 우선, $f_{\theta}(s_L) = (\mathbf{p}, v)$를 계산해준다.
- $s_L$에 각 행동마다 엣지를 달아주고, 엣지의 statistics를 다음과 같이 초기화해주자 ($s_t = s_0$일 때, 그 이후로는 만들었던 트리를 사용한다).
    
    $$
    \left\{ Q(s, a)=0, W(s, a)=0, N(s, a)=0, P(s, a)=p_a \right\},
    $$
    
    이때, $p_a$는 $\mathbf{p}$의 $a$번 째 원소이다.
    

### Backup

- $s_L$부터 시작하며 모든 엣지에 대한 statistics을 업데이트해준다.
    - $N(s, a) = N(s, a)+1$
    - $W(s, a) = W(s, a) + v$
    - $Q(s, a) = \frac{W(s, a)}{N(s,a)}$

### Play

- Backup이 끝나면 $\pi(a|s_t)=\frac{N(s_t, a)^{1/\tau}}{\sum_b N(s_t,b)^{1/\tau}}$을 기반으로 행동한다.

## Self-play

self-play로 $s_0, s_1, s_2, \ldots, s_T$만들고, 이겼는지 졌는지 판단해서 $r_T \in \left\{-1 ,1 \right\}$ 설정. 이 $r_T$는 모든 $t$에 대해서 $z_t$가 된다. $(s, \mathbf{\pi}, z)$가 데이터가 된다. $f_{\theta}(s_t) = (\mathbf{p}, v)$일 때, 다음 손실 함수를 사용해서 파라미터 업데이트

$$
(z-v)^2 + \mathbf{\pi}^\top\log\mathbf{p}+c\lVert \theta\rVert^2
$$

- 각 입력 $s$는 $19 \times 19 \times 17$ 텐서. 나의 돌만 표시한 행렬과 상대방의 돌만 표시한 행렬 각각 과거 8개 그래서 총 16개. 그리고 마지막 하나는 내가 검은돌이면 1이고 흰돌이면 0.