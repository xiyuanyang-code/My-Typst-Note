#import "@preview/dvdtyp:1.0.1": *
#show: dvdtyp.with(
  title: "Reinforcement Learning CS234",
  author: "Xiyuan Yang",
  abstract: [Lecture notes for reinforcement learning],
)
#show link: set text(fill: blue, weight: 700)
#show link: underline
#show ref: set text(fill: blue, weight: 500)


// comment for hiding outlines
#outline()

// uncomment for changing to a new page
// #pagebreak()

= Introduction

= Tabular MDP Planning

== Markov Reward Process

#definition("Markov Process")[
  $ P(X_(t_(n+1)) = x_(n+1) | X_(t_n) = x_n, dots, X_(t_1) = x_1) = P(X_(t_(n+1)) = x_(n+1) | X_(t_n) = x_n) $
]

We define $G_t$ with discount factor.

Definition of State value Function $V(s)$ as a MRP.

In MRP, we only have states, transitions, reward functions and discount factors. We have no actions. The transitions are computed all based upon probability.


=== Bellman Equation

$
  V(s) = underbrace(R(s), "Immediate Rewards") + underbrace(gamma sum_(s' in S) P(s'|s)V(s'), "Discounted Sum of future rewards")
$

- $V(s)$: expected reward from the beginning state $s$
- $R(s)$: Intermediate rewards
- $gamma$: discount factor
- $sum_(s' in S) P(s'|s)V(s')$: expected reward of the next state

#recordings("Why only next state?")[
  - 贝尔曼方程最神奇的问题在于只需要推演下一个时间状态的概率和期望值
  - 同时，我们保证 V 代表的是该策略下的最优决策，这本质上是一种递归的思想。因此我们并不需要把未来无穷时间步的 reward 全部计算出来。
  - 贝尔曼方程是马尔科夫决策状态下许多算法的基础，一般来说，只要认为状态是马尔科夫状态，则算法的关键在于正确的估计最优价值函数 $V(s)$
]

=== Finite Space Solving

For finite state of MDP, we can express $V(s)$ as a matrix or a tabular.

$
  vec(V(s_1), V(s_2), V(s_3), dots, V(s_n)) = vec(R(s_1), R(s_2), R(s_3), dots, R(s_n)) + gamma mat(
    P(s_1|s_1), P(s_2|s_1), ..., P(s_n|s_1);
    P(s_1|s_2), P(s_2|s_2), ..., P(s_n|s_2);
    dots.v, dots.v, dots.down, dots.v;
    P(s_1|s_n), P(s_2|s_n), ..., P(s_n|s_n);
  ) vec(V(s_1), V(s_2), V(s_3), dots, V(s_n))
$

很明显，对于有限状态 $n$ 的马尔科夫决策过程，本质上就是求解一个最基本的线性方程组：

$ bold(V) = bold(R) + gamma bold(P)^T bold(V) $

$ bold(V) = (bold(I) - gamma bold(P)^T)^(-1) bold(R) $

=== Infinite Space Solving

在实际情况中，矩阵求解的方法时间复杂度过于昂贵，并且实际情况下的状态总数 $n$ 会变得极大，这也就意味着模型需要极大的一个状态转移矩阵，这对计算资源的消耗是不可接受的。

因此，下面介绍使用动态规划的算法来实现：

$k$ means the iteration steps. (For the initial state, we purpose $V_0(s) =0$)

$
  V_k(s) = underbrace(R(s), "Immediate Rewards") + underbrace(gamma sum_(s' in S) P(s'|s)V_(k-1)(s'), "Discounted Sum of future rewards")
$

这实际上可以算作是一个优化的估计问题，因为我们事先无得知未来的状态的价值函数，因此我们需要多次迭代来毕竟最优的价值函数，当这个数收敛的时候，就代表估计成功了。

从数学上来看这一点：我们寻找的是最优的价值函数，满足贝尔曼方程：

$
  V^*(s) = max_a (underbrace(R(s,a), "Immediate Rewards") + underbrace(gamma sum_(s' in S) P(s'|s)V^*(s'), "Discounted Sum of future rewards"))
$

定义贝尔曼优化算子 $cal(T)$:

$
  cal(T)(V(s)) = max_a (underbrace(R(s,a), "Immediate Rewards") + underbrace(gamma sum_(s' in S) P(s'|s)V(s'), "Discounted Sum of future rewards"))
$

We hope:

$ V_k = cal(T)(V_(k-1)) $

#definition("压缩映射")[
  $ ||cal(T)V_a - cal(T)V_b||_infinity <= gamma ||V_a - V_b||_infinity $
]

而压缩映射可证明：保证有巴拿赫不动点！
而我们可以证明贝尔曼算子在 $gamma$ 小于 1 的时候是一个压缩算子。

For each iteration, the time complexity is $O(|S|^2)$

== Markov Decision Process

$ S, A, P, R, gamma $

Thus, the transition is: $P(s'|s, a)$

=== MDP Policies

$ pi(a|s) = P(a_t = a|s_t = s) $

It is a Stochastic Policy. MDP with $pi(a|s)$ is MRP.

#recordings("MDP and MRP")[
  - 在 MDP 中，我们加入了策略，即智能体会有一个在给定状态下的概率分布，而这个概率分布会让智能体做出相对应的决策。
  - 而一旦这个智能体的概率分布被固定，这个本质上就是一个MRP系统！因为最终都会化归到一个概率分布上。

  $ P(s'|s) = sum_(a in cal(A)) pi(a|s)P(s'|s,a) $
]

For MDP Iteration:

$
  V_k^pi (s) = sum_a pi(a|s) [R(s,a) + gamma sum_(s' in S) p(s'|s,a)V^pi_(k-1) (s')]
$




= Conclusion


