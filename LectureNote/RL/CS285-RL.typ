#import "@preview/dvdtyp:1.0.1": *
#show: dvdtyp.with(
  title: "Deep Reinforcement Learning",
  author: "Xiyuan Yang",
  abstract: [Lecture Notes for Deep Reinforcement Learning, CS285],
)
#show link: set text(fill: blue, weight: 700)
#show link: underline
#show ref: set text(fill: blue, weight: 500)


// comment for hiding outlines
#outline()

// uncomment for changing to a new page
#pagebreak()

= Introduction


== Works to Cover

1. From supervised learning to decision making
2. Model-free algorithms: Q-learning, policy gradients, actor-critic
3. Model-based algorithms: planning, sequence models, etc.
4. Exploration
5. Offline reinforcement learning
6. Inverse reinforcement learning
7. Advanced topics, research talks, and invited lectures

== Introduction to RL

=== Supervised Learning

Given $D = {(x_i, y_i)}$, we want the supervised learning systems to learn how to predict $y$ from $x$: $f(x) approx y$.

It usually assumes:

- *i.i.d* data.(独立同分布)
- known ground truth outputs in training

For example, Deep Learning for Image Recognitions/Classifications. (*Need High-Labeled Data*)

=== Reinforcement Learning

- Data is not i.i.d: previous outputs influence
- Ground truth answer is not known, only know
if we succeeded or failed, more generally, we know the reward

#recordings[
  强化学习对数据的利用更加的松弛，不需要高质量人工标注的数据，这提升了强化学习的上限，但是这也导致模型对数据的利用率较低。
]

In the mathematical view:

- goal for supervised learning: $f_theta(x_i) = y_i$
  - training data ${(x_i, y_i)}$ are fixed and manually labeled.
- goal for reinforcement learning: learning $pi_theta: s_t arrow a_t$ to maximize $sum_t r_t$
  - the data $(s_1, a_1, r_1, dots, s_T, a_T, r_T)$: own actions, dynamic!

==== Applications

- games, robotics
- RL with Large Language Models 
- RL with image generations
- RL for chip design

=== Deep Reinforcement Learning

Supervised Learning has the upper-bound has labeled data, but RL does not.

“Move 37” in Lee Sedol AlphaGo match: reinforcement learning “discovers” a move that surprises everyone.

- Data Driven AI (learns about the real world from data, but doesn’t try to do better than the data)
- Reinforcement Learning (optimizes a goal with emergent behavior, but need to figure out how to use *at scale*).

Combination: *Deep Reinforcement Learning*!

#recordings("The Bitter Lesson")[
  We have to learn the bitter lesson that building in how we think we think does not work in the long run.
]

- Data without optimization doesn’t allow us to solve new problems in new ways.
- Optimization without data is hard to apply to the real world *outside of simulators*.

The core components (two general building blocks for AI-systems):

- Learning: use data to extract patterns (world laws), understanding the world
- Search: Use computations to extract inferences. Making inferences and leverages that understanding for emergence.

#recordings[
  We have a brain for one reason and one reason only -- that's to produce adaptable and complex movements. Movement is the only way we have affecting the world around us… I believe that to understand movement is to understand the whole brain.
]


=== Sequential Decision Making

Far more than a convex optimization problem!

- Learning reward functions from example (inverse reinforcement learning)
- Transferring knowledge between domains (transfer learning, meta-learning)
- Learning to predict and using prediction to act

real-time reward is hard to design.

- Learning from demonstrations
  - Directly copying observed behavior
  - Inferring rewards from observed behavior (inverse reinforcement learning)
- Learning from observing the world
  - Learning to predict
  - Unsupervised learning
- Learning from other tasks
  - Transfer learning
  - Meta-learning: learning to learn

Will RL be the way to AGI? (using a *general learning algorithms* for interacting observations and actions with the environment)

= Supervised Learning of Behaviors

- policy based on observations: $pi_theta (a_t|o_t)$
- policy based on full observations: $pi_theta (a_t|s_t)$
- policy are distributions (probability)

We can form the bayes net.

#figure(
  image("images/supervised-learning.png")
)

#recordings("Markov Properties")[
  If you get the state $s_t$, then that is all you need to compute future state, and $s_1, s_2, dots, s_(t-1)$ does not matter.

  This gives the properties of the state.
]

== Imitation Learning

Target: Given the labeled data, trying to learn the $pi_theta (a_t|o_t)$ by supervised learning. Given the $o_t$ and $a_t$, it forms the training data.

It is a kind of *behavior cloning*.

这样的问题在于模型只会模仿先验的正确答案，而一旦预测出现微小的偏差，这一部分的偏差就会不断的方法，导致在多个时间步后模型的状态发生较大的偏移。

例如，自动驾驶的三个前置摄像头可以保证一定的鲁棒性的提升。

Core reason: *i.i.d* assumption does not work!

- Data augmentation for Training Data
- Algorithms Change
- Multi-Task Learning

=== Theory

For the training loops, the distributions for the training data is $p_"data" (o_t)$, which is different from the distributions of the testing environment $p_(pi_theta)$. Thus when trained model encountered new observations that not appears in the training set, it will cause the bias.

So how can we define a good policy? Assume *GT behavior* for expert is deterministic if given the whole observations $s_t$. (Just for simplify), and we can define the cost functions:

$
  c(s_t, a_t) = cases(0 " if " a_t = pi^* (s_t), 1 " otherwise")
$

And our goal is to minimize:

$
  EE_(s_t ~ p_(pi_theta)(s_t)) [c(s_t, a_t)]
$

注意！这里在训练是的分布就是 $pi_theta$ 相当于模型直接在训练过程中进行分布的采样，目标是最小化策略在自身轨迹上执行动作与专家动作不一致的概率。

- This is some kind of *Dataset Aggregation*.
- *Attention*! It is the state not the observations! We need to use the Markov properties for state.

Assume supervised learning works:

$
  pi_theta (a != pi^*(s)|s) <= epsilon, forall s in cal(D)_"train"
$

在训练数据分布下，模型动作和专家动作不一致的概率不超过 $epsilon$.

$
  E[sum_t c(s_t, a_t)] <= epsilon T + (1-epsilon)(epsilon(T-1) + (1-epsilon))(dots)) = O(epsilon T^2)
$

This will leads to the cascading errors with many many time steps. (For the worse case.)

More generally:

$
  p_theta (s_t) = (1-epsilon)^t p_("train")(s_t) + (1 - (1-epsilon)^t) p_"mistake" (s_t)
$

For the main distributions $p_"mistake"$, we don't see them in the training data.

$
  |p_theta (s_t) - p_"train" (s_t)| = (1 - (1-epsilon)^t) |p_"mistake" (s_t) - p_"train" (s_t)| <= 2 (1 - (1 - epsilon)^t) <= 2 epsilon t
$

Thus:

$
  sum_t P_(p_theta (s_t)) [c_t] = sum_t sum_(s_t) p_theta (s_t) c_t (s_t) &<= sum_t sum_(s_t) p_"train" (s_t) c_t (s_t) + |p_theta (s_t) - p_"train" (s_t)| c_max \
  &<= sum_t epsilon + 2 epsilon t
$

#recordings[
  - In reality, we can recover from mistakes.
  - A paradox: imitation learning can work better if the data has more mistakes (and recoveries)!

  - The imitation learning:
    - Teach the models when the models are on the right way.
    - *Teach the models to recover when the models are outside the right way*.
]

=== Data Augmentation

- Intentionally add mistakes and corrections (The mistakes hurt, but the corrections help, often more than the mistakes hurt)

- Use *data augmentation*. (e.g. side-facing cameras.), add some "fake" data that illustrates corrections.


=== More Powerful Models

#recordings[
  - Non-Markovian behavior ($pi_theta (a_t|o_1, o_2, dots, o_t)$)
    - Using histories
  - Multimodal behavior
]

==== Using Histories

Using the sequence model (LSTM, transformers.)

However, learning from history may cause confusions: (ICLR-2019 Best Paper)

Behavior Cloning will only learn about the correlations, but not the *cause and effect*, which is mortal in autonomous driving. (*casual confusion*)

Solutions: data augmentation & diffusion models.


==== Multimodal behavior

- mixture of Gaussians

$
  pi(a|o) = sum_i w_i cal(N)(mu_i, sum_i)
$

More specifically, it can be written into:

$
  pi(a|o) = sum_(k=1)^(K) pi_k (o) N(a|mu_k (o), sum_k (o))
$

$mu_k (o)$ and $sum_k (o)$ means they are functions of given input observations, but not sharing the same global parameters.

- latent variable models (conditional variant auto-encoder)

#recordings[
  Latent variable models（潜变量模型）在模仿学习中的核心原理是：用一个低维的、简单分布的潜变量 z，把专家的多模态连续动作分布 $p(a|s)$变得“可解析、可高效采样、可无限表达”。
]

$
  pi (a|s) = integral p(a|z,s) p(z|s) "d"z approx integral cal(N) (a|f_theta (z,s), delta^2 II) cal(N) (z|g_phi (s,a), sum) "d"z
$

- diffusion models

Diffusion models for image generations: $f(x_i) = x_i - x_(i-1)$

For imitation learning:

- $a_(t,0)$ is the true actions
- $a_(t, i+1) = a_(t,i) + "noise"$
- $a_(t, i-1) = a_(t,i) - f(s_t, a_(t,i))$

- Autoregressive discretization (like the sequence language models)

Do predictions: $p(a_(t,i)|s_t, a_(t,1),a_(t,2),dots,a_(t,(i-1)))$ (discretize one dimension at a time)

To multiply together: $p(a_t|s_t)$, like the complexity in the model distributions.

#recordings[
  - 使用自回归分解来逐步离散化高维空间的正确性在于，只要维度之间存在任何相关性，自回归分解都能表达它。
  - 并且自回归分解就像 token 生成一样，是一个信息不断增益的过程，不会对原有的信息造成影响。并且这样做离散化可以极大的减少状态空间的个数 $(K^D arrow K D)$
  - 类似于语言模型逐 token 的生成，不限制句子的长度实现了最终的 adaptive compute 并且每一次输出的维度限制在了词表的维度。
]

=== Multi-Tasking Learning

==== Goal Conditioned Behavioral Cloning

At the training time, even if the expert trajectories fails to achieve the optimal state $s_T$, it may reach the sub-optimal state, and all we want is to maximize it:

The goal state is to learn: $pi_theta (a|s, g)$

For each demo, the trajectory ${s_1^i, a_1^i, dots, s_(T-1)^i,a_(T-1)^i, s_T^i}$, we can maximize $log pi_theta (a_t^i|s_t^i, g=s^i_T)$

=== DAgger

DAgger: DataSet Aggregations

The critical error for behavior cloning is the distributional shift problem caused by the difference of $p_"data"$ and $p_theta$, thus, we need to do some augmentation:

*Enhancing the robustness of learned policy: Let the learned policy to make faults, and ask human experts to correct them in a fixed error state.*

For the specific data:

- Train the policy $pi_theta (a_t|o_t)$ from the original human datasets $D = {o_1, a_1, dots, o_N, a_N}$

- run the trained policy to get dataset $D_pi = {o_1, o_2, dots, o_M}$.

- Ask human to label the $D_pi$ with actions that should be correctly operated.

- Aggregate: $D = D union D_pi$

What is the problem: Stage3: *Human Annotations*.

== Limitations of Behavioral Cloning

- Machines just clone and learn from the human actions, but not the experience.

  - We need curated designed *lost functions* and *reward functions*.

- Human Annotations are not suitable for all categories of tasks.






= Conclusion