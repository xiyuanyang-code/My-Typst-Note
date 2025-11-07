#import "@preview/dvdtyp:1.0.1": *
#show: dvdtyp.with(
  title: "AI1811-Deep Learning",
  author: "Xiyuan Yang",
  abstract: [Lecture notes for deep learning AI-1811.],
)
#show link: set text(fill: blue, weight: 700)
#show link: underline
#show ref: set text(fill: blue, weight: 500)


// comment for hiding outlines
#outline()

// uncomment for changing to a new page
// #pagebreak()

= Introduction

== Lecture Overview

- Introduction
  - High dimensional data
  - 高维数据中的维数灾难
- Model Architecture
  - MLP
  - CNN
  - RNN
  - Transformer
- Advanced about deep learning
  - LLM reasoning
  - 凝聚现象


== NN and Polynomial

深度学习有万有逼近定理，多项式拟合也有 Weierstrass 逼近定理。

#recordings("The bitter lesson")[
  The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin.
]

在实际情况下，我们发现神经网络的过拟合现象远小于多项式拟合。

= High Dimensional Space

== 高维数据的空间特点

$
  x = [x_1, x_2, dots, x_d]^T in RR^d
$

=== 高维空间的稀疏性

Input n points in the cube in $d$ dimensional space, with $k$ sub-length: $k^d$

If we ensure that all the little cube has at least on data point, then wen must ensure:

$
  n >= k^d
$

The increase of $n$ is exponential!

#recordings("High dimensional data is very sparse")[
  For the sampling process in high dimensional space, the state space si extremely large.
]

=== 体积集中在表面

$Omega$ with high dimensional space for dimensional $d$, scale the $Omega$ with factor $epsilon$ for a little bit. We can get the inner part $(1-epsilon) Omega$.

$
  (V(1-epsilon)Omega)/(V(Omega)) = (1-epsilon)^d arrow 0
$

This will lead to the lack of points of inner parts while sampling, which will cause the generalization error for the inner part is extremely big.

=== 距离的集中效应

Randomly sample 2 spaces for point $x$ and $y$, then the distance can be written into:

$
  ||x - y||_2 = (sum^d_(i=1) (x_i - y_i)^2)^(1/2)
$

While d becomes extremely large, the distance will converge into $sqrt(2) R$.

- Angle will converge into $90$.
- Distance will converge into $R$ (in the interface).

#recordings("Space in a ball")[
  - 体积集中的表面
  - 体积集中在赤道
]

==== Gaussian Distribution in High Dimensional Space

For gaussian distribution $x ~ N(0, I_d)$

$
  p(x) = 1/((2 pi)^(d/2)) exp(-(||x||^2)/2)
$

#recordings("概率和概率密度")[
  - 高斯分布可以保证概率密度的分布仅是呈现正态分布的形式。
  - 但是具体的概率上概率密度的积分，而在高维空间下体积分布上不均匀的。
]

$
  F(r) = exp(-r^2/2) r^(d-1)
$

The maximum point is in $r = sqrt(d-1)$.

==== Word Embedding

Word Embedding for NLP: dimension reduction in high dimensional space.


=== 高维数据的线性可分性

线性可分性：空间中存在超平面，将不同类别的数据点完全分开。

- In kernelled functions, we will do a none-linear transformation for original low dimensional space data and embed them into high dimensional space. Then linear classifiers like SVM can be applied!

- In NN, the last layer will make the high dimensional data can be linear classified, which can be output and categorized for softmax.

== 维数灾难

Curse of dimensionality.

For example, the math expression for 2-layer neural network:

$
  f(x;theta) = sum^m_(k=1) a_k sigma (w^T_k x) = a^T sigma (W x)
$

= Recurrent Neural Network

循环神经网络相信状态随着时间的变化，因此训练可学习矩阵来表示隐状态：

$
  S_0 = 0\
  S_t = f(U x_t + W S_(t-1) + b), t = 1,2,3,dots,n\
  o_t = g(V s_t)
$

where $f$ is the activate functions, and g is the softmax functions for final output.

For each state, $S_t$ represents the memory statement for current state $t$.

From this statement, we can design *encoder-decoder* structure for machine translation.

== BackPropagation Through Time (BPTT)

$ frac(partial L_t, partial W_(h h)) = sum_(k=1)^t frac(partial L_t, partial h_t) dot (product_(j=k)^(t-1) frac(partial h_(j+1), partial h_j)) dot frac(partial h_k, partial W_(h h)) $

While doing back propagations, the computation of gradient requires many time steps multiplications, which will result to *Vanishing Gradient* (VG) or *Exploding Gradient* (EG).

== LSTM

We have a sequence: $x_1, x_2, dots, x_n$.

#figure(
  image("images/LSTM.png")
)

For current state $t$, we will do operations as follows:

- Input new knowledge:

$
  accent(C,"~")_(t+1)  = tanh(W_(s c) S_t + W_(x C) x_(t+1) + b_C)
$

- Choose to forget:

$
  F_t = sigma(W_(S f) S_t + W_(x f) x_(t+1) b_f)
$

- Choose to write into long-term memory:

$
  I_t = sigma(W_(S i) S_t + W_(x i) x_(t+1) + b_i)
$

Then, we can update our memory status:

$
  C_(t+1)  = I_t accent(C,"~")_(t+1) + F_t C_t
$

Now, we consider the output gate:

$
  s_(t+1) = o_t * tanh(C_(t+1))
$

$
  o_t = sigma(W_(s o) S_t + W_(x o) x_(t+1) + b_o)
$

#recordings("Why two state?")[
  Why we design the internal memory state and hidden state?
  - 记忆门的控制代表着模型的输入信息如何更新到长期的记忆中，我们可以从公式中看到，记忆门的状态矩阵的更新是不需要通过激活函数的
  - 隐藏状态在 LSTM 中代表浅层记忆，控制输入和输出，并且每一步都会重新计算，快速改变
]

$f_t in [0,1]$. Thus it could control the gradient in a relative bounds.

$
  (partial L)/(partial C_t) = (partial L)/(partial C_(t+1)) * (partial C_(t+1))/(partial C_t)
$

$
  (partial C_(t+1))/(partial C_t) = f_t in [0,1]
$

Thus, we can reduce VG and EG!



