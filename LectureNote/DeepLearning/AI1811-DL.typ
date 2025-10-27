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




= Conclusion
