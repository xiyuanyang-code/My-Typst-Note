#import "@preview/dvdtyp:1.0.1": *
#show: dvdtyp.with(
  title: "AI1811 Machine Learning",
  author: "Xiyuan Yang",
  abstract: [Machine Learning Courses for AI1811, simple machine learning.],
)
#show link: set text(fill: blue, weight: 700)
#show link: underline
#show ref: set text(fill: blue, weight: 500)


// comment for hiding outlines
#outline()

// uncomment for changing to a new page
// #pagebreak()

= Introduction

== Course Content

- Supervised Learning
  - Regression
    - KNN-based regression
    - linear regression
    - least squares
    - gradient descent
  - Classification
    - Decision Trees
    - Logistic regression
    - Support Vector Machine
- Dimensional Reduction
  - PCA
  - Locally Linear Embeddings
- Clustering
  - K-means
  - Expectation-maximization
  - K-means++
- Model Selection and evaluation
  - Overfitting
  - L1/L2 regularization
  - K-fold cross-validation
- MLE / MAP
  - Likelihood
    - Given the output, predict the likelihood of the parameters.
    - $P(x|theta)$: probability
    - $L(theta|x)$: Likelihood
    - $P(x|theta) = L(theta|x)$
    - generated from the probability distribution: $f(x, theta)$
  - Maximum likelihood estimation (MLE)
    - 观测问题，通过统计学来反推结果。
  $ hat(theta) = text("argmax")_theta space L(theta|x) $
  - Maximum a posteriori (MAP)
    - Add a prior
  $ hat(theta) = text("argmax")_theta space L(theta|x) P(theta) $

== Introduction to ML

=== Basic Concepts

#definition("Basic Concepts for ML")[
  - feature/attribute
  - feature/attribute space
  - sample space for the data set
  - label space for output
  - *generalization*
    - learning process is the searching and evaluating process during the *hypothesis space*.
    - inductive bias
      - 机器学习算法对一个假设空间中假设的偏好
      - 学习的关键在于形成正确的假设偏好
      - 奥卡姆剃刀
        - 归纳偏好很容易陷入局部过拟合中
        - 具体问题具体分析
]


=== No Free Lunch Theorem

“奥卡姆剃刀”本身在理论上并不严格成立，即 期望性能相同。

$ sum_f E_(text("ote"))(cal(L)_a|X,f) = sum_f E_(text("ote"))(cal(L)_b|X,f) = 2^(|cal(X)|-1) sum_(x in cal(X) - X) P(x) $

- $cal(X)$ 代表着样本空间
- $cal(L)_a$ 代表着不同的算法
- $X$ 代表着训练数据
- $f$ 代表着目标函数

= Chapter 2 Model Selection

== OverFitting

=== Cross Validation

Given the dataset $D$, splitting the dataset into:

$ D = D_1 union D_2 union D_3 union dots union D_k, D_i inter D_j = emptyset $

- 注意分层采样

- 使用 k 折交叉验证，分别使用不同的 $D_i$ 作为最终的测试集，返回平均的测试结果。 

=== Boot Strapping

返回采样 $m$ 次，概率保证有一定量的数据不会被采样到。这样的估计方法被称作 out-of-bag estimate.

== Evaluation

#definition("MSE Error")[
  $ E(f;D) = 1/m sum_(i=1)^m (f(x_i) - y_i)^2 $
]

=== Precision and Recall

#definition("Precision and Recall")[
  $ P = (T P)/(T P + F P) $
  $ R = (T P)/(T P + F N ) $
]

We need a tradeoff!

- P-R graph
  - 改变分类阈值而画出的曲线
- BEP Point: where P = R
- F1 Score

$ F_1 = (2P R)/(P + R) $
$ F_beta = ((1 + beta^2) P R)/((beta^2 P) + R) $

- $beta > 1$: focus more on Recall（查全率）
- $beta < 1$: focus more on precision

=== ROC and AUC



= Conclusion
