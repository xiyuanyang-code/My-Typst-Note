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

$
  sum_f E_(text("ote"))(cal(L)_a|X,f) = sum_f E_(text("ote"))(cal(L)_b|X,f) = 2^(|cal(X)|-1) sum_(x in cal(X) - X) P(x)
$

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

在预测问题场景下，模型往往输出 $[0,1]$ 之间的一个浮点数来表示输出的概率大小，最终设置一个阈值来表达具体的分类。阈值的选取往往和实际场景和期望性能有关。例如如果更关心查全率，阈值就会更低，如果更关心查准率，阈值就会更高。因此，单纯阈值的设置往往需要根据 P-R 曲线来做具体的 tradeoff，无法表示该模型在一般任务下的泛化性能。

如何衡量一般任务下的泛化性能？我们给出 ROC 和 AUC 的定义指标。

#definition("ROC")[
  Receiver Operating Characteristic
  $ text("TPR") = text("TP")/(text("TP") + text("FN")) $
  $ text("FPR") = text("FP")/(text("TN") + text("FP")) $
]

和 PR 曲线类似，ROC 曲线也是根据实际的混淆矩阵和设置不同的阈值就可以得到 ROC 曲线。根据公式，TPR 和 FPR 的值本身就反应了分类器是否成功进行了分类。我们希望 TPR 尽可能的大，而 FPR 尽可能的小，即图上的 (0,1) 点。


= Chapter 3 Linear Regression

== Introduction to linear regression

- 最基本的多维度线性回归，就是两个向量的点积操作。

- 考虑并行化处理，我们可以使用矩阵实现并行计算。

$ y = X w^TT $

- 其中矩阵 $X in text("number of samples") times text("dimension")$

- 权重向量被每一个样本点所复用

#recordings("Regression is the basis")[
  - 回归和分类的关系：
    - 可以使用单热编码编码成高维的向量
    - softmax
  - 回归和排序的关系
    - 直接将排序问题转化为回归，最小化真实相关度分数 $y_i$ 与预测分数 $hat(y)_i$ 之间的均方误差
    - 使用 pointwise 的方法或者 pairwise 的方法。
  - 回归于密度估计的关系
    - 在高斯假设下的回归模型，本质上是对条件概率分布的参数建模：
    $ y | x ~ cal(N)(x^TT w, sigma^2) $
    - 此处最大似然估计等价于最小均方化误差
]

== Hyperparameter Space

对于一般化的线性回归模型，我们可以通过偏置矩阵消去偏置项，方便训练和计算。

$ hat(y) = X w + b_1 $

$ X' = [1, X] in RR^(n times (d+1)) $

$ w' = mat(b; w) $

$hat(y) = X' w'$

== MSE

在#box[线性回归]中，最常见的优化目标是#box[最小化均方误差 (Mean Squared Error, MSE)]：

$
  cal(L)(bold(w)) = 1/n norm(bold(y) - bold(X) bold(w))^2
$

MSE 可等效写为：

$
  cal(L)(bold(w)) = 1/n (bold(y) - bold(X) bold(w))^T (bold(y) - bold(X) bold(w))
$

通过对损失函数求梯度并令其为零，可以得到#box[闭式解 (普通最小二乘 OLS)]：

$
  bold(w)^* = paren(bold(X)^T bold(X))^(-1) bold(X)^T bold(y)
$

(当 $bold(X)^T bold(X)$ 可逆时)

#recordings("MAE and MSE")[
  $ text("MAE") = 1/n sum^(n)_(i=1) |y_i - hat(y_i)| $

  $ text("RMSE") =sqrt(1/n sum^(n)_(i=1) (y_i - hat(y_i))^2) $

  - MAE 对异常值不敏感
  - MSE 对异常值非常敏感，使用于 Loss 比较大的雨花算法求解。
]

== KNN

#definition("Intuition for KNN")[
  - 一种非训练的基于实例的非参数化方法。
  - 在预测阶段开始计算，懒惰学习
  - 最终根据距离加权 *K* 个最近的样本点。
  - 计算复杂度高（对于多次查询），并且对异常值敏感。
]

== Least Square

在#box[线性回归]中，最常见的优化目标是#box[最小化均方误差 (Mean Squared Error, MSE)]：

$
  cal(L)(bold(w)) = 1/n norm(bold(y) - bold(X) bold(w))^2
$

MSE 可等效写为：

$
  cal(L)(bold(w)) = 1/n (bold(y) - bold(X) bold(w))^T (bold(y) - bold(X) bold(w))
$

通过对损失函数求梯度并令其为零，可以得到#box[闭式解 (普通最小二乘 OLS)]：

$
  (partial cal(L)(w))/(partial w) = 2/n X^T (X w - y) = 0
$

$
  X^T X w = X^T y
$

$
  bold(w)^* = paren(bold(X)^T bold(X))^(-1) bold(X)^T bold(y)
$

(当 $bold(X)^T bold(X)$ 可逆时)

#recordings("几何意义")[
  - 最小二乘法是在参数空间找寻找一个 $d$ 维度的超平面，并且保证所有样本点在做投影时尽可能的接近该平面。
  - $X beta$ 是 X 的列向量的线性组合，或者说张成了 X 的列空间上。
  - 或者说，最小二乘法等价于将目标向量 $y$ 投影到 $X$ 的列空间上，投影向量正是 $X beta$ 而残差向量与列空间正交
  $ X^T (y - X beta) = 0 $

]


== Regularization

=== Ridge Regression

增加 L2 正则化，惩罚参数量过大的情况。

$ min_beta || y - X beta ||^2_2 + lambda ||beta||^2_2 $

在增加 L2 正则化之后，其对应的理论最优解也发生了偏移：

$ hat(beta) = (X^T X + lambda I_p)^(-1) X^T y $

#recordings("Explanation")[
  在 OLS 中，我们最小化的是残差平方和，其等值线是椭圆形。而 正则项 $||beta||^2 <= t$ 对应的约束区域是一个超球体（在二维为圆形）。Ridge 回归等价于在椭圆等值线与圆形约束的交线上寻找最优点，由于圆形约束会均匀压缩系数，使得 Ridge 回归倾向于得到较小但非零的系数。
]

=== Lasso (Least Absolute Shrinkage and Selection Operator)

- 使用 L1 正则化

#definition("Lasso")[
  $
    hat(beta) = arg min_beta paren(1/(2n) norm(y - X beta)_2^2 + lambda norm(beta)_1)
  $
]

#recordings("Lasso")[
  在 L2 正则化是，约束区域是一个球形区域，但是 L1 正则化约束是 L1 范数的菱形区域。
]


=== Elastic Net
设训练数据为

$
  bold(X) in RR^(n times p), bold(y) in RR^n
$

其中 $n$ 为样本数，$p$ 为特征数，参数向量 $beta$ 为 $RR^p$，截距为 $beta_0$。

Elastic Net 的优化目标函数为：

$
  hat(beta) = arg min_beta paren(
    1/(2n) norm(bold(y) - bold(X) beta)_2^2 + lambda (
      alpha norm(beta)_1 + (1 - alpha)/2 norm(beta)_2^2
    )
  )
$

其中：

- $norm(beta)_1 = sum_(j=1)^p abs(beta_j)$ 为 L1 范数 (Lasso)；
- $norm(beta)_2^2 = sum_(j=1)^p beta_j^2$ 为 L2 范数平方 (Ridge)；
- $lambda >= 0$ 为整体正则化强度超参数；
- $alpha in [0, 1]$ 控制 L1 与 L2 的比例：
  - $alpha = 1$ 时退化为 Lasso；
  - $alpha = 0$ 时退化为 Ridge；
  - $0 < alpha < 1$ 时为 Elastic Net 的混合正则化。



= Conclusion

