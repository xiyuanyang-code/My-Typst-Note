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
    $ y|x ~ cal(N)(x^TT w, sigma^2) $
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

== Classification

#recordings("Regression and CLassification")[
  - Classification 是输出到离散的标签点
  - Regression 是输出到连续的实数值
  - 从输入维度来讲，都是有标签数据
]

$ text("Acc") = 1/N sum^N_(i=1) LL(hat(p)_i = y_i) $

但是在模型训练的过程中，需要更精确的损失函数，使用二进制交叉熵损失。

=== Logistic Regression

有关逻辑回归的基本算法内容跳过，下面证明其函数在可导性上的良好性质：

考虑一个基本的逻辑回归：

$ y = 1/(1 + e^(-w^T x + b)) $

考虑上面式子中的 y 为一个后验概率：

$ p(y = 1|x) = 1/(1 + e^(-w^T x + b)) $

可以得到：

$ ln p(y = 1|x)/(y = 0|x) = w^T x + b $

在概率中使用极大似然法来估计参数：

$ cal(l)(w,b) = sum^(m)_(i=1) ln p(y_i|x_i; w,b) $

即令每个样本属于其真实标记的概率越大越好。为便于讨论，令 $beta = (w; b)$,
$hat(x) = (x; 1)$, 则 $w^T x + b$ 可简写为 $beta^T hat(x)$. 再令 $p_1(hat(x); beta) = p(y=1 | hat(x); beta)$,
$p_0(hat(x); beta) = p(y=0 | hat(x); beta) = 1 - p_1(hat(x); beta)$, 则式(3.25)中的似然项可重写为


$ p(y_i | x_i; w, b) = y_i p_1(hat(x)_i; beta) + (1-y_i)p_0(hat(x)_i; beta) $



将式(3.26)代入(3.25)，并根据式(3.23)和(3.24)可知，最大化式(3.25)等价于最小化

$ cal(l)(beta) = sum_(i=1)^m (-y_i beta^T hat(x)_i + ln(1+e^(beta^T hat(x)_i))) $

该式子根据经典的凸优化理论，可以求得其理论解，从理论上证明了其可导性！

= Chapter 4 Decision Tree

决策树的 Intuition 非常的简单，通过树状结构来模拟人类决策的 Options，是一种很自然的模拟机制。最终，当决策过程达到最大深度的叶子结点时，就代表一个最终的分类结果。同时，这也是一种简单而直观的“分而治之”的策略。但是决策树非常容易陷入在训练数据的过拟合中，我们希望产生一颗泛化能力强的决策树。

#figure(
  image("ML/decision-tree.png"),
  caption: [Simple algorithm for decision trees],
)

上述算法是一个经典的递归算法，他的基本思想如下，递归的终点就是叶子结点的生成位置：
- 当输入样本全部属于一个标签类别的时候，说明无需继续分类
  - 直接根据该标签类别生成一个叶节点
- 所有样本在属性集合上的取值相同（或者样本的属性集合为空）：此时无法做出有效的划分，停止递归
  - 需要标记为叶节点，属性是输入样本中样本数最多的类
- 划分的样本集合为空，无法进行划分，停止递归。
  - 根据父节点的样本数最多的类

因此，该算法的关键在于在每一次计算过程中找到 *the best partition properties* 代表着树进行一次分叉的属性。

== Partition Selection

我们可以把基于决策树的分类任务当成是一种提纯的操作。如何定义纯度？信息熵！这就是决策树的基本原理：

$ text("Ent")(D) = - sum^(|cal(Y)|)_(k=1) p_k log_2 p_k $

从简单情况出发，考虑某一个离散属性 $a in A$, which has several possible values: ${a^1, a^2, dots, a^V}$, then it will split the current tree into $V$ new branches, then the input $D$ is split into $V$ partitions: $D_1, D_2, dots, D_V$.

我们希望这是一个有效的划分而不是一个随机的划分，因此我们定义信息增益并希望最大化信息增益：

$ text("Gain")(D,a) = text("Ent")(D) - sum^(V)_(v = 1) (|D^v|)/(|D|) text("Ent")(D^v) $

因为我们希望 Gain 是正相关于我们的选择好坏的，而选择越好，信息熵越小，因此其实这是一个*信息熵的减少量*。

#definition("增益率")[
  - 信息增益确实很好的衡量了熵的变化，但是以为这是一个加权平均数，因此这对可取值数目较多的属性有所偏好。
  - 因此，我们需要定义一个更准确的增益率。

  $ text("IV")(a) = - sum^(V)_(v =1) (|D^v|)/(|D|) log_2 (|D^v|)/(|D|) $

  $ text("Gain Ratio")(D,a) = (text("Gain")(D, a))/("IV"(a)) $

]

== Pruning

树算法一般都会涉及到剪枝操作。这也是决策树学习算法主要应对过拟合的手段之一。

=== PrePruning

预剪枝的处理操作发生在树的生成过程中，我们需要评估模型的泛化性，如果模型的这一次 selection 没有办法带来泛化性能的提升，就停止剪枝，作为叶子结点。

如何评判泛化性？可以考虑验证集，评判划分和不划分在验证集的精度上是否有上升或者下降的趋势。

- 这样的预剪枝处理可以有效的除了训练数据的偏差，并且防止树的层数过深导致问题
- 过度粗暴的预剪枝也会带来欠拟合的风险。

=== PostPruning

后剪枝是在决策树已经生成之后再实现的，如果减去这个节点，是否会带来泛化性能的提升？如果有，就一切从简。

- 因为后剪枝是在决策树生成之后进行的，因此所需要计算的节点更多
- 但是往往不会像预剪枝一样落入贪心的陷阱而进入欠拟合。

== 连续值

我们希望保证连续属性的离散化，在经典的决策树 C4.5 中，采用二分法实现连续属性的决策树处理。

因为最终的样本数据肯定是离散的，因此选择划分点可以选择每两个相邻训练数据点的中点。

== 缺失值

- 我们只能使用不包含缺失值的数据进行计算
- 因此我们需要引入合适的系数来修正信息的增益


#figure(
  image("ML/missing.png"),
  caption: [对缺失值的处理],
)


== 多变量决策树

让我们从高维空间的视角看看决策树到底在干什么？

假设每一个数据点 $(x_i, y_i)$，其中 $y_i$ 代表输出的离散分类，那这其实是一个在高维度的空间下的数据点，而分类问题的关键在于找到某个约束方法或者约束条件，实现对未见过的数据点的正确分类。

单变量决策树保证在每一个节点只会考虑一个属性作为切分，这就导致了单变量决策树的分类边界是和坐标轴平行的。通过这一道道的分类边界将高维空间切割成正正方方的若干块。

因此，对于多变量决策树，每一次寻找的目标不再是最优划分的属性，而是建立一个合适的 *linear classifier*，这样的分类器可以实现更复杂的阈值划分，进而简化决策树的深度。


$ "Gain"(D,a) = rho "Gain"(tilde(D), a) $

= Naive Bayes

= SVM (Support Vector Machine)

= Conclusion

