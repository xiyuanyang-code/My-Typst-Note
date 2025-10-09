#import "@preview/dvdtyp:1.0.1": *
#import "@preview/physica:0.9.6": *
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
    - Support accenttor Machine
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
  cal(L)(bold(w)) = 1/n (bold(y) - bold(X) bold(w))^TT (bold(y) - bold(X) bold(w))
$

通过对损失函数求梯度并令其为零，可以得到#box[闭式解 (普通最小二乘 OLS)]：

$
  bold(w)^* = paren(bold(X)^TT bold(X))^(-1) bold(X)^TT bold(y)
$

(当 $bold(X)^TT bold(X)$ 可逆时)

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
  cal(L)(bold(w)) = 1/n (bold(y) - bold(X) bold(w))^TT (bold(y) - bold(X) bold(w))
$

通过对损失函数求梯度并令其为零，可以得到#box[闭式解 (普通最小二乘 OLS)]：

$
  (partial cal(L)(w))/(partial w) = 2/n X^TT (X w - y) = 0
$

$
  X^TT X w = X^TT y
$

$
  bold(w)^* = paren(bold(X)^TT bold(X))^(-1) bold(X)^TT bold(y)
$

(当 $bold(X)^TT bold(X)$ 可逆时)

#recordings("几何意义")[
  - 最小二乘法是在参数空间找寻找一个 $d$ 维度的超平面，并且保证所有样本点在做投影时尽可能的接近该平面。
  - $X beta$ 是 X 的列向量的线性组合，或者说张成了 X 的列空间上。
  - 或者说，最小二乘法等价于将目标向量 $y$ 投影到 $X$ 的列空间上，投影向量正是 $X beta$ 而残差向量与列空间正交
  $ X^TT (y - X beta) = 0 $

]


== Regularization

=== Ridge Regression

增加 L2 正则化，惩罚参数量过大的情况。

$ min_beta || y - X beta ||^2_2 + lambda ||beta||^2_2 $

在增加 L2 正则化之后，其对应的理论最优解也发生了偏移：

$ hat(beta) = (X^TT X + lambda I_p)^(-1) X^TT y $

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

$ y = 1/(1 + e^(-w^TT x + b)) $

考虑上面式子中的 y 为一个后验概率：

$ p(y = 1|x) = 1/(1 + e^(-w^TT x + b)) $

可以得到：

$ ln p(y = 1|x)/(y = 0|x) = w^TT x + b $

在概率中使用极大似然法来估计参数：

$ cal(l)(w,b) = sum^(m)_(i=1) ln p(y_i|x_i; w,b) $

即令每个样本属于其真实标记的概率越大越好。为便于讨论，令 $beta = (w; b)$,
$hat(x) = (x; 1)$, 则 $w^TT x + b$ 可简写为 $beta^TT hat(x)$. 再令 $p_1(hat(x); beta) = p(y=1 | hat(x); beta)$,
$p_0(hat(x); beta) = p(y=0 | hat(x); beta) = 1 - p_1(hat(x); beta)$, 则式(3.25)中的似然项可重写为


$ p(y_i | x_i; w, b) = y_i p_1(hat(x)_i; beta) + (1-y_i)p_0(hat(x)_i; beta) $


将式(3.26)代入(3.25)，并根据式(3.23)和(3.24)可知，最大化式(3.25)等价于最小化

$ cal(l)(beta) = sum_(i=1)^m (-y_i beta^TT hat(x)_i + ln(1+e^(beta^TT hat(x)_i))) $

该式子根据经典的凸优化理论，可以求得其理论解，从理论上证明了其可导性！

== LDA

经典的使用投影的思想解决问题：

给定高维数据集 $D = ((x_i, y_i))^(m)_(i=1)$:

- 将一个数据点投影到一条直线 $w$：
  - 高维空间下的直线：$r(t) = r_0 + t accent(v, arrow)$
    - 其中 $r_0$ 代表的固定点 P，为一个高维向量
    - $accent(v, arrow)$ 代表一个方向向量，也是高维的
  - 在这里默认直线经过原点（为了后续简化计算），因此可以使用一个向量代表直线
  - 投影点的坐标 $w^TT mu_0$
- 在上文的例子中 $mu_0$ 代表着均值向量
  - 考虑协方差 $w^TT sum_0 w$

因此得到最大化的目标：

$ J = (||w^TT mu_0 - w^TT mu_1||^2_2)/(w^TT sum_0 w + w^TT sum_2 w) $

$ S_b = (mu_0 - mu_1)(mu_0 - mu_1)^TT $

$ S_w = sum_0 + sum_1 $

#recordings("Sb and Sw")[
  - 我们的直觉是同类数据点之间尽可能靠近
    - 在分母上放协方差作为惩罚项
  - 不同类别的中心尽可能远离
    - 以均值中心为代表，衡量投影的二范数（距离）作为分子

  对于多分类任务同样如此：

  - 类内散度矩阵被定义为所有类别散度接矩阵的和，每一个类别散度矩阵通过协方差衡量数据点的分布关系：

  $ S_W = sum^K_(k=1) S_k $

  $ S_k = sum_(x in C_k) (x - mu_k)(x - mu_k)^TT $

  - 类间散度矩阵为每个类别的均值相对于所有样本的全局均值的分散程度。它反映了不同类别中心之间的分离程度。
]


== 多分类学习

多分类学习可以从二分类学习的方法中进行升维并且一般化，但是也可以有一些更通用的策略：

基本思路：将多分类学习的任务拆解为若干个二分类任务求解。

- Splitter
- Classifier

=== OvO

对于 $N$ 个类别的多分类任务，采用 $(N(N-1))/2$ 个二分类任务进行组合。我们训练这么多分类器之后就可以实现分类任务就可以通过正反例的方式进行分类

=== OvR

训练 $N$ 个分类器，将一个类别标记为正类，其余类别标记为反类

=== MvM

相当于分治法划归为子问题，但是其正反类需要特殊的设计

==== ECOC

纠错输出码 Error Correcting Output Codes

- 编码：对于 $N$ 个类别做 $M$ 次划分
  - 形成编码矩阵 $in N times M$
  - 编码矩阵可以是二元码也可以是三元码
- 解码：分别对输出样本做预测
  - 因此解码的时候可以得到一个预测输出向量
  - 计算输出向量和编码矩阵的每一行（代表对一个训练样本数据每一个预测器的判断）的向量比较并计算距离
    - Hamming Distance
    - Euclidean Distance

ECOC 的关键在于选择的 $M$ 会很多，远大于 $log_2 N$ 的最低下限值。因此这会带来比较空间的冗余（因为维度上去了），但是这也提升了鲁棒性。

- 基础二分类器的选择可以自由选择。
- OvR 可以看做是一种特殊的 ECOC，但是 OvR 对应的编码矩阵的鲁棒性很小

== 类别不平衡问题

对于实际 GT 分布存在严重不平衡的样本，对于二分类器的训练带来了难度：

可以使用除了 Precision 之外的其他指标。（这是 evaluation 阶段做的修改）

对于二分类器而言，也可以在 inference 阶段实现优化：

- 对于均分样本，我们一般比较输出的概率大小来输出最终的值，即阈值设定为 0.5

$ y/(1-y) > 1 $

- 对于非均衡样本，可以使用偏斜的预测阈值

$ y/(1-y) > m^+/m^- $

具体而言，我们可以对模型的输出值再加一个*Rescaling*，即：

$ y'/(1-y') = m^-/m^+ y/(1-y) $

#recordings("Rescaling")[
  - 在理想情况下，这个方法是理论可行的，但是这个的前提是 训练集必须是真实样本的无偏估计的理想假设
  - 这一个假设在偏斜数据集的场景下更难满足
  - 具体的，也可以使用欠采样或者过采样的方式使模型的正反例的样本数基本相同
    - 欠采样：使用集成学习机制，将反例划分为不同的集合供学习器使用，类似于 Kfold 的思想
    - 过采样：插值增加采样数据
]

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

从简单情况出发，考虑某一个离散属性 $a in A$, which has several possible values: $(a^1, a^2, dots, a^V)$, then it will split the current tree into $V$ new branches, then the input $D$ is split into $V$ partitions: $D_1, D_2, dots, D_V$.

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

= Chapter 6: SVM (Support accenttor Machine)


= Chapter 7: Naive Bayes


= Ensemble Learning


= Clustering

聚类需要把样本集 $D$ 划分成若干互不相交的子集，即样本簇。

== Evaluation for Clustering

Simply, for K-means clustering:

$
  {C_1, dots, C_k} = "argmin" sum^K_(k=1) sum_(x_i in C_k) ||x_i - mu_k||^2
$

=== External Index

在给定参考模型的聚类结果，我们可以检查每一个数据点的聚类结果是否和 references model 相同来判断具体聚类的优劣。具体来说，两两配对得到 $(n(n+1))/2$ 的点对，考虑每个点对是否被正确的分类到相同或者不同的聚类中。

#figure(
  image("ML/clustering1.png"),
)

根据这些，可以推导一些聚类性能的外部指标。

=== Internal Index

Define these metrics:

$
  "avg"(C) = 2/(|C|(|C|-1)) sum_(1 <= i < j <= |C|) "dist"(x_i, x_j)
$

$
  "diam"(C) = max_(1 <= i < j <= |C|) "dist"(x_i, x_j)
$

$
  d_min (C_i, C_j) = min_(x_i in C_i, x_j in C_j) "dist"(x_i, x_j)
$

$
  d_"cen" (C_i, C_j) = "dist"(mu_i, mu_j)
$


#recordings("All these metrics")[
  - 这些指标无外乎实现两个优化目标：
    - 不同簇之间要相隔尽量明显: inter-cluster similarity
    - 相同簇之间要尽可能紧密: intra-cluster similarity
  - 对于数据点的距离度量很重要。
    - 使用公理化的方法定义 distance measure
  $
    "dist"_"mk" (x_i, x_j) = (sum^n_(u=1) |x_(i u) - x_(j u)|^p)^(1/p)
  $
  - distance metric learning
]

== Prototype Based Clustering

=== K-Means

=== K-Means++

=== Expectation–Maximization/GMM

== Density-Based Clustering

=== DBS-CAN



= Dimension Reduction

== Linear Dimension Reduction

考虑更加一般的线性降维方法，对于待降维的矩阵 $X = (x_1, x_2, dots, x_m) in RR^(d times m)$

我们需要构造一个线性变换矩阵 $W in RR^(d times d')$

而降维的过程就是简单的线性变换的矩阵乘法：$Y = W^TT X in RR^(d' times m)$

#recordings("Why Linear?")[
  - 线性变换矩阵的列向量是降维后新坐标系的基向量
  - 数据之间的线性关系得以保留
  - 更特殊的情况，如果施加的是正交变换：
    - 基向量构成了一组标准正交基
    - 形成了低维的标准正交子空间
    - 而正交矩阵本质上就是 rotation 操作！因此带来了保持距离和角度的良好性质。
]

下面将逐一介绍一些很经典的线性降维和非线性降维的方法。

=== Multiple Dimensional Scaling

定义原始距离矩阵 D 代表 $m$ 个样本的原始空间的距离矩阵，每个元素代表两个特定数据点的距离表示。我们构建的目标是构建降维后的 m 个样本点的坐标，而保持任意两个样本的欧氏距离保证等于原始空间的距离。（距离矩阵作为不变量）

考虑样本在低维空间中的表示：$Z in RR^(d' times m)$，计算内积矩阵 $B = Z^TT Z in RR^(m times m)$

$b_(i,j) = z_i^TT z_j$

考虑降维后的样本被中心化，则内积矩阵具有良好的性质：每一行和每一列都是和为0的。

#figure(
  image("ML/MDS.png"),
  caption: [MDS Proof],
)

#recordings("MDS")[
  - 求解 MDS 的过程首先求解内积矩阵
  - 保证距离的不变形这就会导致内积矩阵的很多完美性质
    - 最终可以保证内积矩阵的每一个元素都被不变量的距离所表示
    - 这是内积所带来的！
  - 求出内积矩阵之后再使用特征值分解
]

求得内积矩阵 B 之后，很显然这并不是一个满秩矩阵，因此可以做矩阵的满秩分解。在这里我们做特征值分解：
#let diag = math.op("diag")
$ B = V diag(lambda_1, lambda_2, dots, lambda_d) V^TT $


假设其中有 $d^*$ 个非零特征值（或者在实际运算，可以手动截断前几个最大的特征值），那么 Z 就可以表达为：

$ Z = sqrt(diag(lambda_1, lambda_2, dots, lambda_(d^*))) V_*^TT $


=== PCA

我们希望寻找一个超平面：

- 最近重构性：样本点到这个超平面的距离足够近
- 最大可分性：样本在超平面上的投影离的足够远

==== Prof1 for PCA

#figure(
  image("ML/PCA_1.png"),
)

下面我们来对上式进行具体证明：

考虑降维前已经归一化的样本 $accent(x_i, arrow)$，我们希望降维后的样本为 $accent(hat(x_i), arrow)$

For the linear transformation:

$ hat(X) = W^TT X $

- 其中 $W^TT$ 是有标准正交基组成的，并且维度为无损压缩的最大上限，这也是矩阵的秩。
- 但是在实际建模的过程中，我们选择 $d' <= d$

$ accent(z_i, arrow) = (z_(i 1), z_(i 2), dots, z_(i d')) = W^TT accent(x_i, arrow) $

$
  accent(hat(x_i), arrow) = sum^(d')_(j=1) z_(i j) accent(w_j, arrow) = sum^(d')_(j=1) accent(w_j^TT, arrow) accent(x_i, arrow) accent(w_j, arrow) = WW^TT x_i
$

$ x_i in RR^d $
$ W = (w_1, w_2, dots, w_(d')) in RR^(d times d') $ (新坐标系基向量), 且 $ W^TT W = I_(d') $ (标准正交基)。
$ z_i = W^TT x_i in RR^(d') $ (降维坐标), 其中 $ z_( ) = w_j^TT x_i $。
$ hat(x)_i = sum_(j=1)^(d') z_(i j) w_j = W z_i = W W^TT x_i $
重构误差平方和 $L$：
$ L = sum_(i=1)^(m) norm(hat(x)_i - x_i)^2_2 = sum_(i=1)^(m) norm(sum_(j=1)^(d') z_(i j) w_j - x_i)^2_2 $

利用二范数展开式
根据 $norm(a-b)^2_2 = norm(a)^2_2 - 2 a^TT b + norm(b)^2_2$，其中 $a=hat(x)_i$ 且 $b=x_i$。
$ L = sum_(i=1)^(m) norm(hat(x)_i)^2_2 - 2 hat(x)_i^TT x_i + norm(x_i)^2_2 $

最后一项 $sum_(i=1)^(m) norm(x_i)^2_2$
$ sum_(i=1)^(m) norm(x_i)^2_2 = "const" $

第二项 $sum_(i=1)^(m) hat(x)_i^TT x_i$
代入 $hat(x)_i = W W^TT x_i$ 并利用 $ (W W^TT)^TT = W W^TT $ (对称性)：
$
  sum_(i=1)^(m) hat(x)_i^TT x_i = sum_(i=1)^(m) (W W^TT x_i)^TT x_i = sum_(i=1)^(m) x_i^TT (W W^TT)^TT x_i = sum_(i=1)^(m) x_i^TT W W^TT x_i
$

第一项 $sum_(i=1)^(m) norm(hat(x)_i)^2_2$
代入 $hat(x)_i = W W^TT x_i$ 并利用 #text(weight: "bold")[$ W^TT W = I $] (标准正交性)：

$
  sum_(i=1)^(m) norm(hat(x)_i)^2_2 = sum_(i=1)^(m) hat(x)_i^TT hat(x)_i
  = sum_(i=1)^(m) (W W^TT x_i)^TT (W W^TT x_i)\
  = sum_(i=1)^(m) x_i^TT (W W^TT)^TT (W W^TT) x_i
  = sum_(i=1)^(m) x_i^TT (W W^TT W W^TT) x_i\
  = sum_(i=1)^(m) x_i^TT W (underbrace(W^TT W, I)) W^TT x_i
  = sum_(i=1)^(m) x_i^TT W W^TT x_i
$

将 A、B、C 三项组合回 $L$:
$ L = sum_(i=1)^(m) (underbrace(x_i^TT W W^TT x_i, "项 C")) - 2 (underbrace(x_i^TT W W^TT x_i, "项 B")) + "const" $
合并：
$L = sum_(i=1)^(m) - x_i^TT W W^TT x_i + "const" = - sum_(i=1)^(m) x_i^TT W W^TT x_i + "const"$

利用迹的性质：
1. 标量 $a = text(tr)(a)$。
2. 循环置换性质：$text(tr)(A B C) = text(tr)(B C A) = text(tr)(C A B)$。
对于每一项 $x_i^TT W W^TT x_i$ (它是一个标量)：
$ x_i^TT W W^TT x_i = text(tr)(x_i^TT W W^TT x_i) $
利用循环置换性质 (将 $x_i^TT$ 挪到最后，将 $x_i$ 挪到最前)：
$ text(tr)(x_i^TT W W^TT x_i) = text(tr)(W^TT x_i x_i^TT W) $
因此，总和可以写成：
$ sum_(i=1)^(m) x_i^TT W W^TT x_i = sum_(i=1)^(m) text(tr)(W^TT x_i x_i^TT W) $
由于迹和求和运算的线性性质，可以交换顺序：
$ = text(tr) W^TT sum_(i=1)^(m) x_i x_i^TT W $

将此结果代回 $L$:
$ L = - text(tr) W^TT sum_(i=1)^(m) x_i x_i^TT W + "const" $
因此，最小化重构误差 $L$ 等价于：
$ min_W - text(tr) W^TT sum_(i=1)^(m) x_i x_i^TT W $
这等价于最大化负号后面的项：
$ max_W text(tr) W^TT sum_(i=1)^(m) x_i x_i^TT W quad text(s.t.) W^TT W = I $


==== Prof 2 for PCA

从第二个角度来推导主成分分析，我们可以考虑投影样本点的方差最大化，而这个的数据表征就是协方差矩阵。

$ sum = ("Cov"(x_i, x_j))_(i,j) $

对于原来的矩阵，投影到一个方向上可以得到一个向量，向量的每一个元素代表该方向的值：

$ y = w^TT X $

投影后数据的方差计算，注意样本已经被归一化：

$ "Var"(y) = EE[(y - EE[y])^2] = EE[y^2] $

$ "Var"(y) = 1/m sum^m_(i=1) y_i^2 = 1/m sum^m_(i=1) (w^TT x_i)^2 = 1/m sum^m_(i=1) (w^TT x_i) (w^TT x_i)^TT $

$ "Var"(y) = 1/m sum^m_(i=1) w^TT (x_i x_i^TT) w = w^TT (1/m sum^m_(i=1) x_i x_i^TT) w $

$ C = 1/m sum^m_(i=1) x_i x_i^TT = "Cov"(X) $

- 注意上面的是指只对中心化的矩阵成立！

因此，从两种优化目标出发，PCA 最终得到了同一个一般的约束条件和最小化目标！

对于具体的求解过程，我们只是需要求解协方差矩阵 $X^TT X$ 的特征值分解并取最大的若干特征值对应的归一化的特征向量。对于实对称矩阵，其特征向量天然的保证正交性。


== Non-Linear Dimension Reduction

下面介绍若干非线性降维的方法，即从高维空间向低维空间的函数映射是非线性的。

=== Why Non-Linear?

理论上，降维是一个高度泛化的任务，数据点的分布会影响函数映射的具体形式，在某些场景下简单的线性降维难以压缩到低维空间。

线性降维本质是通过一个矩阵 $W$ 施加线性变化，在空间上即寻找数据的最佳线性投影（一个超平面或者子空间），他的基本假设是数据在高维空间中是近似线性的或可以被一个直线/平面很好地表示。

经典的例子，线性降维无法通过投影的手段把一个瑞士卷数据展开，而是会发生严重的重叠和混淆。因此，下文介绍若干非线性的函数映射。

=== Kernelized PCA

==== Kernelization and Kernel Function

The intuition of *Kernelization* is to find the non-linear mapping from original data samples into a new space $cal(F)$, ensuring the new data samples $phi(x)$ can be split by a linear bound!

#definition("kernel function")[
  $
    K(x_i, x_j) = phi(x_i)^TT phi(x_j)
  $

  - 直接计算内积相似度，无需耗费时间复杂度计算原始的映射
  - 核函数最大的用处是将线性算法核化为非线性算法
]


对于线性投影：

$
  (sum^m_(i=1) z_i z_i^TT) w_j = lambda_j w_j
$

$
  w_j = 1/lambda_j (sum^m_(i=1) z_i z_i^TT)w_j = sum^m_(i=1) z_i (z_i^TT w_j)/lambda_j = sum^(m)_(i=1) z_i alpha_i^j
$

$alpha_i^j$ 代表着第 $i$ 个数据点在构造第 $j$ 个主成分的时候的权重。

- 对于一般的线性 PCA，$z_i$ 就代表着样本点在高维特征中的像，需要做一次归一化。
- 对于核化 PCA，相当于加了一层函数映射映射到好被线性映射的空间 $cal(F)$, $z_i = phi(x_i)$。

因此：

$
  (sum^m_(i=1) phi(x_i) phi(x_i)^TT) w_j = lambda_j w_j
$

$
  (sum^m_(i=1) bold(k)(x_i, x_i)) w_j = lambda_j w_j
$

$
  w_j = sum^(m)_(i=1) phi(x_i) alpha_i^j
$

下面，我们来推导其矩阵形式：

$
  (sum^m_(i=1) bold(k)(x_i, x_i)) (sum^(m)_(k=1) phi(x_k) alpha_k^j) = lambda_j (sum^(m)_(k=1) phi(x_k) alpha_k^j)
$

展开：

$
  sum^m_(i=1) sum^m_(k=1) phi(x_i) phi(x_i)^TT phi(x_k) alpha_k^j = lambda_j (sum^(m)_(k=1) phi(x_k) alpha_k^j)
$

合并到核函数并且左乘 $phi(x_l)^TT$:

$
  sum^m_(i=1) sum^m_(k=1) K(x_i, x_k) K(x_l, x_i) alpha_k^j= lambda_j sum^(m)_(k=1) K(x_l, x_k) alpha_k^(j)
$

转化为矩阵形式：

$
  (K^2 alpha^j)_l = lambda_j (K alpha^j)_l
$

因为 $l$ 的选择是任意的，因此可以推广：

$
  K^2 alpha^j = lambda_j K alpha^j
$

在实际情况下，为了更方便，我们求解下面的简洁形式：

$
  bold(K) alpha^j = lambda_j alpha^j
$

回到最终 KPCA 的求解目标：

$
  z_j = w_j^TT phi(x) = sum^(m)_(i=1) alpha^j_i phi(x_i)^TT phi(x) = sum^(m)_(i=1) alpha^j_i K(x_i, x)
$

#recordings("Kernel Function")[
  - 核化方法的最精彩的地方就在于不求解复杂的函数映射 $phi$，而是转化为求解核化矩阵，因为最终还是划归到内积求解相似度的场景上！
  - 具体的核函数可以手动选择一些非线性的核函数，例如高斯核
  $
    K(x_i,x_j) = "exp"(- gamma ||x_i - x_j||^2)
  $
]

=== Manifold Learning

虽然数据在高维空间中看起来非常复杂，但它很可能内嵌（嵌入）在一个低维的、弯曲的“表面”上，这个“表面”就是流形（Manifold）。

流形学习中，我们希望解卷这个高维的流型，类似于把一个瑞士卷拉直，找出数据的低维内在结构。

==== Isomap

等度量映射的关键在于在流型中使用欧式空间度量距离会产生较大的误差。但是，我们可以利用流型在局部上与欧式空间同胚的性质，把距离的度量转化为近邻连接图上最短路径的问题。具体实现可以使用 Dijkstra Algorithms.

在得到距离后，在利用 MDS 算法保持距离不变形进行线性压缩。

==== Locally Linear Embeddings (LLE)

保持每个坐标点可以被领域样本线性组合而重构。

$
  x_i = w_(i j) x_j + w_(i k)x_k + w_(i l)x_l
$

并且这些权重的和为1。

具体来说，可以变成这个凸优化问题：

$
  min_(w_1, w_2, dots, w_m) sum^m_(i=1) norm((x_i - sum_(j in Q_i) w_(i j) x_j))\
  "s.t." sum^m_(i=1) w_(i j) =1
$

上述问题存在闭式解。

LLE 在低维空间中保持 $w_i$ 不变，并求解下面的优化目标：

$
  min_(z_1, z_2, dots, z_m) sum^m_(i=1) norm((z_i - sum_(j in Q_i) w_(i j) z_j))
$


矩阵形式表达：

#figure(
  image("ML/lle.png"),
  caption: [LLE Algorithms],
)


= Metric Learning


= Conclusion

