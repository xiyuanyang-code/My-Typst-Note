# 机器学习课程 — 第二节课大纲（Regression, 90分钟）

## 1. 为什么学习回归（10分钟）

### **回归的重要性**

回归（Regression）作为监督学习的一种基本形式，在统计建模与机器学习中占据着核心地位。它的本质是建立自变量（特征）与因变量（目标）之间的关系模型，从而能够**预测连续数值**并解释变量之间的依赖关系。学习回归的重要性主要体现在以下几个方面：

#### 1. 预测能力强，应用广泛

回归模型不仅能够对未来的数值进行预测，还能在不同领域解决大量实际问题：

- **经济与金融**：预测股价、利率、销售额、GDP 增长趋势
- **医疗健康**：预测患者病程、血压水平、药物剂量响应
- **工业与制造**：预测机器寿命、生产效率、能耗水平
- **气候与环境**：预测气温变化、降水量、空气质量指数

实例场景：某电商公司可以通过回归模型，根据广告投放量、商品价格及季节性因素来预测下个月的销售额，从而优化库存和营销策略。

#### 2. 解释变量关系，辅助决策

与仅提供分类结果的模型相比，回归输出的是连续数值，并且模型中**系数的符号与大小可以解释因素之间的相关性和影响程度**：

- 线性回归中的系数 $\beta_i$ 表示自变量每变化一个单位，因变量的变化量
- 可以通过显著性检验（如t检验、p值）判断哪些变量对目标变量有显著影响

例：在房地产价格预测中，回归模型可以识别出房屋面积、交通便利程度对价格影响较大，从而为购房者和开发商提供参考。

#### 3. 构建更复杂模型的基础

回归是许多复杂机器学习方法的出发点：

- 多项式回归、岭回归、Lasso等是对基本线性回归的扩展
- 神经网络、梯度提升树等非线性模型中，局部结构依旧基于回归思想
- 时间序列分析中的ARIMA模型也基于回归原理

学习回归可以帮助理解更高阶算法的参数更新机制和模型优化思路。

#### 4. 处理连续型输出问题的首选方法

在机器学习问题类型中，回归是**连续型目标变量预测**最直接、最自然的选择：

- 分类任务预测类别，回归任务预测数值
- 可以通过调整损失函数（如均方误差MSE、平均绝对误差MAE）优化模型性能

数学形式上，回归问题可表示为：
$$
y = f(\mathbf{x}) + \epsilon
$$
其中，$y$ 为目标输出，$\mathbf{x}$ 为特征向量，$f$ 为真实关系函数，$\epsilon$ 为噪声项。

#### 5. 案例示例

**例子：房价预测**
假设我们有以下特征：

- $x_1$: 房屋面积（平方米）
- $x_2$: 房龄（年）
- $x_3$: 距离市中心的公里数

使用线性回归模型：
$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3
$$
模型训练后可能得到：
$$
\hat{y} = 50 + 0.8x_1 - 0.5x_2 - 3.0x_3
$$
该模型可帮助房地产企业预测房价，并理解面积正相关、房龄负相关、距离市中心负相关的经济意义。

  - 回归是最经典、最基础的机器学习任务
  - 目标是预测连续值（real-valued outputs）
  - 神经网络回归一切

### **回归的激励案例**

在现实世界中，许多任务涉及**预测连续数值**，回归分析正是为解决此类问题而设计的方法。通过回归模型，我们能够基于已知的历史数据，推断出未来可能的数值，帮助决策更科学、精确和可解释。以下是几个典型的激励案例，展示回归在不同领域的广泛价值：  

---

**1. 房价预测（房地产行业）**  
房地产公司和购房者往往需要估算某个区域房屋的合理价格。通过收集过去销售数据（如面积、位置、建造年份、装修情况等特征），利用回归模型建立价格预测方程：
\[
\text{Price} = \beta_0 + \beta_1 \cdot \text{Area} + \beta_2 \cdot \text{Location} + \dots + \epsilon
\]
这样，当一个潜在买家提供房屋信息时，就能快速预测它的市场价值，从而辅助议价和投资决策。  

*实例数据示例：* 假设某城区过去 100 套卖出的房屋价格和特征被收集，训练出的模型在给定一套 90㎡、靠近地铁的房屋时，可能预测它的售价约为 \( 200 \) 万元。  

---

**2. 销售额预测（商业运营）**  
零售企业常用回归方法预测下一季度的销售额，以便安排生产、库存和促销策略。例如，销售额可视作广告投放量、节假日因素和经济指标的函数。当回归模型捕捉到这些关系时，企业可提前预知销量高峰和低谷，优化运营策略。  

---

**3. 医疗健康中的疾病指标预测**  
在医疗领域，回归可用于预测病人的某项连续型临床指标，例如基于血液检测数据预测血糖水平。医生通过这种预测可以及时评估患者风险，有助于早期干预。例如，利用患者的年龄、BMI、饮食信息和生活习惯作为输入变量，构建回归方程：
\[
\text{Blood\_Glucose} = \beta_0 + \beta_1 \cdot \text{Age} + \beta_2 \cdot \text{BMI} + \beta_3 \cdot \text{Diet\_Score} + \epsilon
\]

---

**4. 环境科学中的空气污染指数预测**  
政府或环保机构通过回归分析将天气参数（温度、湿度、风速）、工业排放量等与空气质量指数（AQI）联系起来，就可以提前预测未来几天的空气质量，从而向公众发布预警和建议。  

---

**5. 例子演示（房价预测代码简例）**

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# 构造示例数据
data = pd.DataFrame({
    "Area": [70, 80, 75, 60, 95],
    "Location_Score": [8, 9, 7, 6, 10],
    "Price": [150, 180, 160, 130, 210]
})

# 特征与标签
X = data[["Area", "Location_Score"]]
y = data["Price"]

# 训练回归模型
model = LinearRegression()
model.fit(X, y)

# 预测一套90㎡、地段评分9的房价
pred_price = model.predict([[90, 9]])
print(f"预测价格: {pred_price[0]:.2f} 万元")
```

---

这些案例体现了回归分析在**经济、工业、医疗、环境等各领域的应用潜力**，不仅能提供数值预测，还能帮助解释变量之间的关系，从而优化决策过程。

  - 学生成绩预测：根据作业成绩、出勤率 → 预测期末分数
  - 天气预测：根据历史温度、湿度 → 预测未来温度
  - 股票预测：根据历史价格、交易量 → 预测次日股价

### **回归与其他任务的关系**

在机器学习任务中，回归（Regression）是一类基础且广泛应用的问题形式，其本质是在给定输入特征矩阵的情况下，预测一个**连续型输出变量**。虽然回归看似与分类、排序、聚类等任务不同，但在数学建模的角度，它们之间存在紧密的联系与相互转化的可能性。

**1. 回归与分类（Classification）的关系**  

- 从输出空间的角度看，回归是预测 \( y \in \mathbb{R} \) 的任务，而分类通常预测 \( y \in \{1, 2, ..., K\} \)。
- 许多分类问题可以通过回归模型间接解决。例如，将分类标签进行**one-hot编码**，然后用多输出回归去拟合：
	\[
	\mathbf{Y} = \begin{bmatrix}
	0 & 1 & 0 \\
	1 & 0 & 0 \\
	\vdots & \vdots & \vdots
	\end{bmatrix}, \quad \hat{\mathbf{Y}} = \mathbf{X} \mathbf{W}
	\]
	在预测阶段，可以取回归输出中最大值的索引作为分类结果。
- 逻辑回归（Logistic Regression）虽然名字中含有“回归”，但本质上是一个**分类模型**；它通过对线性回归的结果加上Sigmoid/Softmax映射，将输出限制为类别概率。

**2. 回归与排序（Ranking）的关系**  

- 排序问题可视为回归的一个变种：先预测每个样本的一个**连续评分** \( s_i \)，然后按照这个分数进行排序。
- 在信息检索与推荐系统中，常用的Learning to Rank方法有两类：
	1. **Pointwise方法**：直接将排序问题转化为回归，最小化真实相关度分数 \( y_i \) 与预测分数 \( \hat{y}_i \) 之间的均方误差：
		\[
		\min_{\mathbf{w}} \sum_{i} (y_i - \mathbf{x}_i^\top \mathbf{w})^2
		\]
	2. **Pairwise方法**：预测两样本得分差 \( \hat{y}_i - \hat{y}_j \) 的符号，实际上也是在回归差值上做优化。

**3. 回归与密度估计（Density Estimation）的关系**  

- 在**高斯假设**下的回归模型，本质上是对条件概率分布 \( p(y \mid \mathbf{x}) \) 的参数建模：
	\[
	y \mid \mathbf{x} \sim \mathcal{N}(\mathbf{x}^\top \mathbf{w}, \sigma^2)
	\]
	最大似然估计（MLE）即等价于最小化均方误差（MSE），这是回归与概率建模之间的重要桥梁。
- 如果将输出视为分布的期望，可以进一步扩展至**分位数回归（Quantile Regression）**，用于估计\( F_{Y|\mathbf{X}}^{-1}(\tau) \)等分位点。

**4. 回归与聚类（Clustering）的联系**  

- 某些聚类方法（如K-means）可以看作是对分类回归的特例。当我们把每个簇的中心向量 \(\boldsymbol{\mu}_k\) 看作一个回归模型的权重时，K-means的目标：
	\[
	\min_{\{\boldsymbol{\mu}_k\}} \sum_{i} \|\mathbf{x}_i - \boldsymbol{\mu}_{c(i)}\|^2
	\]
	形式上类似于最小化预测值与输入之间的平方误差。
- 反之，将聚类标签转化为one-hot形式后，也可以用回归去近似样本到簇中心的映射。

**实例：回归与二分类的转化**  
假设我们有二分类任务，类别标签为\(\{0,1\}\)，特征矩阵为\(\mathbf{X} \in \mathbb{R}^{n \times d}\)，我们可以直接用线性回归模型：
\[
\hat{\mathbf{y}} = \mathbf{X} \mathbf{w}
\]
然后在预测时取阈值0.5：
\[
\hat{t}_i =
\begin{cases}
1, & \text{if } \hat{y}_i \geq 0.5 \\
0, & \text{otherwise}
\end{cases}
\]
虽然这种做法在概率解释上不如逻辑回归严谨，但在某些简单任务上依然可行。

  - 分类：预测离散标签 vs 连续值预测
  - 强化学习：价值函数估计本质上是回归问题

---

## 2. 回归的线性代数基础回顾（10分钟）

- **向量 (Vector)**
	- 表示一个样本的特征，如 \( x \in \mathbb{R}^d \)
- **矩阵 (Matrix)**
	- 表示整个数据集，如 \( X \in \mathbb{R}^{n \times d} \)，包含 n 个样本、d 个特征
- **转置、点积与范数**
	- 点积：\( x^T y \)
	- 二范数：\( \|x\|_2 = \sqrt{\sum_i x_i^2} \)
- **矩阵运算在建模中的作用**
	- 预测：\( \hat{y} = Xw \)
	- 损失：\( L = \|y - Xw\|^2 \)


## 3. 回归问题的建模与评价（15分钟）"强调数学、尽量使用matrix/vector的表示形式"

### **回归的基本元素** 

回归问题是监督学习中的一种基本类型，其目标是通过训练数据学习输入变量与连续输出变量之间的映射关系。为了严谨地进行建模与分析，我们通常使用向量与矩阵的形式来表达数学结构，这样不仅可以简化推导，还便于推广到高维和多变量情况。

---

#### 输入与输出表示  

假设我们有 \(n\) 个样本，每个样本有 \(d\) 个特征，定义：  

- 特征矩阵  
	\[
	\mathbf{X} \in \mathbb{R}^{n \times d}, \quad \mathbf{X} = 
	\begin{bmatrix}
	— x_{11} & x_{12} & \dots & x_{1d} \\
	— x_{21} & x_{22} & \dots & x_{2d} \\
	\vdots & \vdots & \ddots & \vdots \\
	x_{n1} & x_{n2} & \dots & x_{nd}
	\end{bmatrix}
	\]  
	每一行 \(\mathbf{x}_i^\top \in \mathbb{R}^d\) 表示一个样本的特征向量。

- 输出向量  
	\[
	\mathbf{y} \in \mathbb{R}^n, \quad \mathbf{y} = 
	\begin{bmatrix}
	y_{1} \\
	y_{2} \\
	\vdots \\
	y_{n}
	\end{bmatrix}
	\]  
	其中 \(y_i \in \mathbb{R}\) 为样本 \(i\) 的连续标签。

---

#### 假设空间与参数向量  

在最简单的线性回归模型中，我们假设：

\[
\hat{\mathbf{y}} = \mathbf{X} \mathbf{w} + b\mathbf{1}
\]  

其中：  

- \(\mathbf{w} \in \mathbb{R}^d\)：权重向量  
- \(b \in \mathbb{R}\)：偏置（可通过在 \(\mathbf{X}\) 的首列添加常数 1 合并到 \(\mathbf{w}\) 中）  
- \(\hat{\mathbf{y}} \in \mathbb{R}^n\)：模型预测的输出向量

一般化表达（将偏置合并）：  
\[
\mathbf{X}' = \begin{bmatrix} \mathbf{1} & \mathbf{X} \end{bmatrix} \in \mathbb{R}^{n\times (d+1)}, \quad
\mathbf{w}' = \begin{bmatrix} b \\ \mathbf{w} \end{bmatrix}, \quad
\hat{\mathbf{y}} = \mathbf{X}'\mathbf{w}'
\]

---

#### 噪声模型  

许多回归模型假设观测值满足：  
\[
y_i = \mathbf{x}_i^\top \mathbf{w} + \epsilon_i
\]  
其中 \(\epsilon_i\) 是噪声项，通常假设 \(\epsilon_i \sim \mathcal{N}(0, \sigma^2)\) 独立同分布。向量化形式为：  
\[
\mathbf{y} = \mathbf{X}\mathbf{w} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})
\]

---

#### 损失函数（以均方误差为例）  

在线性回归中，最常见的优化目标是最小化均方误差（Mean Squared Error, MSE）：  
\[
\mathcal{L}(\mathbf{w}) = \frac{1}{n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2
\]  
MSE 可等效写为：  
\[
\mathcal{L}(\mathbf{w}) = \frac{1}{n} (\mathbf{y} - \mathbf{X}\mathbf{w})^\top (\mathbf{y} - \mathbf{X}\mathbf{w})
\]  

通过对损失函数求梯度并令其为零，可以得到闭式解（普通最小二乘 OLS）：  
\[
\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
\]  
（当 \(\mathbf{X}^\top \mathbf{X}\) 可逆时）

---

#### 例子  

假设我们有一个简单的房价预测数据集，只有一个特征 \(x\) 表示房屋面积（平方米），对应房价 \(y\)（万元）：  
样本数据：  
\[
\mathbf{X} = 
\begin{bmatrix}
50 \\
60 \\
80 \\
100
\end{bmatrix}, \quad
\mathbf{y} = 
\begin{bmatrix}
150 \\
180 \\
240 \\
300
\end{bmatrix}
\]

在矩阵计算中，先在特征矩阵前添加一列 1 作为偏置项：  
\[
\mathbf{X}' = 
\begin{bmatrix}
1 & 50 \\
1 & 60 \\
1 & 80 \\
1 & 100
\end{bmatrix}
\]  
使用公式 \(\mathbf{w}^* = (\mathbf{X}'^\top \mathbf{X}')^{-1} \mathbf{X}'^\top \mathbf{y}\) 可求得最佳拟合系数 \(\mathbf{w}^* = [30, 2.7]^\top\)，即预测模型为：  
\[
\hat{y} = 30 + 2.7 \cdot \text{面积}
\]

---

#### 矩阵/向量形式的优点  

- **推导简洁**：利用矩阵运算避免复杂的求和符号  
- **便于批量计算**：向量化计算可直接用于数值优化与梯度下降  
- **可推广性强**：同样的公式适用多变量、多输出回归（多任务回归）

	- 数据：输入特征与输出标签
	- 模型：函数映射输入到输出
	- 损失函数：度量预测与真实的差距

### **回归的常见评价指标**

在回归任务中，我们需要量化模型预测值与真实值之间的偏差，以便衡量模型性能、比较不同模型或调参。常见的回归评价指标主要包括以下几类：  

---

#### 1. 平均绝对误差（Mean Absolute Error, MAE）

**定义**：样本预测值与真实值差值绝对值的平均。  
公式：  
\[
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]  
**特点**：  

- 对异常值不敏感，适用于噪声较大或含有离群点的数据集。  
- 误差单位与原始目标变量相同，易解释。  

**示例**：  
如果真实值为 `[3, 5, 2]`，预测值为 `[2, 5, 4]`，则  
\[
\text{MAE} = \frac{|3-2| + |5-5| + |2-4|}{3} = \frac{1 + 0 + 2}{3} = 1
\]  

---

#### 2. 均方误差（Mean Squared Error, MSE）

**定义**：预测值与真实值差值的平方的平均。  
公式：  
\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]  
**特点**：  

- 对异常值非常敏感，因为误差平方会放大偏离较大的样本影响。  
- 常用于优化算法求解（如最小二乘回归）。  

---

#### 3. 均方根误差（Root Mean Squared Error, RMSE）

**定义**：均方误差的平方根。  
公式：  
\[
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
\]  
**特点**：  

- 可直接与原始数据单位对比，解释性较好。  
- 与 MSE 一样对离群点敏感。  

---

#### 4. 决定系数（Coefficient of Determination, \( R^2 \) Score）

**定义**：衡量模型拟合程度的指标，取值范围一般为 \((-\infty, 1]\)。  
公式：  
\[
R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
\]  
其中 \(\bar{y}\) 为真实值均值。  
**特点**：  

- \(R^2 = 1\) 代表完美拟合，\(R^2 = 0\) 表示与预测均值效果相同，负值表示比均值预测还差。  
- 更适合在相同数据集上比较模型优劣，而非不同数据集间的性能对比。  

---

#### 5. 平均绝对百分比误差（Mean Absolute Percentage Error, MAPE）

**定义**：误差绝对值与真实值绝对值之比的平均，反映相对误差大小。  
公式：  
\[
\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|
\]  
**特点**：  

- 直观反映预测的相对偏差比例。  
- 当真实值接近 0 时会产生极大误差，不适用于包含零真实值的任务。  

---

**代码示例（Python, 使用sklearn）**：

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = np.array([3, 5, 2])
y_pred = np.array([2, 5, 4])

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.2f}")
```

该部分指标在 KNN 回归中同样适用，通过选择合适的评价指标，可以更好地针对任务性质评估和优化模型性能。

  - MSE（均方误差）
  - MAE（平均绝对误差）
  - R²（决定系数）

---

## 4. KNN 回归（10分钟）

### **KNN的核心思想**

KNN（K-Nearest Neighbors，K近邻）回归的核心思想是：**“相似的输入特征会有相似的输出结果”**。在回归任务中，KNN根据新样本与训练集中样本的距离，找到距离最近的 \(K\) 个邻居，然后利用这些邻居的目标值（标签）来预测新样本的连续型输出。

---

#### 基本理念

与线性回归等参数化方法不同，KNN是一种**基于实例（Instance-based）**的**非参数化**方法，不需要显式学习一个全局函数模型。它直接依赖训练数据进行预测，在预测阶段才进行计算，因此也被称为**懒惰学习（Lazy Learning）**。

在KNN回归中：

1. 给定一个需要预测的样本点 \(x_{new}\)。
2. 计算它与训练集中所有样本点的距离（常用欧氏距离、曼哈顿距离等）。
3. 选出距离最近的 \(K\) 个样本点。
4. 将这 \(K\) 个样本的输出值取加权平均（权重可以是均匀的，也可以根据距离衰减分配）。
5. 加权平均的结果即为预测值 \(\hat{y}\)。

加权平均公式（按距离加权）：
\[
\hat{y} = \frac{\sum_{i=1}^{K} \frac{1}{d(x_{new}, x_i)} y_i}{\sum_{i=1}^{K} \frac{1}{d(x_{new}, x_i)}}
\]
其中 \(d(\cdot, \cdot)\) 表示距离函数。

---

#### 核心要素

- **K值选择**：  
	- 较小的 \(K\) 容易对局部噪声敏感（高方差）  
	- 较大的 \(K\) 会平滑预测结果但可能忽略局部特征（高偏差）  
	- 常通过交叉验证选择合适的 \(K\)

- **距离度量**：  
	- **欧氏距离**（Euclidean Distance）：常用，适合连续数值特征  
		\[
		d(\mathbf{x},\mathbf{z}) = \sqrt{\sum_{j=1}^n (x_j - z_j)^2}
		\]
	- **曼哈顿距离**（Manhattan Distance）：适合高维稀疏特征  
		\[
		d(\mathbf{x},\mathbf{z}) = \sum_{j=1}^n |x_j - z_j|
		\]
	- 对不同量纲特征，应先进行归一化或标准化

- **权重策略**：
	- 均匀权重（Uniform）：所有邻居贡献相等  
	- 距离权重（Distance-weighted）：离得越近，权重越大，反映邻近样本的重要性

---

#### 示例

假设我们要预测某个房屋价格（以万元为单位），训练集中有以下数据（已按欧氏距离计算好最近邻）：

| 样本 | 面积(m²) | 房龄(年) | 价格(万元) | 与测试样本距离 |
| ---- | -------- | -------- | ---------- | -------------- |
| A    | 80       | 5        | 120        | 1.2            |
| B    | 85       | 8        | 115        | 1.5            |
| C    | 78       | 4        | 118        | 2.5            |
| D    | 60       | 10       | 95         | 5.0            |

取 \(K=3\)，使用距离倒数加权预测：
\[
\hat{y} = \frac{\frac{1}{1.2} \times 120 + \frac{1}{1.5} \times 115 + \frac{1}{2.5} \times 118}{\frac{1}{1.2} + \frac{1}{1.5} + \frac{1}{2.5}}
\]
\[
\hat{y} \approx \frac{100 + 76.67 + 47.2}{0.8333 + 0.6667 + 0.4} \approx \frac{223.87}{1.9} \approx 117.8
\]
所以预测房价约为 **117.8 万元**。

---

#### 特点

- **优点**：
	- 简单直观，无需训练过程
	- 对非线性关系有良好适应性
- **缺点**：
	- 预测阶段计算成本高
	- 受维度灾难影响严重
	- 对噪声敏感

---

如果你需要的话，我可以继续为你生成 **KNN回归的应用场景** 或 **与KNN分类的区别** 部分的教学内容，这样你的课程会更完整，是否需要我继续补充？

  - 预测时，找到最近的 K 个邻居
  - 预测值 = 邻居输出的平均

### **KNN的直观示例**

KNN（k-Nearest Neighbors）回归的思想可以用一个生活化的比喻来理解：假设你刚搬到一个新城市，想预测一家餐馆的人均消费，但你对这家餐馆不了解，于是你选择找离它“相似”且已知价格的餐馆作为参考，将这些价格进行平均后，得到一个合理的估计值。这里，“相似”就是“距离”的体现。

在KNN回归中，“距离”通常是指特征空间中的度量，例如欧氏距离（Euclidean Distance）：

\[
\text{distance}(\mathbf{x}, \mathbf{x_i}) = \sqrt{\sum_{j=1}^{d}(x_j - x_{ij})^2}
\]

**直观步骤**：

1. 给定一个新样本点 \(x_{new}\)，我们在训练数据集中找出与它最近的 \(k\) 个邻居（样本），这里的“最近”由距离公式决定。
2. 将这些邻居的目标值（如房价、温度等连续变量）取平均，作为对 \(x_{new}\) 的预测值。
3. 如果需要，可以对不同邻居赋予权重，例如距离越近，权重越大。

**具体例子**：  
假设我们想用KNN回归预测一个房子的租金，使用以下两个特征：

- 当前房屋面积（平方米）
- 距离市中心的公里数

我们有一个训练数据集（简化版）：

| 面积 | 距离市中心 | 租金（元） |
| ---- | ---------- | ---------- |
| 50   | 1          | 3000       |
| 80   | 3          | 2500       |
| 60   | 2          | 2800       |
| 100  | 5          | 2000       |
| 40   | 1.5        | 3200       |

现在我们有一个新房屋：面积 70㎡，距离市中心 2.5km，要预测它的租金。  
设定 \(k=3\)，计算欧氏距离（对面积和距离特征先进行数值标准化是合理的，但这里先略去以直观演示）：

- 与(50,1)：  
	\[
	\sqrt{(70-50)^2 + (2.5 - 1)^2} = \sqrt{20^2 + 1.5^2} = \sqrt{400 + 2.25} \approx 20.06
	\]
- 与(80,3)：  
	\[
	\sqrt{(70-80)^2 + (2.5 - 3)^2} = \sqrt{100 + 0.25} \approx 10.01
	\]
- 与(60,2)：  
	\[
	\sqrt{(70-60)^2 + (2.5 - 2)^2} = \sqrt{100 + 0.25} \approx 10.01
	\]
- 与(100,5)：  
	\[
	\sqrt{(70-100)^2 + (2.5 - 5)^2} = \sqrt{900 + 6.25} \approx 30.10
	\]
- 与(40,1.5)：  
	\[
	\sqrt{(70-40)^2 + (2.5 - 1.5)^2} = \sqrt{900 + 1} \approx 30.02
	\]

按距离排序后，最近的3个邻居为：

- (80,3)，租金2500
- (60,2)，租金2800
- (50,1)，租金3000

预测租金为：
\[
\frac{2500 + 2800 + 3000}{3} \approx 2766.67
\]

因此，KNN回归的预测值为 2767 元左右。

**核心直观感受**：

- **局部性假设**：靠得越近（相似度高）的样本，其输出值越有参考意义。
- **非参数化模型**：模型不需要显式训练过程，而是在预测阶段直接利用全部数据。
- **对特征尺度敏感**：不同量纲可能导致距离计算失真，因此往往需要标准化或归一化处理。

如果需要简单代码示例，可用 `scikit-learn` 快速实现：

```python
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# 训练数据
X = np.array([[50, 1],
              [80, 3],
              [60, 2],
              [100, 5],
              [40, 1.5]])
y = np.array([3000, 2500, 2800, 2000, 3200])

# 新样本
new_house = np.array([[70, 2.5]])

# 创建KNN回归器
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X, y)

# 预测
prediction = knn.predict(new_house)
print(prediction)  # 输出类似于 [2766.67]
```

该示例直观展示了KNN回归基于邻近样本平均的预测过程，可帮助理解算法背后的简单而强大的思想。

  - 1-NN 与 2-NN 的对比
  - Python代码实现与应用

### **KNN的优缺点**

KNN（K-Nearest Neighbors，K近邻）是一种基于实例的监督学习方法，既可用于分类，也可用于回归。在讲授线性回归时引入KNN的优缺点，有助于学生比较“基于参数模型”和“基于实例的非参数模型”的差异，从而加深对模型选择和泛化能力的理解。

---

#### 优点

1. **实现简单，易于理解**
	- 算法思想直观：给定一个新的样本，找到训练集中距离它最近的 \( K \) 个样本，通过多数投票（分类）或均值（回归）得到预测结果。
	- 不需要复杂的训练过程，训练阶段只是将数据存储起来。

2. **非参数化（Non-parametric）**
	- 不对数据的分布形式做先验假设，适用于非线性关系建模。
	- 对复杂决策边界和多模态分布有较强的适应性。

3. **天然支持多分类与多输出**
	- 通过调整投票策略或加权方法，可以直接扩展到多类分类、回归输出多维向量的场景。

4. **增量学习能力**
	- 新样本加入数据集即可立刻被使用，无需重新训练模型（仅需更新数据存储）。

---

#### 缺点

1. **计算复杂度高**
	- 预测阶段需要计算测试样本与所有训练样本的距离，时间复杂度为 \( O(n \cdot d) \)（其中 \( n \) 为训练样本数，\( d \) 为特征维度）。
	- 对大规模数据集不适用，除非进行索引优化（如KD-Tree、Ball Tree、近似最近邻）。

2. **存储成本大**
	- 需要存储全部训练样本，空间复杂度为 \( O(n \cdot d) \)。

3. **对特征尺度敏感**
	- 由于依赖距离度量，特征值的量纲差异会严重影响模型表现，必须进行归一化或标准化处理。

4. **对噪声与无关特征敏感**
	- 噪声样本可能会被选入最近邻而影响预测结果。
	- 高维空间中“距离”失去区分度（维度灾难），需要进行特征选择或降维。

5. **延迟决策**
	- 模型无显式训练阶段，但预测时间往往较长；与线性回归的“先训练、再快速预测”形成对比。

---

#### 举例说明

假设我们要预测一个房屋的价格，特征包含房屋面积、房间数、地理位置等。使用KNN回归时，我们计算该房屋与所有训练样本的欧氏距离，找到距离最近的 \( K = 5 \) 个邻居，将他们的房价取平均作为预测结果。

- **优点表现**：如果房价与特征关系非常复杂且非线性，KNN可以较好地拟合这种模式，无需建立显式函数。
- **缺点表现**：当训练数据量达到几十万条时，单次预测需要计算几十万次距离，可能对实际应用产生性能瓶颈；此外，如果一个房屋的面积单位是平方米而另一个特征房间数非常小，不经过标准化处理，面积特征将主导距离计算，导致模型失效。

---

##### 数学表示

在KNN回归中，对于测试样本 \(\mathbf{x}\)，预测值可表示为：

\[
\hat{y}(\mathbf{x}) = \frac{1}{K} \sum_{\mathbf{x}_i \in \mathcal{N}_K(\mathbf{x})} y_i
\]

其中 \(\mathcal{N}_K(\mathbf{x})\) 表示与 \(\mathbf{x}\) 最近的 \( K \) 个邻居的集合，距离通常使用欧氏距离：

\[
d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^d (x_j - x_{i,j})^2 }
\]

---

##### 小提示

- 为降低计算量，可在预测前使用空间划分数据结构（例如KD树）或近似最近邻算法。
- 特征标准化（Standardization）或归一化（Min-Max Normalization）是KNN的常规预处理步骤。

	- 优点：简单直观，非参数方法
	- 缺点：高维数据表现差，预测开销大

---

## 5. 线性回归（20分钟）"强调数学、尽量使用matrix/vector的表示形式" 

https://www.youtube.com/watch?v=CtsRRUddV2s

### **线性回归的核心思想**

线性回归的核心思想，是通过找到一组最优的模型参数，将输入特征与输出目标之间的关系用**线性函数**表示，并最小化预测值与真实值之间的误差。它是监督学习中最基础且应用广泛的回归算法之一。

在多维特征情况下，线性回归模型可用**矩阵/向量**形式表示为：  

\[
\hat{\mathbf{y}} = \mathbf{X} \mathbf{w}
\]

其中：  

- \(\mathbf{X}\) 为 \(n \times d\) 的特征矩阵，\(n\) 是样本数，\(d\) 是特征维度；  
- \(\mathbf{w}\) 为 \(d \times 1\) 的参数向量，包括每个特征的权重（常用技巧是在 \(\mathbf{X}\) 中添加一列全1来表示截距项 \(b\)）；  
- \(\hat{\mathbf{y}}\) 为 \(n \times 1\) 的预测向量。  

核心目标是让预测值 \(\hat{\mathbf{y}}\) 尽可能接近真实目标值 \(\mathbf{y}\)。

---

### 最优化目标  

在线性回归中，误差通常使用均方误差（MSE）来度量：  

\[
J(\mathbf{w}) = \frac{1}{2n} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2
\]

这里的 \(\|\cdot\|_2^2\) 表示向量的平方欧几里得范数。  
为什么取平方误差？主要原因有：

1. 平方能够放大较大的偏差，从而减少极端误差的影响；
2. 数学上方便求导，得到封闭解。  

---

### 最优解的矩阵推导  

通过最小化 \(J(\mathbf{w})\)，对参数向量求导并令导数为零，可得**正规方程**（Normal Equation）：  

\[
\frac{\partial J(\mathbf{w})}{\partial \mathbf{w}} = \frac{1}{n} \mathbf{X}^\top (\mathbf{X}\mathbf{w} - \mathbf{y}) = 0
\]

\[
\Rightarrow \mathbf{X}^\top \mathbf{X} \mathbf{w} = \mathbf{X}^\top \mathbf{y}
\]

当 \(\mathbf{X}^\top \mathbf{X}\) 可逆时，最优解为：  

\[
\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
\]

这一过程体现了线性回归核心的数学特性：通过最小二乘法（Least Squares），在所有可能的线性模型中选出预测误差最小的那一个。

---

### 几何意义  

从几何的角度看，线性回归是在寻找一个 \(d\)-维超平面，使所有样本点尽可能接近该平面。如在二维情况下，它就是一条直线。矩阵方程 \(\mathbf{X} \mathbf{w}\) 实际是在将目标向量 \(\mathbf{y}\) 投影到特征空间的列空间（Column Space）中，找到最接近它的向量。

---

### 一个具体例子  

假设我们有两个样本、一个特征（加上截距项已转化为两列）：  

\[
\mathbf{X} = \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}
\]

根据正规方程：

\[
\mathbf{X}^\top \mathbf{X} = \begin{bmatrix} 2 & 3 \\ 3 & 5 \end{bmatrix}, \quad \mathbf{X}^\top \mathbf{y} = \begin{bmatrix} 5 \\ 8 \end{bmatrix}
\]

解得：

\[
\mathbf{w}^* = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
\]

解释：预测公式为 \(\hat{y} = 1 + 1 \cdot x\)，即当 \(x=1\) 时预测为 2，\(x=2\) 时预测为 3，与真实数据完全拟合。

---

### 实际应用中的考虑  

- 若 \(\mathbf{X}^\top \mathbf{X}\) 不可逆，可采用正则化（如岭回归）  
- 样本数远小于特征维数时，直接求逆会不稳定  
- 对特征进行标准化有助于提升解的数值稳定性  

这种以矩阵为核心的理解方式，让线性回归的计算过程与几何意义更加直观，并为后续理解更复杂的模型（如正则化回归、广义线性模型）奠定基础。

  - 假设输入与输出存在线性关系

### **线性回归的矩阵表示**

在机器学习中，线性回归（Linear Regression）是一种经典的监督学习方法，利用输入特征变量的线性组合来预测输出变量。使用矩阵与向量的方式表示线性回归，不仅可以让公式更紧凑，也便于推导和实现。

---

**1. 矩阵形式的模型表示**  

假设我们有：

- 训练数据集包含 \( m \) 个样本，每个样本有 \( n \) 个特征  
- 输入矩阵：  
	\[
	X =
	\begin{bmatrix}
	1 & x_{1,1} & x_{1,2} & \dots & x_{1,n} \\
	1 & x_{2,1} & x_{2,2} & \dots & x_{2,n} \\
	\vdots & \vdots & \vdots & \ddots & \vdots \\
	1 & x_{m,1} & x_{m,2} & \dots & x_{m,n}
	\end{bmatrix}
	\]
	这里的第一列为全 1，用于表示偏置项（截距）。因此 \( X \) 的维度为 \( m \times (n+1) \)。

- 参数向量（权重 + 偏置）：
	\[
	\theta =
	\begin{bmatrix}
	\theta_0 \\
	\theta_1 \\
	\theta_2 \\
	\vdots \\
	\theta_n
	\end{bmatrix}
	\]
	维度为 \((n+1) \times 1\)。

- 输出（预测）向量：
	\[
	\hat{y} = X \theta
	\]
	其中 \(\hat{y}\) 的维度为 \( m \times 1 \)，第 \( i \) 个元素为样本 \( i \) 的预测值。

---

**2. 代价函数（损失函数）的矩阵形式**  

常用的均方误差（Mean Squared Error, MSE）为：
\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2
\]
用矩阵形式可写为：
\[
J(\theta) = \frac{1}{2m}(X\theta - y)^{\mathrm{T}} (X\theta - y)
\]
这里 \( y \) 为大小为 \( m \times 1 \) 的真实标签向量。

---

**3. 正规方程（Normal Equation）求解**  

在线性回归的最小二乘法（Ordinary Least Squares, OLS）中，通过对 \( J(\theta) \) 对 \(\theta\) 求导并令导数为零，可以直接得到闭式解：
\[
\frac{\partial J(\theta)}{\partial \theta} = \frac{1}{m} X^{\mathrm{T}} (X\theta - y) = 0
\]
解得：
\[
\theta = (X^{\mathrm{T}}X)^{-1} X^{\mathrm{T}} y
\]
注意：

- 当 \( X^{\mathrm{T}}X \) 可逆时，该公式有效
- 对于特征维度高且存在共线性时，可考虑使用广义逆或正则化方法

---

**4. 矩阵运算的优点**

- 公式简洁，可一次性处理所有样本
- 易于使用 NumPy、TensorFlow、PyTorch 等矩阵运算库加速计算
- 推导统一，方便扩展到多输出回归、正则化回归等

---

**5. 示例（Python实现）**

```python
import numpy as np

# 样本特征矩阵（添加一列全1表示偏置）
X = np.array([
    [1, 1.0],
    [1, 2.0],
    [1, 3.0]
])  # m=3, n=1 (此处n指原始特征数)

# 标签向量
y = np.array([1, 2, 2.5]).reshape(-1, 1)

# 按正规方程计算参数
theta = np.linalg.inv(X.T @ X) @ (X.T @ y)

print("参数向量 theta:")
print(theta)  # theta[0]=截距, theta[1]=斜率
```

该例中：

- \( X \) 含有 3 个样本，每个样本 1 个特征
- 正规方程方法直接给出最优参数，不需要迭代  
- 最终得到的 \(\theta\) 可直接用于线性预测 \(\hat{y} = X\theta\)

	- 将问题写成矩阵形式，方便计算

### **最小二乘法**

最小二乘法（Least Squares Method）是一种在回归分析中广泛使用的参数估计方法，其核心思想是找到一组参数，使得模型预测值与观测值之间误差的平方和最小。在线性回归中，我们假设输入与输出之间存在线性关系，并使用最小二乘法来求解该关系中的权重参数。  

---

**1. 问题形式化**  
假设有数据集  
\[
\{(\mathbf{x}_i, y_i) \mid i=1,2,\dots, n\}
\]  
其中，\(\mathbf{x}_i \in \mathbb{R}^d\) 为输入特征向量，\(y_i \in \mathbb{R}\) 为目标值。  
我们假设线性关系为：  
\[
y_i \approx \mathbf{x}_i^\top \boldsymbol{\beta}
\]  
为了方便矩阵运算，将所有样本堆叠成一个 \(n \times d\) 的矩阵 \(X\) 和一个 \(n \times 1\) 的向量 \(\mathbf{y}\)：  
\[
X =
\begin{bmatrix}

- & \mathbf{x}_1^\top & - \\
- & \mathbf{x}_2^\top & - \\
	& \vdots & \\
- & \mathbf{x}_n^\top & -
	\end{bmatrix}, \quad
	\mathbf{y} =
	\begin{bmatrix}
	y_1 \\
	y_2 \\
	\vdots \\
	y_n
	\end{bmatrix}
	\]  
	模型假设：
	\[
	\mathbf{y} \approx X \boldsymbol{\beta}
	\]  
	其中 \(\boldsymbol{\beta} \in \mathbb{R}^d\) 是需要估计的参数向量。

---

**2. 目标函数定义**  
最小二乘法的目标是最小化残差向量  
\[
\mathbf{r} = \mathbf{y} - X\boldsymbol{\beta}
\]  
的平方和（Sum of Squared Errors, SSE）：  
\[
J(\boldsymbol{\beta}) = \|\mathbf{y} - X\boldsymbol{\beta}\|_2^2 
= (\mathbf{y} - X\boldsymbol{\beta})^\top (\mathbf{y} - X\boldsymbol{\beta})
\]

---

**3. 推导闭式解（Normal Equation）**  
为了找到最优的 \(\boldsymbol{\beta}\)，对 \(J(\boldsymbol{\beta})\) 关于 \(\boldsymbol{\beta}\) 求导，并令导数为零：  
\[
\frac{\partial J}{\partial \boldsymbol{\beta}} 
= -2X^\top (\mathbf{y} - X\boldsymbol{\beta}) = 0
\]  
得出 **正规方程（Normal Equation）**：  
\[
X^\top X \boldsymbol{\beta} = X^\top \mathbf{y}
\]  
假设 \(X^\top X\) 可逆，则有闭式解：  
\[
\boldsymbol{\beta} = (X^\top X)^{-1} X^\top \mathbf{y}
\]  
该解是均方误差意义下的最优解。

---

**4. 几何解释**  
在几何意义上，最小二乘法等价于**将目标向量 \(\mathbf{y}\) 投影到 \(X\) 的列空间**上，投影向量正是 \(X\boldsymbol{\beta}\)，而残差向量与列空间正交：  
\[
X^\top (\mathbf{y} - X\boldsymbol{\beta}) = 0
\]  
这就是正规方程的来源。

---

**5. 数值计算与注意事项**  

- 当 \(X^\top X\) 不可逆（奇异矩阵或病态矩阵）时，可以使用**伪逆（Moore-Penrose Pseudoinverse）**求解：
	\[
	\boldsymbol{\beta} = X^+ \mathbf{y}
	\]
- 对于大规模数据，通常使用 **QR 分解** 或 **SVD 分解** 来提高数值稳定性。  
- 最小二乘法假设误差项为独立同分布的高斯噪声，在该假设下参数估计具有最小方差。

---

**6. 简单例子**  
设有训练数据点：  
\[
(1, 1), \quad (2, 2), \quad (3, 2)
\]  
用模型 \(y = \beta_0 + \beta_1 x\) 进行拟合。  

构造矩阵：  
\[
X =
\begin{bmatrix}
1 & 1 \\
1 & 2 \\
1 & 3
\end{bmatrix}, \quad
\mathbf{y} =
\begin{bmatrix}
1 \\
2 \\
2
\end{bmatrix}
\]  
代入公式：
\[
\boldsymbol{\beta} = (X^\top X)^{-1} X^\top \mathbf{y}
\]  
计算得：
\[
X^\top X = 
\begin{bmatrix}
3 & 6 \\
6 & 14
\end{bmatrix}, \quad
(X^\top X)^{-1} =
\frac{1}{6}
\begin{bmatrix}
14 & -6 \\
-6 & 3
\end{bmatrix}
\]  
\[
X^\top \mathbf{y} =
\begin{bmatrix}
5 \\
11
\end{bmatrix}
\]  
因此：
\[
\boldsymbol{\beta} = \frac{1}{6}
\begin{bmatrix}
14 & -6 \\
-6 & 3
\end{bmatrix}
\begin{bmatrix}
5 \\
11
\end{bmatrix}
= 
\begin{bmatrix}
\frac{5}{3} \\
\frac{1}{2}
\end{bmatrix}
\]  
所以拟合直线为：
\[
\hat{y} = \frac{5}{3} + \frac{1}{2} x
\]

---

**7. 小结要点**  

- 目标函数为残差平方和  
- 通过正规方程可得闭式解  
- 几何解释是向列空间的正交投影  
- 数值计算需注意矩阵可逆性与稳定性  
- 在高斯噪声假设下，该估计是无偏且方差最小的  

	- 目标函数
	- 闭式解 
	- Python代码实现与应用

### **梯度下降** 

梯度下降（Gradient Descent）是一种通用的数值优化方法，在机器学习中被广泛用于最小化损失函数，从而找到模型的最优参数。在**线性回归**中，我们通过最小化均方误差（Mean Squared Error, MSE）损失来求取权重向量 \( \mathbf{w} \)。

---

#### 1. 问题背景与目标函数

设训练数据为：

\[
\mathbf{X} \in \mathbb{R}^{m \times n}, \quad \mathbf{y} \in \mathbb{R}^m
\]

其中：

- \( m \) 表示样本数
- \( n \) 表示特征数（不含偏置项）
- \( \mathbf{w} \in \mathbb{R}^n \) 为权重参数向量
- 若考虑偏置项 \( b \)，可将其并入 \( \mathbf{w} \) 作为额外一维特征（在 \(\mathbf{X}\) 前添加一列 1）

线性回归预测公式为：

\[
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}
\]

均方误差损失函数（向量形式）为：

\[
J(\mathbf{w}) = \frac{1}{2m} \left\| \mathbf{X}\mathbf{w} - \mathbf{y} \right\|_2^2
\]

---

#### 2. 梯度计算

我们需要计算损失函数对参数 \(\mathbf{w}\) 的梯度：

\[
\nabla_{\mathbf{w}} J(\mathbf{w}) 
= \frac{1}{m} \mathbf{X}^\mathsf{T} (\mathbf{X}\mathbf{w} - \mathbf{y})
\]

这是通过矩阵求导规则得出的结论，其中：

- \(\mathbf{X}^\mathsf{T} (\mathbf{X}\mathbf{w} - \mathbf{y})\) 是残差向量通过转置后的数据矩阵加权求和；
- 梯度的方向是当前损失增加最快的方向。

---

#### 3. 参数更新规则

梯度下降的核心是**迭代更新**参数，在每一步用损失函数负梯度方向的步长进行移动：

\[
\mathbf{w} := \mathbf{w} - \alpha \nabla_{\mathbf{w}} J(\mathbf{w})
\]

其中 \(\alpha > 0\) 是学习率（learning rate），控制参数更新幅度。  
结合上面的梯度公式，有：

\[
\mathbf{w} := \mathbf{w} - \frac{\alpha}{m} \mathbf{X}^\mathsf{T} (\mathbf{X}\mathbf{w} - \mathbf{y})
\]

---

#### 4. 梯度下降的变体

- **批量梯度下降（Batch Gradient Descent）**  
	每次使用整个训练集计算梯度，更新参数，稳定但计算量大。
- **随机梯度下降（Stochastic Gradient Descent, SGD）**  
	每次随机选取一个样本计算梯度并更新，更新频繁，收敛速度快但波动大。
- **小批量梯度下降（Mini-Batch Gradient Descent）**  
	每次使用一小批（如 32、64 个样本）计算梯度，兼顾计算效率与收敛稳定性。

---

#### 5. 示例

假设一个简单的二维数据集（已包含偏置列）：

\[
\mathbf{X} = \begin{bmatrix}
1 & 1 \\
1 & 2 \\
1 & 3
\end{bmatrix}, 
\quad
\mathbf{y} = \begin{bmatrix}
1 \\
2 \\
2
\end{bmatrix}
\]

利用批量梯度下降更新步骤：

```python
import numpy as np

# 数据
X = np.array([[1, 1],
              [1, 2],
              [1, 3]], dtype=float)
y = np.array([1, 2, 2], dtype=float)

# 初始化参数
w = np.zeros(X.shape[1])
alpha = 0.1
m = len(y)

# 迭代
for epoch in range(1000):
    grad = (1/m) * X.T @ (X @ w - y)
    w = w - alpha * grad

print("Learned weights:", w)
```

该过程通过不断迭代更新权重 \(\mathbf{w}\)，最终使得预测值与真实值的均方误差最小。

---

#### 6. 学习率选择与收敛

- 学习率过大：可能导致更新越过最优解，甚至发散
- 学习率过小：收敛过慢，需要更多迭代
- 常用方法：网格搜索、衰减学习率（如 \(\alpha_t = \frac{\alpha_0}{1 + kt}\)）

梯度下降是连接**数学理论**与**实际算法实现**的重要桥梁，在掌握其矩阵形式后，可以更容易推广到多元回归以及更复杂的机器学习模型中。

  - 更新公式：数值优化方法，适合大规模数据
  - Python代码实现与应用
- https://www.youtube.com/watch?v=qg4PchTECck&list=PLqwozWPBo-FtNyPKLDPTVDOHwK12QbVsM&index=3

### **线性回归的优缺点**

线性回归（Linear Regression）是机器学习中最基础且应用广泛的回归方法之一，其核心思想是在假设特征与目标之间存在线性关系的前提下，学习一个权重向量 **\(\mathbf{w}\)** 和偏置 **\(b\)** 来最小化预测值与真实值之间的误差。其数学模型可表示为：  

\[
\hat{\mathbf{y}} = \mathbf{X} \mathbf{w} + b
\]

其中：

- \(\mathbf{X} \in \mathbb{R}^{n \times d}\)：样本特征矩阵  
- \(\mathbf{w} \in \mathbb{R}^{d \times 1}\)：权重向量  
- \(b \in \mathbb{R}\)：偏置项  
- \(\hat{\mathbf{y}} \in \mathbb{R}^{n \times 1}\)：预测值向量  

损失函数一般为**最小二乘法（Ordinary Least Squares, OLS）**：

\[
J(\mathbf{w}, b) = \frac{1}{2n} \|\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}\|_2^2
\]

下面从优势与劣势两个方面分析线性回归的特点，并探讨其在实际建模过程中的适用性与局限性。

---

#### 优点

1. **数学形式简单，易于实现与解释**  
	由于预测是输入特征的线性组合，\(\mathbf{w}\) 的每一个元素都具有明确含义：它表示该特征对预测结果的边际贡献（在其他特征不变时，特征值增加 1 单位时预测值的变化量）。  

2. **计算效率高**  
	模型训练可通过解析解直接得到：
	\[
	\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
	\]
	在样本数和特征数不大时，可以非常快速地训练完毕；对于大规模数据，可以使用基于梯度的迭代优化方法。

3. **对线性关系的建模能力强**  
	如果数据确实符合线性假设，线性回归能给出很好的估计，并且在样本量足够大时具有较好的泛化性能。

4. **可扩展性与可修改性强**  
	可以方便地引入多项式特征（Polynomial Regression）、交互项、正则化（Ridge, Lasso等）来提升性能或防止过拟合。

---

#### 缺点

1. **对线性假设敏感**  
	如果真实的特征与目标之间关系是高度非线性的，线性回归的拟合效果将明显欠佳。  
	例如，当数据来自 \(y = \sin(x)\) 且直接使用线性模型时，残差会明显表现出系统性模式。

2. **对异常值敏感**  
	最小二乘法会平方放大误差，单个极端异常点可能对 \(\mathbf{w}\) 的估计产生巨大影响。

3. **特征多重共线性影响显著**  
	若特征之间存在高度相关性，\(\mathbf{X}^\top \mathbf{X}\) 可能接近奇异矩阵或不可逆，导致解析解不稳定、权重估计方差大。  
	此时可以使用岭回归（Ridge Regression）等正则化方法来缓解。

4. **假设条件严格**  
	经典线性回归在推导及统计推断中依赖于高斯噪声、方差齐性、样本独立等假设条件。当这些条件不满足时，参数估计的性质可能受损。

5. **难以自动捕获特征交互与非线性模式**  
	在不进行特征工程的情况下，模型无法自行发现特征间的非线性组合关系，可能限制性能。

---

#### 一个简短的实例

假设我们有一个简单数据集：
\[
\mathbf{X} =
\begin{bmatrix}
1 & 2 \\
2 & 0 \\
3 & 1
\end{bmatrix}, \quad
\mathbf{y} =
\begin{bmatrix}
5 \\
6 \\
7
\end{bmatrix}
\]

使用解析解方法求解：
\[
\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
\]

在Python中实现：

```python
import numpy as np

X = np.array([[1,2],[2,0],[3,1]])
y = np.array([5,6,7])

w = np.linalg.inv(X.T @ X) @ X.T @ y
print(w)  # 输出系数
```

结果中的每个系数可直接解释为对应特征对输出的线性贡献。通过这个简单例子，可以看到线性回归在小数据集上的快速建模能力，但如果数据关系复杂或噪声大，模型表现可能显著下降。

  - 优点：可解释性强，计算高效
  - 缺点：只能建模线性关系

---

## 6. 正则化与改进（15分钟）"强调数学、尽量使用matrix/vector的表示形式"

### **Ridge 回归**

Ridge 回归（Ridge Regression）是一种在最小二乘回归基础上增加 \( L_2 \) 范数正则化项的回归方法，主要用于解决多重共线性问题以及防止模型过拟合。与普通最小二乘（Ordinary Least Squares, OLS）相比，Ridge 回归通过对参数施加惩罚，缩小回归系数的值，从而提高模型的泛化能力。

---

#### 数学定义

给定训练数据集：
\[
X \in \mathbb{R}^{n \times p}, \quad y \in \mathbb{R}^n
\]
其中：

- \(n\) 为样本数
- \(p\) 为特征数
- \(X\) 通常假设已中心化（每列均值为 0）
- \(y\) 也已中心化（均值为 0）

普通最小二乘回归的优化目标为：
\[
\min_{\beta} \ \| y - X\beta \|_2^2
\]

Ridge 回归则在此基础上引入 \( L_2 \) 正则化：
\[
\min_{\beta} \ \| y - X\beta \|_2^2 + \lambda \| \beta \|_2^2
\]
其中：

- \(\lambda \geq 0\) 为正则化系数
- 当 \(\lambda = 0\) 时，退化为普通最小二乘
- 当 \(\lambda \to \infty\) 时，所有 \(\beta_j \to 0\)

---

#### 矩阵形式解

优化问题为：
\[
\min_{\beta} \ (y - X\beta)^\top (y - X\beta) + \lambda \beta^\top \beta
\]

对 \(\beta\) 求导并令其为 0：
\[
-2X^\top(y - X\beta) + 2\lambda \beta = 0
\]
\[
(X^\top X + \lambda I_p) \beta = X^\top y
\]
因此解为：
\[
\hat{\beta}^{ridge} = (X^\top X + \lambda I_p)^{-1} X^\top y
\]
其中 \(I_p\) 为 \(p \times p\) 的单位矩阵。

这一解法与 OLS 的区别在于多加了 \(\lambda I_p\)，这使得矩阵 \(X^\top X + \lambda I_p\) 总是可逆（即使 \(X^\top X\) 奇异），有效解决多重共线性问题。

---

#### 几何解释

在 OLS 中，我们最小化的是残差平方和，其等值线是椭圆形。而 \(L_2\) 正则项 \(\| \beta \|_2^2 \leq t\) 对应的约束区域是一个超球体（在二维为圆形）。Ridge 回归等价于在椭圆等值线与圆形约束的交线上寻找最优点，由于圆形约束会均匀压缩系数，使得 Ridge 回归倾向于得到较小但非零的系数。

---

#### 与 Lasso 的区别

- Ridge 对系数进行“缩小”但不会将其精确压为 0
- Lasso (\(L_1\) 正则化) 会使部分系数精确为 0，从而具有特征选择能力
- 当特征具有相似重要性且数量较多时，Ridge 表现更优

---

#### 示例

假设有数据集：
\[
X = \begin{bmatrix}
1 & 1 \\
1 & 2 \\
1 & 3
\end{bmatrix},
\quad
y = \begin{bmatrix}
1 \\ 2 \\ 3
\end{bmatrix},
\quad
\lambda = 1
\]
则：
\[
X^\top X = \begin{bmatrix}
3 & 6 \\
6 & 14
\end{bmatrix},
\quad
X^\top y = \begin{bmatrix}
6 \\ 14
\end{bmatrix}
\]
根据公式：
\[
\hat{\beta}^{ridge} = (X^\top X + \lambda I)^{-1} X^\top y
\]
\[
= \left( 
\begin{bmatrix}
3 & 6 \\
6 & 14
\end{bmatrix} 

+ \begin{bmatrix}
	1 & 0 \\
	0 & 1
	\end{bmatrix}
	\right)^{-1}
	\begin{bmatrix}
	6 \\ 14
	\end{bmatrix}
	\]
	\[
	= \begin{bmatrix}
	4 & 6 \\
	6 & 15
	\end{bmatrix}^{-1}
	\begin{bmatrix}
	6 \\ 14
	\end{bmatrix}
	\]
	求逆并计算，可以得到缩小后的 \(\hat{\beta}^{ridge}\) 与 OLS 的解相比数值更接近 0。

---

#### Python 实现示例

```python
import numpy as np

# 数据
X = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
y = np.array([1, 2, 3], dtype=float)
lam = 1.0

# Ridge 回归解
I = np.eye(X.shape[1])
beta_ridge = np.linalg.inv(X.T @ X + lam * I) @ X.T @ y
print(beta_ridge)
```

该方法在实际应用中通常需要通过交叉验证调整 \(\lambda\) 的取值，以在偏差与方差之间取得最佳平衡。

  - 核心思想：L2 正则化，防止过拟合
  - 目标函数
  - 闭式解

### **Lasso 回归**

Lasso 回归（Least Absolute Shrinkage and Selection Operator）是一种在线性回归的基础上加入 **L1 正则化** 的技术，其主要目的是在防止过拟合的同时实现**特征选择**。在 Lasso 回归中，通过在损失函数中加入系数绝对值的惩罚项，使得部分回归系数收缩为 0，从而自动完成变量筛选。  

**数学形式**  
给定输入矩阵  
\[
X \in \mathbb{R}^{n\times p}, \quad y \in \mathbb{R}^n
\]
Lasso 回归的优化问题可写为：  
\[
\hat{\beta} = \arg\min_{\beta} \left\{ \frac{1}{2n} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1 \right\}
\]
其中：  

- \(\|\beta\|_1 = \sum_{j=1}^p |\beta_j|\)  
- \(\lambda \ge 0\) 为正则化系数，控制惩罚项的强弱  
- 当 \(\lambda\) 较大时，更多的系数会被压缩到 0，模型更稀疏  
- 当 \(\lambda = 0\) 时，退化为普通最小二乘回归（OLS）

**几何解释**  

- Lasso 惩罚项对应的约束条件是一个 \(L_1\) 范数“菱形”区域：  
	\[
	\{\beta \in \mathbb{R}^p : \|\beta\|_1 \leq t\}
	\]
- 误差平方和的等高线是椭圆形  
- 当椭圆第一次与菱形边界相切时，由于菱形的尖角常与坐标轴对齐，相切点很可能落在坐标轴上，对应于某个系数等于 0  
- 这就是 Lasso 能产生稀疏解的原因

**与 Ridge 回归的对比**  

- Ridge 回归使用 \(L_2\) 范数惩罚项，不会使系数精确为 0，只会变小  
- Lasso 回归使用 \(L_1\) 范数惩罚项，会产生大量精确为 0 的系数，自动去除不重要的特征  
- 在高维稀疏问题中，Lasso 往往更有优势

**优化方法**  
由于 L1 范数在 0 处不可导，无法直接用标准解析解，需要用迭代算法，例如：  

- 坐标下降法（Coordinate Descent）：逐个更新系数，其他系数保持固定  
- 子梯度法（Subgradient Method）：使用 L1 范数的子梯度更新  
- LARS（Least Angle Regression）：通过逐步引入变量得到整条解路径

**简例**  

假设我们有训练数据：
\[
X =
\begin{bmatrix}
1 & 1 \\
1 & 2 \\
2 & 2 \\
2 & 3
\end{bmatrix}, \quad
y = 
\begin{bmatrix}
6 \\
8 \\
9 \\
11
\end{bmatrix}
\]
当使用普通最小二乘回归时，得到的系数可能为：
\[
\beta_{\text{OLS}} = 
\begin{bmatrix}
2 \\
2
\end{bmatrix}
\]
但当我们施加 L1 惩罚（如 \(\lambda = 1\)）时，通过 Lasso 得到：
\[
\beta_{\text{Lasso}} \approx 
\begin{bmatrix}
1.5 \\
2.0
\end{bmatrix}
\]
若 \(\lambda\) 增大到某个值，可能会出现：
\[
\beta_{\text{Lasso}} =
\begin{bmatrix}
0 \\
c
\end{bmatrix}
\]
说明第一个特征被自动剔除。

**Python 示例**  

```python
import numpy as np
from sklearn.linear_model import Lasso

X = np.array([[1, 1],
              [1, 2],
              [2, 2],
              [2, 3]])
y = np.array([6, 8, 9, 11])

lasso = Lasso(alpha=1.0)
lasso.fit(X, y)

print("系数:", lasso.coef_)
print("截距:", lasso.intercept_)
```

这个示例展示了 Lasso 回归在小规模数据中施加 L1 惩罚的效果，有助于理解它在特征选择中的作用。

  - 核心思想：L1 正则化，可产生稀疏解
  - 目标函数
  - 求解

### **Elastic Net**

Elastic Net是一种结合了Lasso回归（L1正则化）与Ridge回归（L2正则化）优点的正则化方法，适用于高维稀疏特征和多重共线性同时存在的场景。它能够在实现特征选择（L1）和权重收缩（L2）之间取得平衡，从而提升模型的泛化能力。  

在Elastic Net中，我们在目标函数中引入了两种正则项，并通过一个混合系数来调节二者的相对权重。  

---

**1. 数学形式**  

设训练数据为  
\[
\mathbf{X} \in \mathbb{R}^{n \times p}, \quad \mathbf{y} \in \mathbb{R}^{n}
\]  
其中 \( n \) 为样本数，\( p \) 为特征数，参数向量 \(\boldsymbol{\beta} \in \mathbb{R}^p\)，截距为 \( \beta_0 \)。  

Elastic Net的优化目标函数为：  
\[
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left[ \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta} \|_2^2 + \lambda \left( \alpha \|\boldsymbol{\beta}\|_1 + \frac{1-\alpha}{2} \|\boldsymbol{\beta}\|_2^2 \right) \right]
\]  

其中：  

- \( \|\boldsymbol{\beta}\|_1 = \sum_{j=1}^{p} |\beta_j| \) 为L1范数（Lasso）；  
- \( \|\boldsymbol{\beta}\|_2^2 = \sum_{j=1}^{p} \beta_j^2 \) 为L2范数平方（Ridge）；  
- \( \lambda \ge 0 \) 为整体正则化强度超参数；  
- \( \alpha \in [0,1] \) 控制L1与L2的比例：  
	- \( \alpha=1 \) 时退化为Lasso；  
	- \( \alpha=0 \) 时退化为Ridge；  
	- \( 0 < \alpha < 1 \) 时为Elastic Net的混合正则化。  

---

**2. 算法特性与优势**  

- **特征选择与稀疏性**：由于包含L1部分，部分回归系数可以被压缩为零，实现自动特征选择。  
- **处理多重共线性**：L2部分会使得高度相关的特征参数趋向于接近而非二选一，从而提升稳定性。  
- **灵活性**：通过调整\(\alpha\)可在稀疏性与稳定性之间灵活切换。  
- **适合高维数据**：当 \(p \gg n\) 时，也能保持较好的预测性能。  

---

**3. 几何解释**  

Elastic Net的约束集合为L1球与L2球的"嵌入"组合，其形状类似于带圆角的菱形。L1部分促使解落在坐标轴上（产生稀疏解），L2部分则在L1约束的尖角处引入平滑性，使解在高度相关特征下更加稳定。  

---

**4. 示例**  

假设我们有房价预测问题，特征包括多个房间相关指标（高度相关），同时还有一些可能无关的特征。我们希望在去掉无关特征的同时避免在相关特征间随机丢失信息，此时Elastic Net是合适的选择。

```python
import numpy as np
from sklearn.linear_model import ElasticNet

# 构造示例数据
np.random.seed(42)
X = np.random.randn(100, 10)
y = 3*X[:, 0] + 1.5*X[:, 1] - 2*X[:, 2] + np.random.randn(100)

# Elastic Net回归
model = ElasticNet(alpha=0.1, l1_ratio=0.7)  # alpha: λ, l1_ratio: α
model.fit(X, y)

print("系数估计：", model.coef_)
```

---

**5. 参数调优**  

Elastic Net需要调整的关键参数包括：  

- **\(\lambda\) (alpha参数 in sklearn)**：整体正则化强度，越大表示惩罚越强，系数更趋向于零。  
- **\(\alpha\) (l1_ratio in sklearn)**：L1与L2的比例。  
	可以使用交叉验证（如`ElasticNetCV`）同时选择最佳\(\lambda\)和\(\alpha\)。  

---

**6. 矩阵求解与实现要点**  

实际中由于L1范数不可导，Elastic Net的求解通常依赖**坐标下降法（Coordinate Descent）**。  
其思想是每次固定除一个参数外的其他参数，更新该参数以最小化目标函数，并在L1部分使用soft-thresholding进行稀疏化：  
\[
\beta_j \leftarrow \frac{S\left( \frac{1}{n} \mathbf{x}_j^\top(\mathbf{y} - \mathbf{X}_{-j}\boldsymbol{\beta}_{-j}),\ \lambda\alpha \right)}{ 1 + \lambda(1-\alpha) }
\]  
其中 \(S(z, \gamma) = \text{sign}(z) \cdot \max(|z|-\gamma, 0)\) 为soft-thresholding算子。  

  - 核心思想：L1 + L2 结合，适合高维数据
  - 目标函数
  - 求解

---

## 7. 神经网络与回归任务（5分钟）"简略介绍下数学上的联系"

- **回归作为神经网络的基础**
	- 最简单的神经网络（单层感知机）就是线性回归模型：
		\[
		\hat{y} = w^T x + b
		\]
	- 如果去掉非线性激活函数，神经网络退化为线性回归

- **多层感知机 (MLP) 与非线性回归**
	- 引入非线性激活函数（ReLU, Sigmoid, Tanh 等），神经网络能够逼近任意连续函数
	- 神经网络的训练目标依然是最小化均方误差 (MSE) 等回归损失：
		\[
		L(\theta) = \frac{1}{n}\sum_{i=1}^n (y_i - f_\theta(x_i))^2
		\]

- **神经网络与回归任务的联系**
	- 线性回归是神经网络的「最简原型」
	- 神经网络可以被视为 **“非线性基函数扩展的回归模型”**
	- 从优化角度看：依旧是 **参数化函数拟合 + 损失最小化**

---

## 8. 总结 & 课堂互动（5分钟）

- **本节回顾**
	- 回归任务的重要性
	- KNN 回归与线性回归
	- 最小二乘与梯度下降
	- 正则化方法
- **互动问题**
	- 如果数据关系不是线性的，该怎么办？
	- 为下一节 **分类（Classification）** 做铺垫