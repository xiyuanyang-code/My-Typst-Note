# 机器学习课程 — 第七节课详细大纲（Parameter Estimation, 90分钟）

## 1. 导入与动机（10分钟）
- **为什么关注参数估计？**
  - 模型学习的核心任务就是确定模型参数
  - 线性代数视角 vs 概率视角
- **示例**：预测房价、股票、学生成绩
- **课堂互动**：讨论“点估计 vs 分布估计”的区别

## 2. Maximum Likelihood Estimation (MLE)（15分钟）
- 定义与直观解释
- 数学公式：
\[
\hat{\theta}_{MLE} = \arg\max_\theta p(Y|X,\theta)
\]
- 示例：
  - 伯努利分布
  - 高斯分布
- 与线性回归最小二乘解的对应关系

## 3. Maximum a Posteriori (MAP)（15分钟）
- 定义与公式：
\[
\hat{\theta}_{MAP} = \arg\max_\theta p(\theta|X,Y) = \arg\max_\theta p(Y|X,\theta)p(\theta)
\]
- 引入先验
- 与正则化的对应：
  - L2 ↔ 高斯先验
  - L1 ↔ 拉普拉斯先验
- 对比 MLE 与 MAP 的区别

## 4. 参数估计在 Linear Regression 中的应用（20分钟）
- **常规 Linear Regression**
  - \(y = Xw + b + \epsilon\)
  - 最小二乘法求解
- **MLE-based Linear Regression**
  - 假设误差服从高斯分布
  - 最大化似然 = 最小化平方误差
- **MAP-based Linear Regression**
  - 加入先验分布
  - 对应正则化（Ridge / Lasso）
- **Bayesian Linear Regression**
  - 参数为随机变量
  - 求预测分布
  - 可计算不确定性

## 5. Gaussian Process (GP)（20分钟）
- 非参数贝叶斯方法
- 核函数表示无限维线性回归
- 预测均值与置信区间
- 训练与推理的数学形式：
  - 协方差矩阵 \(K(X,X)\)
  - 条件高斯分布求预测
- 对比 Linear Regression 与 GP 的关系

## 6. 小结与对比（10分钟）
- **MLE → MAP → Bayesian → GP**
- 线性代数视角 vs 概率视角
- 参数估计对模型学习的核心意义
- 实践注意事项：
  - 数据量与参数数量关系
  - 正则化与先验选择
  - 训练误差 vs 泛化能力

## 7. Q&A 与课堂互动（5分钟）
- 回答学生问题
- 讨论概率视角在非线性和深度模型中的应用