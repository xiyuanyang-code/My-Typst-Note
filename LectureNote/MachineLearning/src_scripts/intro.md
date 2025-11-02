# 机器学习课程 — 第一节课详细大纲（Introduction, 90分钟） https://pan.sjtu.edu.cn/web/share/507ca3a1d4dc07a3a4572183abd34c44

## 1. 开场 & Fancy 前沿案例展示（20分钟）
- **课程开场 & 自我介绍**
  - 教师背景、研究方向
  - 本课程目标与收获
- **前沿案例展示（吸引注意力）**
  - **AlphaGo / AlphaZero**：通过强化学习击败世界围棋冠军
    - 展示强化学习智能体如何通过自我博弈不断优化策略
  - **ChatGPT / GPT-5**：生成自然语言对话、文章、代码 https://www.youtube.com/watch?v=boJG84Jcf-4
    - 说明大模型（LLM）的概念和应用
  - **DALL·E / nano banana**：AI 文生图生成 https://www.youtube.com/watch?v=-tMERzjAvgw&t=22s
    - 展示从文本到图像的生成能力，说明多模态学习
  - **Waymo / Tesla 自动驾驶**：感知、决策与控制结合的实际应用 
     waymo https://www.youtube.com/watch?v=hA_-MkU0Nfw
     tesla  https://www.youtube.com/watch?v=We2ZD0-IXPM
  - **Boston Dynamics / 仿人机器人**：具身智能与环境交互 https://www.youtube.com/watch?v=_EZQx87DyzM
    - 强调 RL 与计算机视觉在物理机器人中的应用
- **课堂互动**
  - 提问学生：“你最惊讶或最感兴趣的 AI 技术是什么？”
  - 鼓励学生分享已知的 AI 产品或应用

## 2. 机器学习基础概念（20分钟）
- **机器学习定义**
  - 传统编程 vs 数据驱动
  - 从数据中学习规律并做预测或决策
- **发展历史与相关学科**
  - 统计学、信号处理与机器学习的关系
  - 从线性回归、决策树到深度学习的演进
- **核心概念**
  - **数据**：特征（Feature）、样本（Sample）
  - **模型**：函数映射输入到输出
  - **训练**：通过优化损失函数调整模型参数
  - **预测**：将训练好的模型应用到新数据
  - 可用简单公式示意：  
    \[
    y = f_\theta(x), \quad \text{优化 } \theta \text{ 最小化 } L(y, \hat{y})
    \]

## 3. 机器学习的主要分支与应用（30分钟）

### 3.1 深度学习（Deep Learning）
- **概念**：多层非线性神经网络，自动学习特征
- **应用示例**
  - **图像**：卷积神经网络 (CNN) 进行图像识别
  - **时间序列 / NLP**：RNN、LSTM处理序列数据
  - **现代趋势**：Transformer 架构，基础大模型

### 3.2 强化学习（Reinforcement Learning）
- **概念**：智能体通过与环境交互学习策略
- **应用示例**
  - AlphaGo 的自我博弈策略  
  - ai游戏: 星际争霸 https://www.youtube.com/watch?v=VZ7zmQ_obZ0
- **核心概念**：状态 (State)、动作 (Action)、奖励 (Reward)

### 3.3 计算机视觉（Computer Vision）
- **概念**：让机器理解图像或视频
- **应用示例**
  - 自动驾驶感知（目标检测、分割）
      https://www.youtube.com/watch?v=OopTOjnD3qY
  - 医学影像分析
  - 多模态大模型：图像 + 文本
      gemini https://www.youtube.com/watch?v=bbkcQp5X3h0
  - 文生图、文生视频生成  
      sora: https://www.youtube.com/watch?v=HK6y8DAPN_0
  - 世界模型
      nvidia cosmos https://www.youtube.com/watch?v=9Uch931cDx8
- **课堂互动**
  - 展示生成图片案例

### 3.4 自然语言处理（Natural Language Processing）
- **概念**：机器理解和生成文本
- **应用示例**
  - 大模型 (LLM) 文生文、对话系统
  - 机器翻译、情感分析、文本摘要

### 3.5 具身智能（Embodied Intelligence）
- **概念**：AI 拥有身体，可与环境交互
- **应用示例**
  - 具身大模型 VLA（视觉 + 语言 + 动作） https://www.youtube.com/watch?v=Z3yQHYNXPws
  - 人形机器人 RL 训练（行走、抓物） 
      unitree https://www.youtube.com/watch?v=mSPxRVTJW1I
      optimus https://www.youtube.com/shorts/gOYAfEOeg1Y
## 4. 课程脉络与学习路线（15分钟）
### 第二节课：Regression（回归）
- **目标**：理解如何预测连续数值输出
- **主要内容**
  - **KNN-based regression**
    - 基于最近邻的预测方法
    - 简单直观，非参数模型
  - **Linear regression**
    - 假设输出与输入特征线性相关
    - 数学公式：\( y = w^T x + b \)
  - **Least squares**
    - 最小化预测值与真实值的平方误差
    - 公式：\(\min_w \sum_i (y_i - w^T x_i)^2\)
  - **Gradient descent**
    - 梯度下降求解参数优化问题
    - 学习率与收敛示意

### 第三节课：Classification（分类）
- **目标**：理解如何预测类别标签
- **主要内容**
  - **Decision trees**
    - 树结构决策，分裂特征选择
    - 信息增益 / 基尼系数
  - **Logistic regression**
    - 概率预测，输出 0-1 概率
    - Sigmoid 函数：\(\sigma(z) = \frac{1}{1+e^{-z}}\)
  - **SVM (Support Vector Machine)**
    - 最大化类别间隔
    - 核函数处理非线性分类问题

### 第四节课：Dimensionality Reduction（降维）
- **目标**：理解如何降低高维数据维度，同时保留信息
- **主要内容**
  - **PCA (Principal Component Analysis)**
    - 主成分分析，线性降维
    - 保留最大方差方向
  - **Locally Linear Embedding (LLE)**
    - 非线性降维，保留局部结构
    - 流形学习方法

### 第五节课：Clustering（聚类）
- **目标**：理解如何发现数据内在结构，无监督学习
- **主要内容**
  - **K-means**
    - 距离最小化划分簇
    - 随机初始化簇中心
  - **Expectation–maximization (EM) algorithm**
    - 软聚类方法
    - 高斯混合模型 (GMM) 参数估计
  - **K-means++**
    - 改进初始化，提高收敛速度

### 第六节课：Model Selection（模型选择）
- **目标**：理解如何选择最优模型，防止过拟合
- **主要内容**
  - **Overfitting phenomenon**
    - 模型在训练集表现很好，但在测试集表现差
  - **Complexity & Regularization**
    - L1 / L2 正则化控制模型复杂度
  - **Cross-validation**
    - K-fold 交叉验证，评估泛化能力

### 第七节课：MLE / MAP（最大似然 / 最大后验）
- **目标**：理解概率视角下模型参数估计
- **主要内容**
  - **Likelihood**
    - 数据给定模型参数的概率
  - **Log-likelihood**
    - 对数化方便求导和优化
  - **Maximum likelihood estimation (MLE)**
    - 最大化似然函数求参数
  - **Maximum a posteriori (MAP)**
    - 考虑先验信息的参数估计
  - **MLE for linear regression**
    - 与最小二乘回归对应
  - **MAP for linear regression**
    - 引入先验，实现正则化效果
  - **Bayesian linear regression**
    - 参数为概率分布，输出不确定性
  - **Gaussian Process**
    - 非参数贝叶斯方法，预测带置信区间

## 5. Q&A 与课堂互动（5分钟）
- 回答学生疑问
- 调查学生兴趣点
- 鼓励关注最新研究与应用