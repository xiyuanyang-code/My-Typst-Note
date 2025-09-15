# 机器学习课程 — 第五节课大纲（Clustering, 90分钟）

## 1. 导入与动机（10分钟）
- **为什么需要聚类？**
  - 数据自动分组，无监督学习
  - 探索数据结构与模式
  - 降噪、特征压缩、可视化
- **实际应用**
  - 客户分群（市场分析）
  - 图像分割
  - 异常检测
  - 社交网络社区发现

---

## 2. 问题表述：聚类的数学建模（15分钟）
- 数据矩阵 \(X \in \mathbb{R}^{n \times d}\)，\(n\) 个样本，\(d\) 维特征
- 聚类目标：
\[
\{C_1, \dots, C_K\} = \arg \min \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2
\]
- 核心概念：
  - 簇中心 \(\mu_k\)
  - 簇内距离平方和（SSE, Sum of Squared Errors）
  - 簇间相似性最小化
- 聚类评价指标：
  - SSE、轮廓系数（Silhouette Score）、Davies–Bouldin Index

---

## 3. K-means 聚类（30分钟）
### 3.1 算法推导
- 优化问题：
\[
\min_{\{\mu_k\}, \{C_k\}} \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2
\]
- 迭代求解：
  1. 固定簇中心 \(\mu_k\)，更新簇分配 \(C_k = \{ x_i: \|x_i - \mu_k\|^2 \le \|x_i - \mu_j\|^2, \forall j\}\)
  2. 固定簇分配，更新簇中心 \(\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i\)
- 收敛条件：SSE 不再下降

### 3.2 优缺点
- 优点：简单、迭代收敛快
- 缺点：局部最优、对初始簇中心敏感、必须指定簇数 \(K\)

---

### 4. K-means++ 初始化（10分钟）

- **目标**：改进 K-means 对初始簇中心敏感的问题

- **算法步骤**：
  1. 随机选择第一个簇中心
  2. 后续中心按距离平方概率选择：
\[
P(x_i) = \frac{D(x_i)^2}{\sum_j D(x_j)^2}, \quad D(x_i) = \min_k \|x_i - \mu_k\|
\]

- **数学直觉与证明**：
  - 定义目标函数：簇内平方和（SSE）
  \[
  \Phi = \sum_{i=1}^{n} \min_{k} \|x_i - \mu_k\|^2
  \]
  - K-means++ 选择中心的概率与点距离平方成正比：
    - 离已有中心更远的点被选中概率更大
    - 避免中心过于集中，减少局部最优风险
  - **理论保证**（Arthur & Vassilvitskii, 2007）：
    - 使用 K-means++ 初始化，期望得到的 SSE 不超过最优 SSE 的 \(O(\log K)\) 倍
    \[
    \mathbb{E}[\Phi_\text{init}] \le 8(\ln K + 2) \Phi_\text{OPT}
    \]
    - 这里 \(\Phi_\text{OPT}\) 是最优簇划分的 SSE
  - 直观理解：
    - 距离较远的点被优先选择为中心 → 保证簇中心覆盖数据空间
    - K-means 迭代收敛到局部最优时，初始 SSE 已经比较低，效果更好

- **优点**：
  - 加快收敛，提高聚类质量
  - 避免部分簇空或者簇中心过近
- **Python 实现**：

---

## 5. Expectation–Maximization (EM) 聚类（20分钟）

### Motivation: EM 与 K-means / KNN 的关系
- **背景**：
  - K-means 是硬分配聚类：每个点被分配到最近的簇中心  
  - KNN 思路：利用局部邻居的距离关系进行预测或划分
- **联系**：
  - EM 的概率分配是 K-means 的软化推广：
    - K-means 假设各簇方差相同且趋近于零  
    - 每个点被赋予 0/1 的归属权重（硬分配）  
    - EM 则通过概率分布对每个点分配软权重
- **直观理解**：
  - K-means = 特殊条件下的 EM（方差相同且趋近 0）
  - EM 可以捕捉簇间模糊边界，并处理不同形状/大小的簇
- **图示**：
  - 硬分配 vs 软分配示意图
  - 最近邻思想对应硬分配，概率分配对应软分配

### 高斯混合模型（GMM）
- 假设数据来自 \(K\) 个高斯分布的混合：
\[
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)
\]
- 目标：最大化对数似然：
\[
\mathcal{L}(\pi, \mu, \Sigma) = \sum_{i=1}^{n} \log \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i|\mu_k, \Sigma_k)
\]

### EM 算法推导
1. **E-step（期望步）**：
\[
\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i|\mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i|\mu_j, \Sigma_j)}
\]
2. **M-step（最大化步）**：
\[
\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}, \quad
\Sigma_k = \frac{\sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_i \gamma_{ik}}, \quad
\pi_k = \frac{\sum_i \gamma_{ik}}{n}
\]

### 优缺点
- 优点：软分配，可捕捉复杂数据分布
- 缺点：初值敏感，局部最优，计算量大

---

## 6. 小结与对比（10分钟）
- **算法比较**
  - K-means：硬分配，快速
  - K-means++：优化初始化，稳定
  - EM/GMM：软分配，概率模型，更灵活
- **课堂互动**
  - 提问：如何选择簇数 \(K\)？
  - 小练习：在模拟数据上比较 K-means 与 EM 聚类结果