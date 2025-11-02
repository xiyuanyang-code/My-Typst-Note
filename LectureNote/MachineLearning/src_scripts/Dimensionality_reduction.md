# 机器学习课程 — 第四节课大纲（降维 Dimensionality reduction, 90分钟）

# 第三节课：Dimensionality Reduction（降维）

## 1. 导入与动机（10分钟）
- **为什么需要降维？**
  - 高维数据的挑战：
    - 维度灾难（curse of dimensionality）
    - 计算复杂度高
    - 可视化困难
  - 降低噪声，提高模型泛化能力
  - 提升数据可视化与理解性
- **实际应用**
  - 人脸识别（高维像素数据 → 低维特征空间）
  - 文本数据降维（词向量压缩）
  - 数据可视化（2D / 3D 展示高维结构）
- **课堂互动**
  - 提问：学生思考生活中有哪些高维数据？

---

## 2. 问题表述：降维的数学建模（15分钟）
- **输入数据**  
  - 数据矩阵 \(X \in \mathbb{R}^{n \times d}\)，\(n\) 个样本，\(d\) 维特征
- **目标**  
  - 寻找一个低维表示 \(Z \in \mathbb{R}^{n \times k}\) （\(k \ll d\)）
  - 保留原始数据的“信息”或“结构”
- **两类主要方法**
  - **线性降维**（PCA）
  - **非线性降维**（流形学习 manifold learning，如 LLE）

---

## 3. 主成分分析（Principal Component Analysis, PCA, 40分钟） https://www.youtube.com/watch?v=FD4DeN81ODY&list=PLqwozWPBo-FtNyPKLDPTVDOHwK12QbVsM&index=4

### 直观解释
- 在高维空间中找到方差最大的方向
- 最大化数据投影方差，同时最小化重建误差
- 目标：压缩信息，同时尽量不丢失主要特征

### 数学建模
- **数据中心化**：\(\tilde{X} = X - \bar{X}\)
- **协方差矩阵**：\(S = \frac{1}{n} \tilde{X}^T \tilde{X}\)
- **优化目标**：寻找特征向量 \(w\) 最大化方差 
- **求解方法**：特征值分解（Eigen-decomposition）
- **矩阵形式**：

### Gram trick （小样本高维处理）**

### 实例 & Python 代码
- 人脸识别 eigenfaces

### Kernel PCA （简要）
#### 背景
- 标准 PCA 只能捕捉数据的线性结构
- 核 PCA 通过**核技巧（Kernel Trick）**在高维特征空间中进行线性 PCA，实现非线性降维
#### 基本思想

### 优缺点
- 简单、直观、计算效率高
- Gram trick 适合小样本高维数据
- Kernel PCA 可处理非线性结构
- PCA 对非线性流形不敏感
- Kernel PCA 计算量大，对核函数敏感
---

## 4. 局部线性嵌入（Locally Linear Embedding, LLE, 20分钟））
https://www.youtube.com/watch?v=scMntW3s-Wk
### 动机
- 数据可能位于高维空间的低维流形上
- PCA 捕捉不到弯曲的非线性流形

### 基本思想
- 每个数据点可以由其邻居的线性组合表示,
- 在低维空间中保持这种线性关系

### 数学建模
- 局部重建权重
- 低维嵌入

### 实例 & Python 代码
- "Swiss Roll" 数据集降维可视化

### 优缺点
- 优点：简单直观、计算效率高、广泛应用
- 缺点：仅能捕捉线性结构，对非线性数据不足

## 5. 其他降维方法（10分钟））

### Isomap
  - 基于最短路径保持全局几何结构
  - 核心思想：
    - 构建邻居图
    - 计算各点间最短路径距离（geodesic distance）
    - 多维尺度分析（MDS）实现低维嵌入
  - 优点：捕捉全局非线性结构
  - 缺点：对噪声敏感，计算大规模数据成本高

### t-SNE (t-Distributed Stochastic Neighbor Embedding)**
  - 高维数据可视化的流行方法
  - 核心思想：
    - 将高维数据点的相似度转换为概率分布
    - 在低维空间中保持相似概率分布
  - 优点：二维/三维可视化效果直观
  - 缺点：不适合大规模数据降维，只适合可视化，参数调节影响大

## 5. 小结与对比（10分钟）

### PCA vs LLE
- PCA：线性方法，适合全局线性结构
- LLE：非线性方法，适合局部流形结构

### 思考
- 高位数据点的本征维度 (intrinsic dimension)

### 下一步
- Clustering（聚类）：如何在降维后的空间中发现数据的内在簇结构
