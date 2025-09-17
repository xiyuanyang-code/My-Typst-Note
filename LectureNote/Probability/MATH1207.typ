#import "@preview/dvdtyp:1.0.1": *
#show: dvdtyp.with(
  title: "MATH1207 Probability",
  author: "Xiyuan Yang",
  abstract: [Lecture and Summary Notes for MATH 1207: *Probability*],
)
#show link: set text(fill: blue, weight: 700)
#show link: underline
#show ref: set text(fill: blue, weight: 500)


// comment for hiding outlines
#outline()

// uncomment for changing to a new page
// #pagebreak()

= Introduction




= Chapter1: Random Events and Probability

== 基本概念

*Axiomatic Probability Theory*

=== Probabilities as Sets

#definition("Basic Definition for probability")[
  - 样本点
    - 单个样本点之间存在互斥性，类似于集合中的 partition。
  - 样本空间：由样本点组成的集合 $Omega$
  - 基本事件：单个样本点组成的集合 ${x} in cal(P)(Omega), x in Omega$
  - 随机事件 $A in cal(P)(Omega)$
]

#recordings("Probabilities as sets")[
  用集合论的眼光来看概率的定义
]

==== 事件运算的化简

- 巧用各种公式进行集合运算
- 注意括号和集合之间的分配率

$ A - B = A overline(B) = A - A B $

- *Attention!*，注意 $overline(A B)$ and $overline(A) text(" ")overline(B)$
  - $overline(A B) = overline(A inter B)$
  - $overline(A) text(" ")overline(B) = overline(A union B)$

=== Axiomatic Probability Theory

- 概率本身来源于测度论，或者说概率就是一种概率测度，并且是一种满足$P(Omega) = 1$ 的特殊测度。

- 测度论的三个公理条件

== Classical Probability

在分析古典概型使用不要忘了下面几点：

#recordings("classical probability")[
  - 古典概型必须保证两个前提条件
    - $Omega$ 为有限集
      - 测度就是元素的个数
    - 保证每个样本点的概率都是相同的

  - 解题思路：
    - 判断是否等可能，考虑使用几何概型
      - 这一步最关键的就是建模出有限的样本空间和等可能的样本点
    - 计算不同时间的测度
      - 计算 $Omega$ 的测度
      - 计算 $A$ 的测度
  
  - 相关模型
    - 抽球问题：
      - 放回：二项分布
      - 不放回：超几何分布
    - 抽签模型
      - 等可能性
      - 使用排列组合求解
    - 球入盒子模型
      - 可以考虑计算对立事件的概率
      - 看清楚题目是否指定盒子 和 指定球，这将决定是否需要再额外乘一个组合数（从不指定要指定涉及一个先选球的排列问题）
      - 生日悖论
]


== Geometric Probability

几何概型作为古典概型的延伸，具体的延伸在于将样本空间的数量提升到了无限。但是此时基本事件的概率变为 0，因此无法比较等可能性，因此定义如下：

#definition("Geometric Probability")[
  - 存在有限区域 $Omega$
    - 区域可以是 1D 2D 3D
  - 对于区域测度的度量就是对应的面积、体积和长度等等
  - 关键：落入子区域 $G$ 的概率和区域 $G$ 的测度成正比。
]

#recordings("Recordings for Geometric Probability")[
  - 有关几何概型的具体结论和思考
  - 同样的，识别出几何概型之后，往往需要将题目中的事件约束条件刻画为一个不等式或者其他约束条件
    - 再将这个约束条件的测度转化为一个区域
    - 有点类似于线性规划问题
]



= Conclusion