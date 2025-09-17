#import "@preview/dvdtyp:1.0.1": *
#show: dvdtyp.with(
  title: "AI1807 Numerical Analysis",
  author: "Xiyuan Yang",
  abstract: [Lecture Notes and Code for AI 1807, Numerical Analysis],
)
#show link: set text(fill: blue, weight: 700)
#show link: underline
#show ref: set text(fill: blue, weight: 500)


// comment for hiding outlines
#outline()

// uncomment for changing to a new page
// #pagebreak()

= Introduction

== Preliminaries

```python
print("Hello world!")
```

== Definition

- 数值计算方法、理论和计算实现
- 作为计算数学的一部分

== Error

误差来源本身多种多样：
- 模型误差（建模时产生）
- 观测误差
- 截断误差 & 方法误差 (Truncation Error)
  - 求近似解
- 舍入误差 (RoundOff Error)
  - 机器字长有限


#example("Error in Polynomial Computation")[
  - 直接计算会导致多次昂贵且无意义的乘法操作
  - 使用秦九韶算法可以减少乘法操作的次数
  - 更优的算法优化：因式分解
]

#example("Solving Matrix")[
  求解 $A x = b$, we need: 

  - 使用克莱姆法则，则求解 $n$ 个未知数需要 n+1 次矩阵行列式运算。
  - 基于代数余子式的计算行列式的方法达到了 $O(n!)$ 的时间复杂度
  - 行列式计算优化：$O(n^3)$ 
]

更少的运算次数往往意味着更少的误差！

=== 相对误差

$ e_r^* = e^* / x = (x^* - x) /x $

相对误差限：

$ epsilon^*_r = epsilon^* / (|x^*|) $


=== 有效数字

- 有效数字的设计和科学计数法无关，可以实现科学计数法的归一化
$ x^* = 10^m (a_1.a_2 a_3a_4 dots a_n) $

计算误差限要看小数点后的位数

$ epsilon^*_x = |x - x^*| <= 1/2 times 10^(m-n+1) $

= Conclusion