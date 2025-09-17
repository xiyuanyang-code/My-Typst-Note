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

== Definition

- 数值计算方法、理论和计算实现
- 作为计算数学的一部分
- 精读和误差分析在计算机领域至关重要。

== Error

=== Definition

误差来源：
- 模型误差（建模时产生）
- 观测误差
- 截断误差 & 方法误差 (Truncation Error)
  - 求近似解
- 舍入误差 (RoundOff Error)
  - 机器字长有限

=== Several Examples for Error


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

#example("Error for integrate")[
  在实际计算的过程中，往往需要考虑更多和理论计算有差异的部分，例如：

  $ I_n = 1/e integral^1_0 x^n text("d") x $

  We have:

  $ I_n = 1 - n I_(n-1) $

  - 在实际计算中，如果从 $I_0$ 开始计算，因为多次乘法操作的实现，会导致在 $n$ 非常大的时候，浮点数误差很大，精度低。
  - 精度更高的方法：估值

  $ I_(n-1) = 1/n (1 - I_n) $
    - 首先误差分析确定上下限：$1/e(n+1) < I_n < 1/(n+1)$
    - 取中间值进行估计，再倒回去计算到 $I_0$
    - 虽然进行了很多次乘法操作，但是在这个操作中误差逐渐减小。 

  #recordings("Explanation")[
    - 解释：在正向递推式中，浮点数乘法带来的误差 $epsilon$ 会随着 $n$ 的增大而不断的被放大
    - 但是在逆向递推式中，一开始的插值误差因为 $1/n$ 的缩减效应导致其被减小。 
  ]
]


=== Absolute Error

$ e^* = x^* - x $

误差限：误差的绝对值的上界

$ |x - x^*| <= epsilon^* $


=== Relative Error

$ e_r^* = e^* / x = (x^* - x) /x $

实际计算中通常取 $x^*$ 的实际值作为分母。 

相对误差限：

$ epsilon^*_r = epsilon^* / (|x^*|) $


=== Significant Figures

- 有效数字的设计和科学计数法无关，可以实现科学计数法的归一化
$ x^* = plus.minus 10^m (a_1.a_2 a_3a_4 dots a_n) $

上式中的 $m$ 代表着一个数字转化成科学计数法的表现形式需要提取的 10 的幂次。 

计算误差限要看小数点后的位数

$ epsilon^*_x = |x - x^*| <= 1/2 times 10^(m-n+1) $

$ 1/2 times 10^(m-n+1) $ 为有效数字定义的相对误差限。而我们 $n$ 就是有效数字。

#recordings("Significant figures")[
  - 找有效数字 $n$ 的方法和高中一样
  - 找移位 $m$ 的方法就是转变成科学计数法
  - 找相对误差限 $m+n-1$ 看小数点后有几位数字
]

= Conclusion