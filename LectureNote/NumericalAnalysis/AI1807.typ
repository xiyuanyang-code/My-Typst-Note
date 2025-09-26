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


== Python and Numerical Analysis

```python
def demo_2():
    print("0.1 + 0.2 == 0.3? ", (0.1 + 0.2 == 0.3))
    print(0.1 + 0.2)
```

The answer is:

```text
0.1 + 0.2 == 0.3?  False
0.30000000000000004
```

问题在于计算机中浮点数的存储方式，或者说，二进制的根源问题。

#recordings("About Binary and floating number")[
  In binary (or base-2), the only prime factor is 2, so you can only cleanly express fractions whose denominator has only 2 as a prime factor. In binary, 1/2, 1/4, 1/8 would all be expressed cleanly as decimals, while 1/5 or 1/10 would be repeating decimals. So 0.1 and 0.2 (1/10 and 1/5), while clean decimals in a base-10 system, are repeating decimals in the base-2 system the computer uses. When you perform math on these repeating decimals, you end up with leftovers which carry over when you convert the computer’s base-2 (binary) number into a more human-readable base-10 representation.
]

#example("使用 Python 求解相对有效数字")[
  ```python
  def get_significant_figure(ref: str, est: str) -> int:
      """计算实数估计值 est 相对于实数参考值 ref 的有效数字位数

      Args:
      ref (str): 实数参考值的字符串形式
      est (str): 实数估计值的字符串形式

      Returns:
      n (int): 有效数字位数
      """
      try:
          ref_val = float(ref)
          est_val = float(est)
      except ValueError:
          raise ValueError("输入必须是有效的实数字符串。")

      if ref_val == est_val:
          return 15
      if ref_val == 0:
          return 0

      error = abs(ref_val - est_val)

      if ref_val != 0:
          ref_magnitude = math.floor(math.log10(abs(ref_val)))
      else:
          return 0

      if error != 0:
          error_magnitude = math.floor(math.log10(error))
      else:
          return 15

      sig_fig = int(ref_magnitude - error_magnitude)
      last_sig_fig_magnitude = 10**error_magnitude
      if error < 0.5 * last_sig_fig_magnitude:
          sig_fig += 1

      return max(0, sig_fig)
  ```
]

= 插值法

== Definition

#definition("插值")[
  - 原函数 $f$ 在区间 $[a,b]$ 存在定义
  - $a <= x_0 < x_1 < dots < x_n <= b, P(x_i) = y_i$
]

- 多项式插值
- 分段插值
- 三角插值


== Lagrange 插值

考虑插值多项式 $P(x) in H_n$，可证明插值多项式的存在唯一性，即集合 $H_n$ 中有且仅有一个多项式满足插值多项式的定义。

#proof[
  - 使用范德蒙行列式
  - 转化为一个线性方程组问题
  - 范德蒙行列式不等于 0，说明该线性方程组有唯一解
]

=== 线性插值

- 直接求解上述的线性方程组来找到合适的解是困难且昂贵的。

- 这也是原始的解方程的办法。

- 并且因为唯一性，这样得到的解往往即为复杂，在实际情况中用处不大

因此，我们可以牺牲一些精度和准确性，采用更简单的线性插值

从基本的点斜式变形：

$ L_1(x) = (x_(k+1) - x)/(x_(k+1) - x_k) y_k + (x - x_k)/(x_(k+1) - x_k) y_(k+1) $

这可以看做是两个一次插值基函数的线性组合。

$ l_1(x) = (x_(k+1) - x)/(x_(k+1) - x_k) $
$ l_2(x) = (x - x_k)/(x_(k+1) - x_k) $

=== 抛物插值

考虑 $n=2$ 的二次函数插值拟合。

此时需要确定三个基函数。这些基函数都是二次函数，满足：

- 两个插值点的函数值为 0
- 剩下一个插值点的函数值为 1
- 这些插值函数因为确定了零点，很容易通过零点式求解唯一的缩放参数。
  - 规定剩下一个插值点的函数值为 1 的目的也就是作为基要标准化


$ l_(k-1)(x_(k-1)) = 1, l_(k-1)(x_j) = 0,(j = k,k+1) $

$ l_(k)(x_(k)) = 1, l_(k)(x_j) = 0,(j = k-1,k+1) $

$ l_(k+1)(x_(k+1)) = 1, l_(k+1)(x_j) = 0,(j = k-1,k) $

最终，我们可以得到插值的形式：

$ l_(k-1)(x) = ((x-x_k)(x-x_(k+1)))/((x_(k-1)-x_k)(x_(k-1)-x_(k+1))) $

这个函数的形式非常的简洁，也非常的直观，几乎是直接构造出来而不需要任何的运算技巧！

于是最终，我们就可以得到抛物插值的基本公式：

$ L_2(x) = y_(k-1)l_(k-1)(x) + y_(k)l_(k)(x) + y_(k+1)l_(k+1)(x) $

=== 一般化的插值多项式

考虑有 $n+1$ 个插值点的 $n$ 次插值多项式 $L_n(x)$：

$ l_j (x) = (product^(n)_(i=0) (x - x_i))/(product^(n)_(i=0) (x_j - x_i)) $

$ L_n(x) = sum^(n)_(i=0) y_i l_i (x) $

考虑 $omega_(n+1)(x) = product^(n)_(i=0) (x - x_i)$

$ omega'_(n+1)(x_k) = (x_k - x_0) dots (x_k - x_(k-1))(x_k - x_(k+1)) dots (x_k - x_n) $

$ L_n(x) = sum^(n)_(i=0) (y_i w_(n+1)(x)) / ((x - x_i) omega'_(n+1)(x_i)) $

=== 插值余项

#definition("插值余项")[
  $ R_n(x) = f(x) - L_n(x) $
]

#proposition("插值余项")[
  $ R_n(x) = f(x) - L_n(x) = (f^(n+1)(xi))/((n+1)!) w_(n+1)(x) $
]

== 逐次线性插值法

#recordings("拉格朗日插值的问题")[
  - 拉格朗日插值的精度达到了理论最优，实际上，他也给出了一种可行的求解唯一的插值函数的算法
  - 但是其最大的问题在于拉格朗日插值如果需要增加一个插值节点，这插值函数需要完全重新计算
    - 这在实际应用中会带来大量的计算资源的浪费
  - 我们希望插值的精度是不断提升的，
]

$I_(i_1, i_2, i_3, dots, i_n)(x)$ 为函数 $f(x)$ 关于节点 $x_(i_1), x_(i_2), dots, x_(i_n)$ 的 $n-1$ 次插值多项式。

#figure(
  image("aitken.png"),
  caption: [Aitken 逐次线性插值公式],
)

=== Aitken Interpolation

目标是找到 $n$ 阶插值多项式 $P_n (x)$

$P_(i,k)(x)$ 定义为通过点 $(x_i, y_i), (x_(i+1), y_(i+1)), dots, (x_k, y_k)$ 的 $k-i$ 阶插值多项式。

$P_(i,i)(x) = y_i$

#recordings("Like DP?")[
  - 怎么一股动态规划状态转移方程的味道
]

从零阶多项式出发，作为初始条件不断的归纳到更高阶的多项式。


$ P_(i, k)(x) = 1/(x_k - x_i) mat(delim: "|", P_(i, k-1)(x), x_i - x; P_(i+1, k)(x), x_k - x) $

$ = ( (x_k - x)P_(i, k-1)(x) - (x_i - x)P_(i+1, k)(x) ) / (x_k - x_i) $

== 差商与 Newton 插值公式

=== 差商

#definition("差商")[
  $ f[x_0, x_k] = (f(x_k) - f(x_0))/(x_k - x_0) $

  为函数的一阶差商。

  高阶差商的定义是递归定义而来的：

  $
    f[x_0, x_1, x_2, dots, x_k] = (f[x_0, x_1, x_2, dots, x_(k-2), x_k] - f[x_0, x_1, x_2, dots, x_(k-1)])/(x_k - x_(k-1))
  $
]

#recordings("Newton 插值")[
  - 对于 N 阶的可导函数来说，其形式和泰勒展开极为类似。
]



= Conclusion
