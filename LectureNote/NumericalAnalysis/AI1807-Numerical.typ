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

==== Quick Judgement

- 对于四舍五入的有效数字评判（这也是一般情况），可以按照数数位的方式进行判断有效数字的位数 $n$

- 有效数字的设计和科学计数法无关，可以实现科学计数法的归一化
$ x^* = plus.minus 10^m (a_1.a_2 a_3a_4 dots a_n) $

==== Strict Definition

Given the original number $x$ and the truncated number $x^*$:

$
  x^* = plus.minus 10^m (a_1.a_2 a_3a_4 dots a_n), a_i in {0,1,2,dots,9}
$

$ epsilon^*_x = |x - x^*| <= 1/2 times 10^(m-n+1) $

- 相对误差限：$1/2 times 10^(m-n+1)$ 为有效数字定义的相对误差限。
- 有效数字：$n$

#recordings("Significant figures")[
  - 找有效数字 $n$ 的方法和高中一样
  - 找移位 $m$ 的方法就是转变成科学计数法
  - 找相对误差限 $m+n-1$ 看小数点后有几位数字
]

#example("An Example for significant figure")[
  考虑 $x = 3.14159265357$ and different $x^*$:

  - $x^* = 3.1416$:
    - $x^* = 10^0 times 3.1416$, $m = 0$
    - $abs(x - x^*) approx 0.0000073 <= 0.00005 = 1/2 times 10^(-4)$, $m-n+1 = -4$
    - $n = 5$

  - $x^* = 3.1415$:
    - $x^* = 10^0 times 3.1415$, $m = 0$
    - $abs(x - x^*) approx 0.0000927 <= 0.0005 = 1/2 times 10^(-3)$, $m-n+1 = -3$
    - $n = 4$
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

= Floating Number

== 浮点数的十进制和二进制表示

#recordings("Binary")[
  $
    sum^(+infinity)_(-infinity) b_i 2^i
  $

  The number $dots b_2 b_1 b_0.b_(-1) b_(-2) dots$ forms the binary digit for the original number.
]

- 具体的计算过程中，可以不断的乘除2并从小数点向左向右记录余项0或者1
- 在很多情况下，小数部分的计算会形成循环，即无限循环小数
  - 这也是为什么有些在十进制下的有限小数在计算机存储中成为了二进制下的无限不循环小数，带来的精度误差
  - 因此在具体的转换过程中可以使用方程求解

#example("For the digit?")[
  Suppose $x = 0.3$:
  - $0.3 * 2 = 0.6$: mod 0
  - $0.6 * 2 = 1.2$: mod 1
  - $0.2 * 2 = 0.4$: mod 0

]

== 实数在计算中的浮点表示

在 Python 中，默认为双精度 fp64 的表示方式。

=== IEEE 754 二进制浮点数算数标准

#figure(
  image("images/fp64.png"),
)

为了方便比较大小，我们希望将不同的浮点数直接看做是有符号整数来比较大小。在未经过移码之前，指数部分的取值范围是：

$
  - 2^(E-1) + 2 <= e <= 2^(E-1) -1
$

$
  "Bias" = 2^(E-1) -1\
  E' = e + "Bias"\
  1 <= E' <= 2^E - 2
$

在加上这一部分后，指数部分的数值范围变成 $[1, 2^M -2]$.

补码表示法中，负数的最高位是 1，正数的最高位是 0。因此，负数的补码值在无符号比较时会大于正数。这会打乱数值的大小关系。而加上移码之后，就更方便比较。

#recordings("+0/-0")[
  - 在浮点数中，将指数和尾数部分设置为全0就可以得到0值
  - 但是符号位可以选择正或者负：
    - $+0$: Sign=1
    - $-0$: Sign=0
  - 更方便计算正无穷和负无穷
]

在计算中，通常受到字长的限制，需要将尾数部分截断。

=== 浮点数的算术运算误差

在实际运算过程中，浮点数的加法和乘法均会带来舍入误差 Round-of Error.

==== 浮点数加法

#definition("Plus")[
  - 补齐阶码
    - 严重的舍入误差出现在这一步！
  - 尾数求和
  - 正规化
]

#figure(
  image("images/fp-2.png"),
)

==== 浮点数乘法

#definition("浮点数乘法")[
  $
    g_3 = (1+f_1) times (1+f_2)\
    p_3 = p_1 + p_2\
    s_3 = s_1 xor s_2
  $

  可以看到，浮点数乘法是相当昂贵的操作。
]

#recordings("Error Analysis")[
  - 大数吃小数的过程中会导致在阶码对齐的过程中损失相当多的精度

  - $x >> y$: $x/y$ is not recommended!
    - Overflow
  - $x approx y$: $x - y$ is not recommended!
  - 运算顺序要注意，避免大数吃小数
  - 使用相关数学运算技巧，例如分母分子有理化等。不过先估值估计误差很重要。
]

= 插值法

== Definition

#definition("插值")[
  - 原函数 $f$ 在区间 $[a,b]$ 存在定义
  - $a <= x_0 < x_1 < dots < x_n <= b, P(x_i) = y_i$
]

- 多项式插值
  - 线性插值
  - 抛物线插值
  - 分段插值
  - 样条插值
- 三角插值
- 有理插值


== Lagrange 插值

考虑插值多项式 $P(x) in H_n$，可证明插值多项式的存在唯一性，即集合 $H_n$ 中有且仅有一个多项式满足插值多项式的定义。

#proof[
  - 使用范德蒙行列式
  - 转化为一个线性方程组问题
  - 范德蒙行列式不等于 0，说明该线性方程组有唯一解
]

#recordings("Using Vandermonde Matrix")[
  - 这个是一般的解法。
  - 只要保证每一个采样点的 $x$ 值不同，就可以保证左侧的范德蒙行列式不为零，对应的范德蒙矩阵可逆
  - 因此可以唯一计算具体的多项式的参数向量 $[a_0, a_1, dots, a_(n-1)]$
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

$ l_j (x) = (product^(n)_(i=0) (x - x_i)(i != j))/(product^(n)_(i=0) (x_j - x_i)) $

$ L_n(x) = sum^(n)_(i=0) y_i l_i (x) $

考虑 $omega_(n+1)(x) = product^(n)_(i=0) (x - x_i)$

$ omega'_(n+1)(x_k) = (x_k - x_0) dots (x_k - x_(k-1))(x_k - x_(k+1)) dots (x_k - x_n) $

$ L_n(x) = sum^(n)_(i=0) (y_i w_(n+1)(x)) / ((x - x_i) omega'_(n+1)(x_i)) $

#recordings("Definition for Lagrange")[
  - 从原理上看，本质上只是找到对应多项式空间的一组基，在范德蒙行列式中，因为 coefficient 被直接求出，相当于使用了 ${x^i}$ 的一组基本基。
  - 对于拉格朗日求解的方法，本质上就是求解一组基向量，在给定系数的情况下
  $
    l_j(x) = cases(1 "if" k = j, "0" "if" k!= j)
  $
]

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
  image("images/aitken.png"),
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

$
  f(x) = f(x_0) + f[x, x_0] (x - x_0)
$

Until... (This is the definition of that!)

$
  f[x, x_0, dots, x_(n-1)] = f[x_0, x_1, dots, x_n] + f[x, x_0, dots, x_n] (x - x_n)
$

从 $f(x) = f(x_0) + f[x, x_0] (x - x_0)$，将差商不断展开到高价差商，直到写成如下的形式：


$
  f(x) = f(x_0) + sum_(i = 1)^n f[x_0, x_1, dots, x_i] product_(j=0)^(i-1)(x - x_j) + f[x, x_0, dots, x_n] product_(j=0)^(n)(x - x_j)
$

#recordings("差商的几何意义")[
  - 对于二阶差商，对于连续可导函数来说，二阶差商的极限值就是函数在该点处的导数

  $
    lim_(h arrow 0) f[x_0, x_0 +h] = f'(x_0)
  $

  - 对于更高阶的差商

  $
    lim_(x_0, dots, x_n arrow x) f[x_0, dots, x_n] = (f^((n))(x))/(n!)
  $
]

#recordings("差商的更清晰的代数性质")[
  - 差商具有轮换对称性
  $
    f[x_0, x_1, x_2, dots, x_k] &= (f[x_0, x_1, x_2, dots, x_(k-2), x_k] - f[x_0, x_1, x_2, dots, x_(k-1)])/(x_k - x_(k-1))\
    &= (f[x_1, x_2, dots, x_k] - f[x_0, x_1, x_2, dots, x_(k-1)])/(x_k - x_0)
  $

  - 差商可以看成函数值的线性组合

  $
    f[x_0, x_1, x_2, dots, x_k] = sum^k_(j=0) f(x_j)/(product_(i=0)^k (x_j - x_i)(i != j))
  $
]

=== 差商表的计算

== Several Codes for interpolation

=== Using Vandermonde Matrix

```python
import numpy as np
import matplotlib as mlp
mlp.use("Agg")
import matplotlib.pyplot as plt
import time
from tqdm import trange
from typing import List, Tuple, Callable, Dict, Any, Union
from scipy.interpolate import lagrange, KroghInterpolator


class InterpolationSolver:
    """
    A class for solving polynomial interpolation problems.

    This solver uses various methods to find the coefficients of the unique
    polynomial that passes through a given set of points. It supports
    pluggable methods and includes time measurement for the solution process.
    """

    def __init__(self, methods: Dict[str, Callable] = None):
        """
        Initializes the Interpolation Solver.

        The default method provided is 'vandermonde' (using the Vandermonde matrix).

        Args:
            methods: A dictionary where keys are the method names (str) and
                     values are the corresponding solving functions (Callable).
                     The signature of a solving function should be:
                     f(points: List[Tuple[float, float]]) -> np.ndarray.
                     Custom methods will be merged with the default ones.
        """
        # Default method: Vandermonde matrix solution
        # * will add more solvers in the future
        self.methods: Dict[str, Callable] = {
            "vandermonde": self._solve_vandermonde,
            "lagrange": self._solve_lagrange,
            "lagrange_fast": self._solve_lagrange_fast,
        }

        if methods:
            self.methods.update(methods)

        self.last_result: Union[np.ndarray, None] = None
        self.last_method: Union[str, None] = None
        self.last_time: Union[float, None] = None

    def _solve_vandermonde(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Solves for the polynomial coefficients using the Vandermonde matrix method.

        For n+1 data points, the method solves the linear system V * a = y,
        where V is the Vandermonde matrix and a is the vector of coefficients.

        Args:
            points: A list of (x, y) coordinate tuples. Must contain at least
                    one point.

        Returns:
            np.ndarray: The array of polynomial coefficients, ordered from
                        highest degree to lowest degree:
                        [a_n, a_{n-1}, ..., a_1, a_0]
                        for the polynomial P(x) = a_n * x^n + ... + a_0.
                        Returns an empty array if no points are given.

        Raises:
            ValueError: If the linear system is singular (e.g., duplicate x-values
                        or ill-conditioned data), preventing a unique solution.
        """
        n_points = len(points)
        if n_points == 0:
            return np.array([])

        n_degree = n_points - 1  # Degree of the polynomial

        # Separate x and y coordinates
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        V = np.vander(x, n_points)

        # Solve the linear system V * a = y
        try:
            coefficients = np.linalg.solve(V, y)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                f"Failed to solve the linear system. The matrix "
                f"might be singular (e.g., duplicate x-values). Error: {e}"
            )

        return coefficients

    def _solve_lagrange_fast(self, points: List[Tuple[float, float]]) -> np.ndarray:
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        coeff = lagrange(x, y).coef
        return coeff

    def _solve_lagrange(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        使用拉格朗日插值法展开并求和，以获得标准多项式系数。

        P(x) = sum_{j=0}^{n} y_j * l_j(x)，其中 l_j(x) 是拉格朗日基多项式。

        Args:
            points: 包含 (x, y) 坐标点的列表。

        Returns:
            np.ndarray: 多项式系数数组，顺序为从高次到低次：
                        [a_n, a_{n-1}, ..., a_1, a_0]。
        """
        n_points = len(points)
        if n_points == 0:
            return np.array([])

        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])

        # 初始化最终的多项式系数为零。系数顺序: [a_n, ..., a_0]
        final_coeffs = np.zeros(n_points)

        # 迭代计算每个基多项式 l_j(x) 的贡献
        for j in range(n_points):
            x_j = x[j]
            y_j = y[j]

            # compute pi (x - x_j) (k != j)
            roots = np.delete(x, j)

            # 使用 np.poly() 计算多项式 N_j(x) 的系数 (即 x - r1)(x - r2)...
            # 返回的系数顺序是 [a_m, a_{m-1}, ..., a_0]
            numerator_coeffs = np.poly(roots)

            # 计算分母 D_j = product_{k != j} (x_j - x_k)
            denominator = np.prod(x_j - roots)

            if np.isclose(denominator, 0):
                raise ValueError("Error, repeated x value is found.")

            # l_j(x) 的系数 = N_j(x) 的系数 / D_j
            l_j_coeffs = numerator_coeffs / denominator
            term_coeffs = l_j_coeffs * y_j
            final_coeffs = np.polyadd(final_coeffs, term_coeffs)

        return final_coeffs

    def _format_polynomial(self, coefficients: np.ndarray) -> str:
        """
        Formats the polynomial coefficients into a readable string representation.

        Args:
            coefficients: The array of polynomial coefficients
                          [a_n, a_{n-1}, ..., a_0].

        Returns:
            str: The string representation of the polynomial, e.g., "1.0x^3 + 2.0x + 1.0".
        """
        if len(coefficients) == 0:
            return "0"

        terms = []
        n_degree = len(coefficients) - 1

        for i, a in enumerate(coefficients):
            degree = n_degree - i

            if np.isclose(a, 0):
                continue

            sign = " + " if a > 0 else " - "
            if not terms:
                sign = "" if a > 0 else "-"

            abs_a = abs(a)
            coeff_str = f"{abs_a:.6f}"
            if np.isclose(abs_a, 1) and degree != 0:
                coeff_str = ""

            if degree == 0:
                # Constant term
                term_str = f"{sign}{abs_a:.6f}"
            elif degree == 1:
                # x term
                term_str = f"{sign}{coeff_str}x"
            else:
                # x^k term
                term_str = f"{sign}{coeff_str}x^{degree}"

            terms.append(term_str)

        return "".join(terms).strip() or "0"

    def solve(
        self, points: List[Tuple[float, float]], method: str = "vandermonde"
    ) -> Dict[str, Any]:
        """
        The core function to solve the polynomial interpolation using the specified method.

        The function measures the time taken by the chosen solving method.

        Args:
            points: A list of (x, y) coordinate tuples for interpolation.
            method: The name of the interpolation method (str) to use, which
                    must exist as a key in `self.methods`. Defaults to "vandermonde".

        Returns:
            Dict[str, Any]: A dictionary containing the solution results:
                            - 'coefficients': np.ndarray of the polynomial coefficients.
                            - 'method': The name of the method used.
                            - 'time_s': The elapsed time for the calculation in seconds.
                            - 'polynomial_str': A readable string of the polynomial P(x).

        Raises:
            ValueError: If the specified `method` is not recognized.
            ValueError: If the underlying solver function fails (e.g., singular matrix).
        """
        if method not in self.methods:
            raise ValueError(
                f"Unknown interpolation method: '{method}'. "
                f"Available methods: {list(self.methods.keys())}"
            )

        solver_func = self.methods[method]

        if not points:
            return {
                "coefficients": np.array([]),
                "method": method,
                "time_s": 0.0,
                "polynomial_str": "0",
            }

        start_time = time.time()
        coefficients = solver_func(points)
        end_time = time.time()
        elapsed_time = end_time - start_time
        poly_str = self._format_polynomial(coefficients)

        # Store results in the cache
        self.last_result = coefficients
        self.last_method = method
        self.last_time = elapsed_time

        return {
            "coefficients": coefficients,
            "method": method,
            "time_s": elapsed_time,
            "polynomial_str": poly_str,
        }


if __name__ == "__main__":
    solver = InterpolationSolver()
    NUM_POINTS = 50
    NUM_RUNS = 100
    TEST_METHODS = ["vandermonde", "lagrange", "lagrange_fast"]
    results_by_method = {method: [] for method in TEST_METHODS}

    print(f"--- Efficiency Test ---\nPoints N: {NUM_POINTS}, Repeated Nums: {NUM_RUNS}")

    for run in trange(NUM_RUNS):
        points_data = [
            (np.random.random() * 100, np.random.random() * 100)
            for _ in range(NUM_POINTS)
        ]

        for method_name in TEST_METHODS:
            try:
                result = solver.solve(points_data, method=method_name)
                results_by_method[method_name].append(result["time_s"])

            except np.linalg.LinAlgError:
                print(
                    f"Warning: {method_name} failed at run {run+1} due to singular matrix."
                )
                continue
            except ValueError as e:
                print(f"Warning: {method_name} failed at run {run+1} with error: {e}")
                continue

    print("\n--- Efficiency Test End ---")

    vandermonde_times = results_by_method["vandermonde"]
    lagrange_times = results_by_method["lagrange"]
    lagrange_fast_times = results_by_method["lagrange_fast"]

    # 打印统计摘要
    print("\n--- Cost Time Count ---")
    for method, times in results_by_method.items():
        if times:
            print(
                f"  {method.ljust(15)}: Mean = {np.mean(times):.6f}, Median = {np.median(times):.6f}, Min = {np.min(times):.6f}, Max = {np.max(times):.6f} (N={len(times)})"
            )
        else:
            print(f"  {method.ljust(15)}: No data recorded.")

    # 绘制箱线图
    data_to_plot = [vandermonde_times, lagrange_times, lagrange_fast_times]
    labels = TEST_METHODS

    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, vert=True)

    plt.title(
        f"Polynomial (N={NUM_POINTS}, {NUM_RUNS} repetitions)", fontsize=14
    )
    plt.ylabel("Eecutions times", fontsize=12)
    plt.xlabel("Method", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("./images/interpolation.pdf")
```