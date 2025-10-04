#import "@preview/dvdtyp:1.0.1": *
#show: dvdtyp.with(
  title: "MIT6.046J Design and Analysis of Algorithms",
  author: "Xiyuan Yang",
  abstract: [Lecture Notes for advanced algorithms for Open lecture MIT 6.046J],
)
#show link: set text(fill: blue, weight: 700)
#show link: underline
#show ref: set text(fill: blue, weight: 500)


// comment for hiding outlines
#outline()

// uncomment for changing to a new page
// #pagebreak()

= Introduction

== Course Overview

1. Divide and Conquer - FFT, Randomized algorithms
2. Optimization - greedy and dynamic programming
3. Network Flow
4. Intractibility (and dealing with it)
5. Linear programming
6. Sublinear algorithms, approximation algorithms
7. Advanced topics

== Complexity Recall

- *P*: class of problems *solvable* in polynomial time. $O(n^k)$ for some constant $k$.
  - P 类问题可以使用确定性图灵机在多项式时间内解决的问题集合

- *NP*: class od problems *verifiable* in polynomial time.

#example("Hamiltonian Cycle")[
  Find a simple cycle to contain each vertex in $V$.
  - Easy to evaluate, but hard to calculate!

  We have $"P" subset "NP"$.
  - 在多项式时间内找到正确答案，那么肯定可以在多项式时间内验证答案是否正确。（默认比较两个答案是否相同是可以在多项式实现内实现的）
]

- *NPC*: NP Complete
  - 问题本身属于 NP 复杂度类
  - 为 NP 困难问题：
    - 所有的 NP 问题都可以在多项式时间内归约到问题 C 上。

== Interval Scheduling

Requests $1,2,dots,n$: single resource.

- $s(i)$: the start time
- $f(i)$: the finish time
- $s(i) < f(i)$
- two requests are compatible: $[s(i), f(i)] inter [s(j), f(j)] != emptyset$

Goal: select a compatible subset of requests with the maximum size.

Solving: Greedy Search!

- Use a simple rule to select a request i.
- Reject all requests incompatible with i.
- Repeat until all requests are processed.

= Divide and Conquer

== Paradigm

*Intuition*: Splitting bigger problems into smaller problems.

- Solve the sub-problems recursively
- Combine solutions of sub-problems to get overall solutions.

$
  T(n) = a T(n/b) + ["work for merge"]
$

- $a$: The number of sub-problems during recursion.
- $b$: The size of each sub-problems

For example, for the merge sort:

$
  T(n) = a T(n/2) + O(n)
$

== Convex Hall

=== Brute force for Convex Hull

$C^2_n$ segments, testing each segment:

- All other points are on the single side: correct
- Else: false

Time Complexity: $O(n^3)$

=== Gift Wrapping Algorithms

Given $n$ points in the plane, the goal is to find the smallest polygon containing all points in $S = {(x_i, y_i)|i = 1,2,dots,n}$. We ensure no two points share the same x coordinates or the y coordinates, no three points are in the same line.

Intuition: Gift wrapping algorithms.

#recordings("Simple Gift Wrapping Algorithms")[
  - Select the initial point
    - Find the point which has the smallest $x$ coordinates or the smallest $y$ coordinates.
  - 找到旋转角度最大的点，作为凸包上的点选入
    - 这也可以看做是一种橡皮筋手搓生成凸包的过程
  - Time Complexity: $O(n dot h)$
]

=== Divide and Conquer for Convex Hall

#recordings("When to use Divide and Conquer")[
  - 分治最关键的是两个步骤：
    - 分解成若干个子问题（递）
    - 把子问题的结果合并起来（归）
  - 如果使用分治法，那务必重视的一点是递归的终点（最简单的情况）必须是简单可解的。($O(1)$ Time Complexity)
]

For simple condition: when $n <= 3$, the convex hall is quite simple! All the points are the vertices of the convex hall.

Now, we need to solve two things:

- When to divide
  - With x coordinates
  - More like half splitting!

- When to conquer
  - The most critical step!
  - We need to finding the bridges (Upper Bridge and Lower Bridge) to form the bigger convex hall.
    - *Two Finger Algorithms*

#recordings("Two Finger Algorithms")[
  - 基本思路类似于双指针法实现线性扫描
  - 基本思想还是不断旋转找到最外部的切线
]

#figure(
  image("images/convexhall.png"),
  caption: [Convex Hall Conquering Steps],
)

Time Complexity:

$
  T(n) = 2 T(n/2) + Theta(n)
$

Thus total time complexity: $Theta(n log n)$

For the compute of this time complexity, we can use Master Theorem.

== Master Theorem

For simple cases:

$
  T(n) = a T(n/b)
$

To compute this complexity, we use the recursive tree to solve this:

$
  T(n) = a^k T(n / b^k)
$

For the recursion endpoint, $n/b^k = 1$, we can compute:

$
  T(n) = a^(log_b n) T(1) = n^(log_b a) T(1) = O(n^(log_b a))
$

For general cases:

$
  T(n) = a T(n/b) + f(n)
$

We need to compare $f(n)$ and $n^(log_b a)$.

=== The Work at the Leaves Dominates

$
  f(n) = O(n^(log_b (a-epsilon)))
$

Then it means that recursion part dominates! ($T(n) = a T(n/b)$). Thus, the time complexity is:

$
  T(n) = a T(n/b) + f(n) = a T(n/b) + O(n^(log_b (a-epsilon))) = Theta(n^(log_b a))
$

=== The Work is Balanced

$
  f(n) = Theta(n^(log_b a) dot log^(k)n)
$

Then it means the two parts are both the dominant parts! The total time complexity remains the same.

$
  T(n) = a T(n/b) + f(n) = a T(n/b) + Theta(n^(log_b a) dot log^(k)n) = Theta(n^(log_b a) dot log^(k+1) n)
$

=== The Work at the Root Dominates


$
  f(n) = Omega(n^(log_b a + epsilon))
$

and:

$
  exists c in R, exists N_0 in NN, forall n > N_0: a f(n/b) <= c f(n)
$

#recordings("Regular Condition")[
  - 这个条件说明划归到子问题的时候时间复杂度可能很大，但是对于大问题“分而治之”的复杂度是非常昂贵的。
  - 总复杂度有递归的最高层（根节点）的代价决定，这也保证该情况下时间复杂度的量级为 $Theta(f(n))$
]

Then the total time complexity:

$
  T(n) = Theta(f(n))
$

#example("Example for the work at the root dominates")[
  例如如果递归的时间复杂度为：
  $
    T(n) = 3 T(n/4) + n^2
  $

  $a=3$,$b=4$, $T(n) = Theta(n^2)$
]

== Median Finding

#problem("Median Finding")[
  Given set of $n$ numbers, define $"rank"(x)$ as number of numbers in the set that are $≤ x$.
  Find element of rank $floor (n+1)/2 floor.r$ (lower median) and $ceil (n+1)/2 ceil.r$ (upper median).
]

Obviously, we can use *sorting algorithms* to solve this! The time complexity is $Theta(n log n)$.

Simple Algorithms: Define problem as `Select(S,i)` to find the `i` th element value in the set S.

- Pick $x in S$
  - We just pick it cleverly
- Compute $k = "rank"(x)$
- $B = {y in S|y < x}$
- $C = {y in S|y > x}$
- algorithms:
  - If $k = i$: `return x`
  - If $k < i$: `return Select(C,i-k)`
  - If $k > i$: `return Select(B,i)`

For dummy choices for selecting $x in S$, for the worse case, the time complexity is $Theta(n^2)$.

=== Picking $x$ cleverly

- Arrange S into columns of size 5 ($ceil n/5 ceil.r$ cols).
- Sort each columns in linear time.
- Find *medians of medians* as the selected $x$.

#recordings("Why selecting this?")[
  - 对于简单的取常数或者中间值的方法在极端情况下会退化到平方时间复杂度，因为我们难以知道全局数据的分布特征，因此我们很难选择一个好的 splitting
  - 和快速排序很类似！我们希望选择一个好的 splitting，这样让递归算法变成对数级别的。
  - 而下面的选择可以保证 splitting 的效率，即至少有 $3 ( ceil n/10 ceil.r -2 )$ 的点被分到左边并且至少有 $3 ( ceil n/10 ceil.r -2 )$ 的点被分到右边。
]


#figure(
  image("images/median.png"),
  caption: [SELECT for medians of medians],
)


Recurrence:

$
  T(n) = T(ceil n/5 ceil.r) + T((7 n)/10 + 6) + Theta(n)
$

- $T(ceil n/5 ceil.r)$ 是找到中位数的中位数的算法时间
- $Theta(n)$ 是分组线性扫描需要的时间复杂度
- $(7 n)/10 + 6$ 代表子问题的规模，因为我们保证$3 ( ceil n/10 ceil.r -2 )$ 会被分到对应的组，因此最坏情况就是 $(7 n)/10 + 6$

Solving this recurrence.

#figure(
  image("images/induction-proof.jpg"),
  caption: [Induction proof for median finding algorithms],
)

== Matrix Multiplication

For simple matrix multiplication, the time complexity is $O(n^3)$.

$
  c_(i,j) = sum^p_(k=1) a_(i,k) b_(k,j)
$

- $n^3$ times multiplication.
- $n^3- n^2$ times addiction.

=== Strassen Algorithms

$
  A = mat(A_11, A_12; A_21, A_22), B = mat(B_11, B_12; B_21, B_22), C = mat(C_11, C_12; C_21, C_22)
$

For simple divide and conquer algorithms:

$
  C_11 = A_11 B_11 + A_12 B_21 \
  C_12 = A_11 B_12 + A_12 B_22 \
  C_21 = A_21 B_11 + A_22 B_21 \
  C_22 = A_21 B_12 + A_22 B_22
$

这个是基本的分治算法，对于子矩阵，需要进行 8 次 子矩阵的乘法和 4 次子矩阵的加法。

$
  T(n) = 8 T(n/2) + Theta(n^2)
$

Based on Master theorem, the time complexity is $O(n^3)$, remains unchanged!

The break through for strassen algorithms are reducing matrix multiplication from 8 times into 7 times by reducing repeated computation!

$
  M_1 = (A_11 + B_22)(B_11 + B_22)\
  M_2 = (A_21 + A_22)B_11\
  M_3 = A_11 (B_12 - B_21)\
  M_4 = A_22 (B_21 - B_11)\
  M_5 = (A_11 + A_12) B_22\
  M_6 = (A_21 - A_11)(B_11 + B_12)\
  M_7 = (A_12 - A_22)(B_21 + B_22)
$

7 times matrix multiplication ($n/2 times n/2$), and 18 times addiction.

$
  C_11 = M_1 + M_4 - M_5 + M_7\
  C_12 = M_3 + M_5\
  C_21 = M_2 + M_4\
  C_22 = M_1 - M_2 + M_3 + M_6
$

Thus the time complexity:

$
  T(n) = 7 T(n/2) + Theta(n^2) = Theta(n^(log_2 7)) approx Theta(n^2.807)
$

== FFT





= Conclusion
