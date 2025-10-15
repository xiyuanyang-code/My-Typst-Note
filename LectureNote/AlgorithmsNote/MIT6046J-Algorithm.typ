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
#pagebreak()

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

=== Solving Master Theorem Using Recursive Tree

#figure(
  image("images/recursive_tree.png"),
)

- 每一个叶子结点代表被递归分解的小问题的规模
- 每一个非叶子节点的时间复杂度为 $c f(n')$ (n' 代表当前层数 k的规模 $n/(b^(k-1))$)
- 最终总的时间复杂度就是这个递归树的所有节点的时间复杂度的总和
- 因此，只需要把这个递归树展开到最底层，并计算所有结点的代价，就是最终的时间复杂度。

使用等比数列求和：

$
  T(n) = Theta(n^(log_b a)) + sum^(log_b (n)-1)_(j=0) a^j f(n/(b^j))
$

这就是为什么主定理需要比较 f 和这个的大小，在这里仅仅做主要阐述：特别考虑 boundary $f(n) = Theta(n^(log_b a))$

$
  T(n) = Theta(n^(log_b a)) + sum^(log_b n-1)_(j=0) a^j (n/(b^j))^(log_b a) = log_b n times n^(log_b a) = Theta(n^(log_b a) log n)
$

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
  image("images/median.png", width: 10cm),
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

#example("Medians of Medians")[
  给定一个数组和 $k$ 值，尝试求解这个数组中的第 $k$ 大的元素。要求保证时间复杂度为 $O(n)$.
]

```python
from typing import List

class Solution:
    def find_median_of_small_array(self, arr: List[int]) -> int:
        # O(1) for constant length arrays
        arr.sort()
        return arr[len(arr) // 2]

    def select_pivot(self, arr: List[int]) -> int:
        n = len(arr)
        if n <= 5:
            return self.find_median_of_small_array(arr)

        # splitting into sub-lists
        sublists = [arr[i : i + 5] for i in range(0, n, 5)]
        medians = [self.find_median_of_small_array(sublist) for sublist in sublists]

        # ! very important recursion!
        # That is the T(n/5) part
        return self.findKthSmallest(medians, len(medians) // 2 + 1)

    def findKthSmallest(self, arr: List[int], k: int) -> int:
        n = len(arr)
        if n == 1:
            return arr[0]

        pivot = self.select_pivot(arr)

        # do partition based on selected pivot
        less = [x for x in arr if x < pivot]
        equal = [x for x in arr if x == pivot]
        greater = [x for x in arr if x > pivot]

        len_less = len(less)
        len_equal = len(equal)

        # the core recursion part remains unchanged
        if k <= len_less:
            return self.findKthSmallest(less, k)
        elif k <= len_less + len_equal:
            return pivot
        else:
            new_k = k - len_less - len_equal
            return self.findKthSmallest(greater, new_k)

    def findKthLargest(self, nums: List[int], k: int) -> int:
        n = len(nums)
        k_smallest = n - k + 1
        return self.findKthSmallest(nums, k_smallest)
```

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

=== Polynomials

All about polynomial:

$
  A(x) = sum_(k=1)^(n-1) a_k x^k = [a_0, a_k, dots, a_(n-1)]
$


=== Operations for $A(x)$


==== Evaluation

#definition("Evaluation")[
  Given $x$, calculate $A(x)$.
]


#theorem("Horner's Rule")[
  我们希望更少次数的乘法和加法
  $
    A(x) = a_0 + x(a_1 + x(a_2 + dots + x(a_(n-1))))
  $

  - Before: $Theta(n^2)$ times multiplication and $Theta(n)$ times addition.
  - After: $Theta(n)$ times multiplication and $Theta(n)$ tines addition.
  - Thus the time complexity: $O(n^2) arrow O(n)$
]

==== Addition

#definition("Addition")[
  $
    C(x) = A(x) + B(x)
  $

  Obviously, the time complexity is $O(n)$ for addition.
]

==== Multiplication

#definition("Multiplication")[
  $
    C(x) = A(x) times B(x), forall x in X
  $
]

- Naive calculation: $O(n^2)$

$
  c_k = sum_(j=0)^K a_j b_(K-j)
$


We want to achieve $O(n log n)$

#figure(image("images/poli.png", width: 10cm))


=== Representations

- Coefficient Vectors
- Roots and a scale term
- Samples

=== Vendermonde Matrix


$
  V dot A = mat(1, x_0, x_0^2, dots, x_0^(n-1); 1, x_1, x_1^2, dots, x_1^(n-1); 1, x_2, x_2^2, dots, x_2^(n-1); dots.v, dots.v, dots.v, dots.down, dots.v; 1, x_(n-1), x_(n-1)^2, dots, x_(n-1)^(n-1)) vec(a_0, a_1, a_2, dots, a_(n-1)) = vec(y_0, y_1, y_2, dots, y_(n-1))
$

#recordings("多项式插值")[
  - 这个本质上也可以看做是一种多项式插值的手段
  - 我们希望从 sample 的形式转变为 coefficient 的形式
  - 根据线性代数的知识，范德蒙行列式只有在 sample 的点均不相同的情况下才是不可逆的，因此只要 sample 了 n 个不相同的点，就可以保证能够求解可逆矩阵，但是可逆矩阵的时间复杂度是 $O(n^3)$，因此实际插值并不会采用这个原始的算法。

  $
    Pi_(0 <= i < j <= n-1) (x_j - x_i) = 0
  $
]

We want to calculate A, thus we need to calculate:

$
  A = V^(-1) Y
$

=== Divide and Conquer for FFT

The original input: $A_"eff"$ and $B_"eff"$ as two vectors, and we need to calculate $C_"eff"$ for the new coefficients after polynomial multiplications.

We know, if we have $N$ samples for two polynomials, just calculate $A(x_k) dot B(x_k)$.

因此，如果我们需要解决多项式乘法的问题，实际上我们可以将问题拆分为：

- 预处理计算 $"FFT"(A)$ and $"FFT"(B)$: we want it to be $O(N log N)$
- $C_"point" = A_"point" dot.circle B_"point"$: it is $O(N)$, not the bottleneck
- $"IFFT"(C_"point")$: we want it to be $O(N log N)$

==== How to divide?

Divide into Even and Odd Coefficients $O(n)$

$
  A_"even" = sum_(k=0)^(ceil n/2 -1 ceil.r) a_(2k) x^(k) = [a_0, a_2, dots, a_(2l)]
$
$
  A_"odd" = sum_(k=0)^(ceil n/2 -1 ceil.r) a_(2k+1) x^(k) = [a_1, a_3, dots, a_(2l)]
$

==== How to conquer

$
  A(x) = A_"even" (x^2) + x dot A_"odd" (x^2) "for" x in X
$

Thus, we need to recursively calculate $A_"even"(y), y in X^2 = {x^2 |x in X}$

$
  T(n, |X|) = 2 T(n/2, |X|) + O(n + |X|) = O(n^2)
$

Actually, the time complexity does not change... However, if we can achieve the conquer time complexity as follows, we can do a great improvement:

$
  T(n, |X|) & = 2 T(n/2, (|X|)/2) + O(n + |X|) \
       T(n) & = 2 T(n/2) + O(n)
$

Thus, the time complexity is $O(n log n)$! We should select $X$ in a clever way in which it is *collapsing*, or we can say:

$
  |X^2| = (|X|)/2
$

for the base case, $X = {1}$ when $|X| = 1$.

#recordings("Roots of Unity")[
  - 从数值来讲，就是在复平面内不断求根
  - 在复平面内就是不断的取中间的过程
  $
    (cos theta, sin theta) & = cos theta + i sin theta = e^(i theta) \
                     theta & = 0, 1/n tau, 2/n tau, dots, (n-1)/n tau (tau = 2 pi)
  $
]

Then, we successfully implement FFT with the time complexity for $O(n log n)$!

- Well defined recursion
- Selected $X$, for $x_k = e^((i tau k)/n)$

=== Inverse Discrete Fourier Transform (IFFT)

We want to return the coefficients from the multiplied samples. The transformation of this form is $A = V^(-1) Y$. Thus, all we need to do is calculate $V^(-1)$, or the inverse of Vendermonde Matrix!

#theorem("Calculate the inverse")[
  $
    V^(-1) = 1/n overline(V)
  $
  where $overline(V)$ is the complex conjugate of $V$.
]

#figure(
  image("images/ifft.png", width: 13.3cm),
)

#recordings("Inverse")[
  - 换句话说，范德蒙行列式是可以快速求解的，因为旋转基本根的良好性质。
]

== van Emde Boas Tree

#definition("vEB Tree")[
  Goal: We want to maintain $n$ elements of a *set* in the range ${0,1,2,dots,u-1}$ and perform *Insert*, *Delete* and *Successor* Operations in $O(log log u)$ time.
  - Successor: 后继操作，即找到严格大于 $x$ 的最小元素。
]

By using an ordered binary search tree like AVL Tree, we can implement this query in $O(log n)$ time on average.

For example,  this data structure can be used in Network Routing Tables, where u = Range of IP Addresses → port to send. (u = $2^(32)$ in IPv4)

=== Intuition

Where might the $O(log log u)$ bound arise?

Binary search over $O(log u)$ elements.

$
      T(k) & = T(k/2) + O(1) = Theta(log n) \
  T(log u) & = T((log u)/2) + O(1) \
      T(u) & = T(sqrt(u)) + O(1)
$

Thus, we want to find the recurrence like:

$
  T(u) = T(sqrt(u)) + O(1)
$

=== Improvement

==== Bit Vector (Hash and Bucket)

Define a vector of size $u$, where $V[x] = 1 "iff" x in S$.

- Insert & Delete: $O(1)$
- Successor/Predecessor: Need to traverse all the bit vector, requires worst $O(n)$.

==== Split Universe into Clusters

#recordings("It is Chunking!")[
  - 经典的分块思想
  - 牺牲空间换取查询的更快时间复杂度
  - 在这个具体的问题场景下，选择分块可以避免一些重复计算，先对整体的块进行操作而加速。
]

#figure(
  image("images/cluster.png"),
)

- Insert: Set `V.cluster[i][j]` to 1, then mark cluster `high(x)` as no empty. $O(1)$
- Successor: $O(sqrt(u))$
  - Look within the cluster `i`
  - Else, find next non-empty cluster `i`
  - Find Minimum Entry `j` for that cluster
  - Return `index(i,j)`


==== Recurse

- `V.cluster[i]` is a size-$sqrt(u)$ van Emde Boas structure ($∀ 0 ≤ i < u$)
- `V.summary` is a size-$sqrt(u)$ van Emde Boas structure
- `V.summary[i]` indicates whether `V.cluster[i]` is nonempty

===== Insert

`Insert(V,x)`:
- `Insert(V.cluster[high(x)], low[x])`
- Remark that this cluster is non-empty: `Insert(V.summary, high[x])`.

Time complexity: $T(u) - 2 T(sqrt(u)) + O(1)$, thus the time complexity is $O(log u)$.

===== Successor

`SUCCESSOR(V, x)`:
- i = high(x)
- j = Successor(V.cluster[i], j)
  - if j == ∞
    - i = Successor(V.summary, i)
    - j = Successor(V.cluster[i], −∞)
- return index(i, j)

Time complexity for the worse case:

$
  T(u) = 3 T(sqrt(u)) + O(1)
$

$
  T(u) = O((log u)^(log_2 3)) approx O((log u)^1.585)
$

#recordings("Time complexity is not enough!")[
  - 分块的思想在 Bit Vector 的基础之上空间换时间达到了 log 级别的时间复杂度优化，但是仍然不够！
  - 如何进一步优化，根据主定理，我们优化的关键在于递归的分路数 $a$，缩减 $a$ 就有可能进一步缩减时间复杂度！
]

==== Maintain Min and Max

We store the minimum and maximum entry in each structure. This gives an $O(1)$ time overhead for each Insert operation.

`SUCCESSOR(V, x)`:
- if x < V.min return V.min
- i = high(x)
- if low(x) < V.cluster[i].max:
  j = Successor(V.cluster[i], low(x))
- else:
  - i = Successor(V.summary, i) $T(sqrt(n))$
  - j = V.cluster[i].min $O(1)$
- return index(i, j)

Now the time complexity is:

$
  T(n) = T(sqrt(n)) + O(1) = O(log log u)
$

==== Don't Store min recursively

#figure(
  image("images/insert_veb.png"),
  caption: [Divide and Conquer Algorithms for veb Tree insertion.],
)

#recordings("Pay Attention to min and max")[
  - vEB 树最关键的地方在于每一个 vEB 树的最大值和最小值的管理是不在分块数组内部的，而是作为当前树的缓存额外存储。
  - 在插入过程中，如果最小值被替换，原先的最小值需要被插入，因此可以等效的交换两者
  - 如果最大值被替换，因为最大值被正常的插入，因此只需要正常的更新就可以了。
  - 注意，一棵树的最小值的节点是不存储在 Bit Vector 里面的！这样是为了保证对于空树的递归调用实现不要太多次。
]

==== Deletions Operations

#recordings("Deletions")[
  - 如果需要删除最小值
    - vEB 树的精妙之处，额外缓存数据结构的最小值，因此这样就可以找到第二小的元素！
      - i = V.summary.min
      - second_min = index(i, V.cluster[i].min)
    - 接着，删除这个第二小的元素，因为他不可以在 Vector Bit 中出现，然后把这个元素的值更新到最小缓存中。
  - 如果删除中间值或者最大值：
    - 直接递归调用子簇 `Delete(T.cluster[i], low(x))`
  - 如果删除后的子簇是空的：
    - 直接删掉 Summary 对应的部分 (Second Call)
  - 如果删除最大值
    - 因为 max 是正常被储存的，因此我们需要更新 max
    - 找到新的 max：
      - 如果 Summary 是空的，那么新的 T.max 就是 T.min
      - 如果不是，就更新为最大簇的最大元素
]

#figure(
  image("images/veb.png"),
)

= Amortization

#definition("Amortization")[
  - 摊还分析 (Amortized Analysis) 的基本思想是：分析一系列操作的总成本，并将这个总成本平摊（平均）到每一个操作上。
  - 它关注的不是单个操作的“最坏情况”耗时，而是保证即使某个操作（例如每隔一段时间发生的重置/扩容操作）成本非常高，在考虑了所有操作的成本后，平均到每个操作上的成本仍然很低。
]

= Randomization and Randomized Algorithms

= Advanced Dynamic Programming

= Greedy Algorithms

= Graph Algorithms

= Linear Programming

= Complexity

= More Advanced Algorithms

#recordings("More advanced algorithms")[
  You know, algorithms are fascinating...
]

= Conclusion
