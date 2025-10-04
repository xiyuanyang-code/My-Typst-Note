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
  T(n) = a^(log_b n) T(1) = n ^ (log_b a) T(1) = O(n ^ (log_b a))
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

=== The Work is Balanced

=== The Work at the Root Dominates










= Conclusion
