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

Given $n$ points in the plane, the goal is to find the smallest polygon containing all points in $S = {(x_i, y_i)|i = 1,2,dots,n}$



= Conclusion
