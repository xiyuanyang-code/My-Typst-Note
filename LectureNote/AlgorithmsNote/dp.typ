#import "@preview/dvdtyp:1.0.1": *
#show: dvdtyp.with(
  title: "Dynamic Programming",
  author: "Xiyuan Yang",
  abstract: [MIT 6.006 Introduction to algorithms, focusing on *dynamic programming*.],
)
#show link: set text(fill: blue, weight: 700)
#show link: underline
#show ref: set text(fill: blue, weight: 500)


// comment for hiding outlines
#outline()

// uncomment for changing to a new page
// #pagebreak()

= Introduction

MIT 6.006 Introduction to algorithms.

This section will focus on *Dynamic Programming*.

= Recursive Algorithms

== SRTBOT

+ Subproblem definition
+ Relate subproblem solutions recursively
+ Topological order on sub-problems ($\to$ subproblem \textbf{DAG}, for dependencies for all the sub problems.)
+ Base cases of relation
+ Original problem solution via subproblem(s)
+ Time analysis

#example("Merge Sort")[
  + Define sub problems as a function accepting parameters!
  + Sometimes, the huge problem itself is one kind of sub problems.
  + Key: Finding the related relation between different sub problems.
  + Base Case: The initial & final statement for recursion
]

*Memorization*: The computation of sub problems may in multiple times!
- e.g. Fibonacci Problems
- In simple sub-problem decomposition, $F(k)$ has been computed several times when solving $F(k+i), i > 0$.
- We need to use memorization to avoid repeated computation.

#recordings("Save memory and time")[
  - 空间换时间的算法思想，通过预处理把计算问题变成查询问题。
  - 同样，也可以节省算法的空间复杂度。
]

#recordings("Important for finding relations")[
  - 动态规划问题的关键在于找到合适的子结构问题
  - 从这个子结构出发，构建不同参数子规模的连接，可以从顶向下走递归，也可以自底向上走分治（本质还是递归的问题），中间可以做 mem 存储来节省时间复杂度。
  - 使用 DAG 建模复杂问题。
    - 找到最优复杂度的问题本质可以看做是找 DAG 中的最短路问题！
]


== Reusing Subproblem Solutions

- Draw subproblem dependencies as a *DAG*
- How to solve them?
  - *Top down*: record subproblem solutions in a memo and re-use(*recursion + memoization*)
  - Bottom up: solve subproblems in *topological sort order* (usually via loops)
- For Fibonacci, n + 1 subproblems (vertices) and < 2n dependencies (edges)
- Time to compute is then O(n) additions

A subtlety is that Fibonacci numbers grow to $Θ(n)$ bits long, potentially ≫ word size $w$. This means the number of bits needed to store the n-th Fibonacci number is proportional to n. When n is large, this number can be much bigger than the standard word size of your computer's CPU (e.g., 32 or 64 bits). Each addition costs $O(⌈n/w⌉)$ time, so total cost is:
$ O(n⌈n/w⌉)=O(n+n^2/w) $
time.

== DAG Simulations

Recall for DAG problems:

#problem("SSSP Problems for DAG")[
  Given a graph $G = (V, E)$ and a starting point $u in V$. We need to *compute $f(u,i), forall i in V$.* We define $f(i,i) = 0$.
]

For an edge $u arrow v$ with weight $w(u,v)$, we can find a shorter path using *relaxation*:

$ text("dist")[v] = min(text("dist")[v], text("dist")[u] + w(u,v)) $

对于 DAG 来说，因为其独特的无环拓扑结构，使用拓扑排序来确定唯一的处理顺序，然后每个节点按照拓扑排序的顺序处理，最终保证每个节点处理的使用前驱节点已经处理过了。

#recordings("DFS and DP")[
  - 把问题分解成子问题之后再使用 DP 解决，本质上就是带记忆的 DFS Search。
  - 递归函数本质上也可以看做是对一个 DAG 的 DFS 的过程。
  - 在 DAG 建模的问题中，可以保证最终的出度为0的点只能是问题最后的结果，而入度为0的点是已知的结果。（比如拆分下来最小的结果）
    - 递归的过程本质上可以建模成对依赖关系的反向图的 DFS Search 过程，从大问题开始不断 搜索知道遇到出度（原图入度）为 0 的点。
  - 说到底还是建模问题，要建模出结构良好的子结构问题。
]

== Example

#example("Bowling Example")[
  #figure(
    image("bowling.png"),
    caption: [Bowling Problems Demo]
  )
]

Sub problems design:

IF the input is a sequence:
- Prefixes `x[:i]`, $O(n)$
- suffixes `x[i:]`, $O(n)$
- substrings `x[i:j]`

=== Solution with suffixes

- Sub problem: $B(i) arrow text("[i:]")$
- We need to compute B(0)
- relate: 
$ B(i) = max{B(i + 1), v_i + B(i + 1), v_i · v_i+1 + B(i + 2)} $
- Classical Bottom DP from bottom-up!

== Conclusion for relate subproblem solution

- The general approach we're following to define a relation on subproblem solutions:

  - Identify a question about a subproblem solution that, if you knew the answer to, would reduce to "smaller" subproblem(s)

    - In case of bowling, the question is "how do we bowl the first couple of pins?"

  - Then locally brute-force the question by trying all possible answers, and taking the best

    - In case of bowling, we take the max because the problem asks to maximize

  - Alternatively, we can think of correctly guessing the answer to the question, and directly recurring; but then we actually check all possible guesses, and return the "best"

- The key for efficiency is for the question to have a small (polynomial) number of possible answers, so brute forcing is not too expensive

- Often (but not always) the nonrecursive work to compute the relation is equal to the number of answers we're trying

= Conclusion

DP is really a fantastic algorithms!