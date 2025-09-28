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
    image("/LectureNote/AlgorithmsNote/images/bowling.png"),
    caption: [Bowling Problems Demo],
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

= Dynamic Programming Sub-Problems: LCS & LIS & Coins

#recordings("Why Sub problems?")[
  - Recursion and Reuse
  - When sub problems overlap
  - careful brute force. (Or clever brute force)
]

== Longest Common Subsequence (LCS)

#problem("Basic LCS")[
  Given two strings A and B, find a longest (*not necessarily contiguous*) subsequence of A that is also a subsequence of B.
]

Define sub problems for multiple inputs: using a matrix, which is the *product of multiple sub problem spaces*.

For example, in this case, we define two sub-problems which are the suffixes for string A and suffixes for string B, the final sub problem space is the doc product of two independent sub-problem spaces. Of course, we can define matrix with higher dimensions for more complex problems.

Thus, define:

$ L(i,j) = text("LCS")(A[i:], B[j:]), 0 <= i <= |A|, 0 <= j <= |B| $

For `A[|A|:]`, it means an empty string, it is easy to construct the initial statement.

Then, for the state transition equation:

We have:

$ max(L(i,j+1), L(i+1,j)) <= L(i,j) <= min(L(i,j+1), L(i+1,j)) + 1 $

if $A(i) == B(j)$:
- We can prove that `L(i,j) = L(i+1, j+1) + 1`
else:
- at least one of `A[i]` and `B[j]` are not in the LCS.
- Thus, transform this problem into smaller sub problems.
- $max(L(i,j+1), L(i+1,j))$

#recordings("Finding state transition equation")[
  - 问题的关键在于新加入的字母，这些字母往往会在状态转移方程中出现或者作为分支的判定条件
  - 要思考在什么状态下，该问题可以被修改成为更小的子问题进行运算
]

== Longest Increasing Subsequence (LIS)

#problem("LIS problems")[
  Given a string A, find a longest (not necessarily contiguous) subsequence of A that strictly
  increases (lexicographically)
]

Still using the suffixes:

$ L(i) = text("LIS")(A [i:]) $

- final statement, we need to solve `L(0)`
- initial statement, `L(length_A) = 0` for it is an empty string.
- transition:
  - if i is in the longest sequence:
    - `L(i) = L(i+1) + 1`
  - if not:
    - `L(i) = L(i+1)`

However, it is not easy to find the "if" statement, thus, we need to *change the definition of sub problems*

We define: $x(i)$ = length of longest increasing subsequence of suffix `A[i :]` that includes `A[i]`. Then, we solve it again:

- final statement: result is the maximum value of ${x(i)|i in {1,2,3,dots,n}}$

- initial statement: `x(length_A)` = 0

- transition:
  - if `s[i]` < `s[i+1]`:
    - `x(i) = x(i+1) + 1`
  - else:
    - it is still difficult
    - $x(i) = max{1 + x(j) | i < j < |A|, A[j] > A[i]} union {1}$
    - We need to traverse the processed string, which concat thr added `s[i]` into the current strings.

== Alternating Coin Game

#problem("Alternating coin games")[
  Given sequence of $n$ coins of value $v_0, v_1, dots, v_(n-1)$
  - Two players (“me” and “you”) take turns
  - In a turn, take first or last coin among remaining coins
  - My goal is to maximize total value of my taken coins, where I go first
]

The structure of the sub-problems are quite simple: we just need to define $L(i,j)$ as the optimal total value I can get for the substring of `[i..j]`($i <= j$)

- final statement: `L[0, n-1]`
- initial statement: `L[i,i]`

However, it is hard to write the transition!

Thus, we will change to another solution:

$x(i,j,p)$ = maximum total value I can take when player $p in {text("me"), text("you")}$ starts from coins of values $v_i dots v_j$.
- me is p = 0
- you is p = 1

#recordings("Adding a new dimension")[
  We can add a new dimension when solving complex dp tasks.
]

- final statement: $x(0, n-1, 0)$
- initial statement:
  - $x(i,i,0) = s[i]$
  - $x(i,i,1) = 0$
- Then, the transition is:
$ L(i,j,0) = max(L(i, j-1, 1) + a[j], L(i+1, j,1) + a[i]) $
$ L(i,j,1) = min(L(i, j-1, 0) + a[j], L(i+1, j,0) + a[i]) $
- We must assume the component is clever enough.

Thus, we finish this problem in $O(n^2)$ running time.

= SubProblems Constraints and Expansions

== Bellman-Ford Expansion SSSP

=== DAG Shortest Path

It is time when we solve the DAG problems using relaxation!

- Define sub problems:

$delta_k (s,v)$ means the weight of shortest path from $s$ to $v$ using at most $k$ edges.

- We want to solve:

$ delta_(|E|)(s,v) $

- What we have:

$ delta_0 (s,v) = 0 , forall v in V $

$ delta_i (s,s) = 0, forall i in {0,1,2,dots, |V|} $

- Status transform

$ delta(s, v) = min {delta(s, u) + w(u,v) | u in "Adj"^-(v) } union { delta_(k-1)(s,u)} $

#recordings("When to use DP?")[
  - 动态规划的状态转移方程经常出现 min max 等求最大最小值的组合
  - 这是因为动态规划经常子问题定义的是一个最优化问题
  - 而最优化问题的处理基本逻辑就是暴力枚举+取最值
    - 因此检查动态规划的正确性（尤其遇到最大最小值）可以看所有枚举情况是否被包含
    - 这也是为什么更复杂的动态规划需要分类讨论而不可以直接取最值，因为有时候并不是简单的枚举！
]


== APSP

For all pairs shortest path: *Floyd-Warshall*

- For simple SSSP:
  - $O(|V|^2 |E|) = O(|V|^4)$

$d(u, v, k)$ = minimum weight of a path from $u$ to $v$ that only uses vertices from ${1, 2, . . . , k} union {u, v}$

- What we want to solve: $d(u,v,|V|)$
- What we have:
  - $d(u,v,0) = w(u,v)$ if $(u,v) in E$
  - $d(u,v,0) = w(u,v)$, otherwise
- transformation:
  - If $k$ is in the shortest path: $d(u,v,k) = d(u,k,k-1) + d(k,v,k-1)$
  - If not: $d(u,v,k) = d(u,v,k-1)$
$ d(u,v,k) = min(d(u,v,k-1), d(k,v,k-1) + d(u,k,k-1)) $
- Time complexity: $O(|V|^3)$

== Arithmetic Parenthesization

#problem("Arithmetic Parenthesization")[
  - 给定一个数字序列，两个数字之间存在加号或者乘号
  - 你可以任意改变运算优先级（加括号）
  - 求解：最终最大的输出值
  - Allow negative numbers
]

Idea: find the operations from the root.
Sub-Problems: using substring!

Define sub problems:

$x(i,j "opt")$ = opt value with sub string from [i..j)

- $0 <= i,j <= n$ and $"opt" in {"min", "max"}$

- What we have:
  - $x(i,i,"opt") = 0$
  - $x(i,i+1,"opt") = a_i$

- Original Problems: $x(0,n,"max")$

$ x(i, j, "opt") = "opt" {x(i, k, "opt"') ∗ k x(k, j, "opt"'')) | i < k < j; "opt"' , "opt"'' in {min, max}} $



== Piano Fingering

#problem("Piano Fingering")[
  #figure(
    image("images/piano.png"),
    caption: [Piano Fingering],
  )
]

$x(i, f)$ = minimum total difficulty for playing notes $t_i$, $t_(i+1)$, . . . , tn−1 starting with finger f on note $t_i$.

Define: $x(i,f)$ is $min{x(i + 1, f') + d(t_i, f, t_(i+1), f') | 1 ≤ f' ≤ F }$


Time Complexity: $O(n times F^2)$

== Guitar Fingering


#problem("Guitar Fingering")[
  #figure(
    image("images/guitar.png"),
    caption: [Guitar Fingering],
  )
]

$F arrow F times S$

Time Complexity: $O(n F^2 S^2)$


=== Multiple Notes at Once

$t_i$ is a set of notes to play at time $i$.

$f_i$ is a mapping of placing notes with different fingers: $t_i arrow {1,2,3,dots,F}$

For each fingering $f_i$, at most $T^F$ choices, $T = max_i |t_i|$

Time Complexity: $O(n T^(2F))$


= Pseudopolynomial

== Rod Cutting

== SubSet Sum

= Conclusion

== Main Features of Dynamic Programs

DP is really a fantastic algorithms!
