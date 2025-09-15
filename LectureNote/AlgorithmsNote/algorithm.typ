#import "@preview/dvdtyp:1.0.1": *
#show: dvdtyp.with(
  title: "Algorithms",
  author: "Xiyuan Yang",
  abstract: [SJTU Semester 2.1 Algorithms],
)
#show link: set text(fill: blue, weight: 700)
#show link: underline
#show ref: set text(fill: blue, weight: 500)


// comment for hiding outlines
#outline()

// uncomment for changing to a new page
// #pagebreak()

= Lec1 Introduction

== Basic Introduction for algorithms

The basis of *AI*:
- Search
- Learning

#definition("计算问题")[
  - Given a *input*, $I$, $x in I$
  - output, $O$, $y in O$
  - relation: $f: x arrow y$: use the algorithm!
    - we have some boundaries: (s.t.)
]

#definition("Problem Domain")[
  The set of all the problems.
  $<I,O,f>$
]

#definition("Problem Instance")[
  one simple case in the problem domain
  $<I,O,f,x>$
]

What is algorithm:
- a piece of code
- handling the mapping from x to y

== algorithm

#definition("Algorithm")[
  - 代码长度固定?
  - 接受任何长度输入
  - at finite time terminate
]

- Natural Language
- 伪代码
- Written Codes

For example, birthday matching problems.

Definition:
- Input: a list of students
- Output: (name1, name2) s.t. have the same birthday

=== 伪代码

- if else end

- foreach end

- init data structure

=== Dynamic Programming

- 最优子结构 optimal sub-structure
  - 这也是和 divide and conquer 算法之间最显著的区别
- 重叠的子问题 overlapping sub-problems
  - 这保证了动态规划的重复利用的部分，也是动态规划的高效性所在（不再重复计算）

=== Back Tracking

回溯法
启发式算法

=== All the algorithms to be learned




= Conclusion