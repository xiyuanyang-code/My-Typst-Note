#import "@preview/dvdtyp:1.0.1": *
#import "@preview/lovelace:0.3.0": *
// for writing pseudocode
#show: dvdtyp.with(
  title: "Algorithms",
  author: "Xiyuan Yang",
  abstract: [SJTU Semester 2.1 Algorithms: Design and Analysis],
)
#show link: set text(fill: blue, weight: 700)
#show link: underline
#show ref: set text(fill: blue, weight: 500)


// comment for hiding outlines
#outline()

// uncomment for changing to a new page
// #pagebreak()

= Lec1 Introduction

== Problems and Computation

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

== Algorithm

=== Definition

#definition("Algorithm")[
  - Fixed length code
  - accept input with any length (or we say it can scale up!)
  - at finite time terminate
]

- Natural Language
- pseudocode
- Written Codes

For example, birthday matching problems.

=== Pseudocode

- if else end

- foreach end

- init data structure (Use ←)


#pseudocode-list[
  + do something
  + do something else
  + *while* still something to do
    + do even more
    + *if* not done yet *then*
      + wait a bit
      + resume working
    + *else*
      + go home
    + *end*
  + *end*
]

#pseudocode-list[
  + function BinarySearch(A, x)
  + low ← 0
  + high ← A.length - 1
  + while low <= high
    + mid ← low + floor((high - low) / 2)
    + if A[mid] == x then
      + return mid
    + else if A[mid] < x then
      + low ← mid + 1
    + else
      + high ← mid - 1
    + end
  + end
  + return -1
  + end
]


== Detailed Contents

=== Greedy algorithms

- making the *locally optimal choice* at each stage with the hope of finding a global optimum.

=== Divide and Conquer

- Like the merge sort!
- Split bigger problems into smaller ones.

=== Dynamic Programming

- 最优子结构 optimal sub-structure
  - 这也是和 divide and conquer 算法之间最显著的区别
- 重叠的子问题 overlapping sub-problems
  - 这保证了动态规划的重复利用的部分，也是动态规划的高效性所在（不再重复计算）

=== Back Tracking

- a brute-force searching algorithms with pruning.
- Like the DFS algorithm
  - N Queens Problems

=== Heuristic Algorithms

- when encountering large solve space
- optimize (or tradeoff) for traditional searching algorithms.
- great for NP-hard problems.


= Conclusion
