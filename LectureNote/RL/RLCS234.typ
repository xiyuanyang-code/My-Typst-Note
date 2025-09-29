#import "@preview/dvdtyp:1.0.1": *
#show: dvdtyp.with(
  title: "Reinforcement Learning CS234",
  author: "Xiyuan Yang",
  abstract: [Lecture notes for reinforcement learning],
)
#show link: set text(fill: blue, weight: 700)
#show link: underline
#show ref: set text(fill: blue, weight: 500)


// comment for hiding outlines
#outline()

// uncomment for changing to a new page
// #pagebreak()

= Introduction

= Tabular MDP Planning

== Markov Process

#definition("Markov Process")[
  $ P(X_(t_(n+1)) = x_(n+1) | X_(t_n) = x_n, dots, X_(t_1) = x_1) = P(X_(t_(n+1)) = x_(n+1) | X_(t_n) = x_n) $
]

We define $G_t$ with discount factor.

Definition of State value Function $V(s)$ as a MDP.


= Conclusion


