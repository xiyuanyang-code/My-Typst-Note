#import "@preview/dvdtyp:1.0.1": *
#show: dvdtyp.with(
  title: "AI1811-Deep Learning",
  author: "Xiyuan Yang",
  abstract: [Lecture notes for deep learning AI-1811.],
)
#show link: set text(fill: blue, weight: 700)
#show link: underline
#show ref: set text(fill: blue, weight: 500)


// comment for hiding outlines
#outline()

// uncomment for changing to a new page
// #pagebreak()

= Introduction

== Lecture Overview

- Introduction
  - High dimensional data
  - 高维数据中的维数灾难
- Model Architecture
  - MLP
  - CNN
  - RNN
  - Transformer
- Advanced about deep learning
  - LLM reasoning
  - 凝聚现象


== NN and Polynomial

深度学习有万有逼近定理，多项式拟合也有 Weierstrass 逼近定理。

#recordings("The bitter lesson")[
  The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin.
]

在实际情况下，我们发现神经网络的过拟合现象远小于多项式拟合。




= Conclusion
