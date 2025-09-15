#import "@preview/dvdtyp:1.0.1": *
#show: dvdtyp.with(
  title: "AI1811 Machine Learning",
  author: "Xiyuan Yang",
  abstract: [Machine Learning Courses for AI1811, simple machine learning.],
)
#show link: set text(fill: blue, weight: 700)
#show link: underline
#show ref: set text(fill: blue, weight: 500)


// comment for hiding outlines
#outline()

// uncomment for changing to a new page
// #pagebreak()

= Introduction to machine learning

- Supervised Learning
  - Regression
    - KNN-based regression
    - linear regression
    - least squares
    - gradient descent
  - Classification
    - Decision Trees
    - Logistic regression
    - Support Vector Machine
- Dimensional Reduction
  - PCA
  - Locally Linear Embeddings
- Clustering
  - K-means
  - Expectation-maximization
  - K-means++
- Model Selection and evaluation
  - Overfitting
  - L1/L2 regularization
  - K-fold cross-validation
- MLE / MAP
  - Likelihood
    - Given the output, predict the likelihood of the parameters.
    - $P(x|theta)$: probability
    - $L(theta|x)$: Likelihood
    - $P(x|theta) = L(theta|x)$
    - generated from the probability distribution: $f(x, theta)$
  - Maximum likelihood estimation (MLE)
    - 观测问题，通过统计学来反推结果。
  $ hat(theta) = text("argmax")_theta space L(theta|x) $
  - Maximum a posteriori (MAP)
    - Add a prior
  $ hat(theta) = text("argmax")_theta space L(theta|x) P(theta) $


= Conclusion