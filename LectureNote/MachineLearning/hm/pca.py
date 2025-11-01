import numpy as np
import matplotlib.pyplot as plt

# --- 1. DEFINE DATASET ---
c1 = np.array([[2, 1], [2, 2], [2, 3]])
c2 = np.array([[4, 3], [5, 4], [6, 4]])
X = np.concatenate((c1, c2), axis=0)
N = X.shape[0]  # N = 6 observations
D = X.shape[1]  # D = 2 dimensions

# --- 2. GLOBAL PCA CALCULATION ---

# Calculate the Global Centroid (x_bar)
x_bar = np.mean(X, axis=0)

# Center the data
X_centered = X - x_bar

# Calculate the Global Covariance Matrix (C)
# rowvar=False means columns are variables (x1, x2)
C = np.cov(X, rowvar=False)

# Find Eigenvalues and Eigenvectors
W, V = np.linalg.eig(C)

# Determine the Principal Component (w)
max_idx = np.argmax(W)
lambda1 = W[max_idx]
w = V[:, max_idx]

# Ensure w points in the positive x1 direction for consistent visualization
if w[0] < 0:
    w = -w

# Calculate Offset (w_0)
w0 = -np.dot(w, x_bar)

# --- RESULTS FOR PART (a) ---
print("--- Part (a) Results: Principal Direction Equation ---")
print(f"Global Centroid (x_bar): [{x_bar[0]:.4f}, {x_bar[1]:.4f}]")
print(f"Principal Direction Vector (w): [{w[0]:.4f}, {w[1]:.4f}]")
print(f"Offset Parameter (w0): {w0:.4f}")
print(f"Equation: {w[0]:.4f} * x1 + {w[1]:.4f} * x2 + {w0:.4f} = 0\n")

# --- 3. VISUALIZATION FOR PART (a) ---

w1, w2 = w[0], w[1]
x_bar_1, x_bar_2 = x_bar[0], x_bar[1]

plt.figure(figsize=(10, 8))

# Plot the Data Points
plt.scatter(c1[:, 0], c1[:, 1], color='blue', label='$c_1$', marker='o', alpha=0.7)
plt.scatter(c2[:, 0], c2[:, 1], color='red', label='$c_2$', marker='x', alpha=0.7)

# Plot the Centroid
plt.plot(x_bar_1, x_bar_2, 'kx', markersize=10, label='Centroid $\\mathbf{\\bar{x}}$')

# Plot the Principal Direction Line (The equation w^T * x + w_0 = 0)
x1_line = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
x2_line = (-w1 * x1_line - w0) / w2
plt.plot(x1_line, x2_line, 'k--', label='Principal Direction Line')

# Plot the Direction Vector w originating from the centroid
vector_scale = 1.5
plt.quiver(x_bar_1, x_bar_2, w1, w2,
           angles='xy', scale_units='xy', scale=1/vector_scale,
           color='green', linewidths=2,
           label='Direction Vector $\\mathbf{w}$ (Scaled)', zorder=3)

# --- 4. PCA TRANSFORMATION AND RECONSTRUCTION (b) & MSE (c) ---

# (b) Transform observations (Z)
# The transformation matrix is just the principal eigenvector 'w'
Z = np.dot(X_centered, w) # Z is 6x1 (1D projected coordinates)

# (b) Reconstruct coordinates (X_reconstructed)
# X_reconstructed = Z * w^T + x_bar
X_reconstructed = np.outer(Z, w) + x_bar

# (c) Calculate the Mean Squared Error (MSE)
# MSE is the aggregate mean squared error (discrepancy) between X and X_reconstructed
mse = np.mean(np.sum((X - X_reconstructed)**2, axis=1))

# --- RESULTS FOR PART (b) and (c) ---
print("--- Part (b) and (c) Results: Transformation and MSE ---")
print(f"1D Projected Coordinates (Z):\n{Z.round(4)}")
print(f"Reconstructed Coordinates (X_reconstructed):\n{X_reconstructed.round(4)}")
print(f"Aggregate Mean Squared Error (MSE): {mse:.4f}\n")

# --- VISUALIZATION FOR PART (b) ---
# Plot reconstructed points and lines connecting original to reconstructed
for i in range(N):
    color = 'blue' if i < c1.shape[0] else 'red'
    # Plot reconstructed point
    plt.plot(X_reconstructed[i, 0], X_reconstructed[i, 1], marker='s', markersize=5, color=color, alpha=0.8)
    # Plot projection line
    plt.plot([X[i, 0], X_reconstructed[i, 0]], [X[i, 1], X_reconstructed[i, 1]],
             linestyle=':', color='gray', alpha=0.5)

# Label reconstructed points
plt.scatter([], [], marker='s', color='gray', label='Reconstructed Points') # for legend

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Global PCA: Principal Direction, Projection, and Reconstruction (a, b)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()


# --- 5. FISHER DISCRIMINANT RATIO (d) ---

# Separate the 1D projected coordinates (Z) back into two classes
Z_c1 = Z[:c1.shape[0]]
Z_c2 = Z[c1.shape[0]:]

# Centroid of projected class 1 (m1)
m1 = np.mean(Z_c1)
# Centroid of projected class 2 (m2)
m2 = np.mean(Z_c2)

# Variance of projected class 1 (sigma1^2)
sigma1_sq = np.var(Z_c1, ddof=1) # ddof=1 for sample variance
# Variance of projected class 2 (sigma2^2)
sigma2_sq = np.var(Z_c2, ddof=1)

# Compute Fisher Discriminant Ratio (FR)
# FR = (m1 - m2)^2 / (sigma1^2 + sigma2^2)
fr = (m1 - m2)**2 / (sigma1_sq + sigma2_sq)

# --- RESULTS FOR PART (d) ---
print("--- Part (d) Results: Fisher Discriminant Ratio (FR) ---")
print(f"Projected Centroid m1: {m1:.4f}")
print(f"Projected Centroid m2: {m2:.4f}")
print(f"Projected Variance sigma1^2: {sigma1_sq:.4f}")
print(f"Projected Variance sigma2^2: {sigma2_sq:.4f}")
print(f"Fisher Discriminant Ratio (FR): {fr:.4f}")