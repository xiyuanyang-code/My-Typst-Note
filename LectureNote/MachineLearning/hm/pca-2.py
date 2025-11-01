import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_digits

# --- PCA Implementation Class ---

class CustomPCA:
    """
    Custom Principal Component Analysis (PCA) implementation.
    The primary method is eigendecomposition of the Covariance Matrix.
    """
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None  # The Principal Components (Eigenvectors)
        self.explained_variance_ = None  # The Eigenvalues

    def fit(self, X):
        """
        Fit the PCA model using the standard covariance approach.
        X: (n_samples, n_features)
        """
        start_time = time.time()
        
        # 1. Mean-Centering
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        C = np.cov(X_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(C)
        sorted_indices = np.argsort(eigvals)[::-1]
        self.explained_variance_ = eigvals[sorted_indices]
        # Sort eigenvectors (principal components)
        eigvecs = eigvecs[:, sorted_indices]
        
        # Select n_components
        if self.n_components is not None:
            self.components_ = eigvecs[:, :self.n_components]
        else:
            self.components_ = eigvecs
            self.n_components = self.components_.shape[1]

        end_time = time.time()
        return end_time - start_time

    def transform(self, X):
        """Project data X onto the principal components."""
        if self.mean_ is None or self.components_ is None:
            raise Exception("Model not fitted.")
        
        X_centered = X - self.mean_
        # Projection: X_centered @ self.components_
        return np.dot(X_centered, self.components_)

    def inverse_transform(self, X_reduced):
        """Reconstruct data from the principal components."""
        # Reconstruction: X_reduced @ components_T + mean_
        X_reconstructed_centered = np.dot(X_reduced, self.components_[:, :X_reduced.shape[1]].T)
        return X_reconstructed_centered + self.mean_

# --- Gram Matrix PCA Shortcut ---

def gram_matrix_pca_components(X, n_components=None):
    """
    Extracts principal components using the Gram Matrix shortcut (for N << D).
    X: (n_samples, n_features), assumed to be already mean-centered.
    """
    start_time = time.time()
    n_samples, n_features = X.shape

    K = np.dot(X, X.T) / (n_samples - 1)
    eigvals_K, V_K = np.linalg.eigh(K)

    # 3. Sort Eigenpairs
    sorted_indices = np.argsort(eigvals_K)[::-1]
    eigvals_K = eigvals_K[sorted_indices]
    V_K = V_K[:, sorted_indices]
    positive_eigvals_mask = eigvals_K > 1e-10
    eigvals_C = eigvals_K[positive_eigvals_mask] * (n_samples - 1) # Estimated Covariance Eigenvalues
    V_K_pos = V_K[:, positive_eigvals_mask]
    
    # Compute U (Principal Components)
    U = np.dot(X.T, V_K_pos) # U = X^T @ V_K_pos

    norms = np.linalg.norm(U, axis=0)
    non_zero_norms = norms > 1e-10
    U[:, non_zero_norms] = U[:, non_zero_norms] / norms[non_zero_norms]
    
    # Final selection
    if n_components is not None:
        U = U[:, :n_components]
        eigvals_C = eigvals_C[:n_components]
    else:
        n_components = U.shape[1]
        
    end_time = time.time()
    return U, eigvals_C, end_time - start_time


# --- Data Handling and Simulation ---

def load_and_preprocess_data(max_samples_per_class=1000):
    data = load_digits()
    X = data.data.astype(np.float64) # (n_samples, 64 features)
    y = data.target
    
    X_subsampled = []
    y_subsampled = []
    
    for digit in range(10):
        indices = np.where(y == digit)[0]
        selected_indices = indices[:max_samples_per_class]
        
        X_subsampled.append(X[selected_indices])
        y_subsampled.append(y[selected_indices])

    X_final = np.concatenate(X_subsampled, axis=0)
    y_final = np.concatenate(y_subsampled, axis=0)
    
    print(f"Dataset loaded and subsampled. Final shape: {X_final.shape}")
    return X_final, y_final, 8

# --- Main Analysis Function ---

def run_pca_analysis():
    # Load data (simulating memory constraint with load_digits data)
    X, y, img_dim = load_and_preprocess_data(max_samples_per_class=1000)
    n_samples, n_features = X.shape

    mean_vector = np.mean(X, axis=0)
    X_centered = X - mean_vector
    
    # --- Part (a): PCA Execution and Visualization ---
    print("\n" + "="*50)
    print("Part (a): PCA Comparison and Visualization")
    print("="*50)
    
    # --- Standard Covariance Approach ---
    pca_cov = CustomPCA()
    time_cov = pca_cov.fit(X)
    components_cov = pca_cov.components_
    
    print(f"Standard Covariance Approach Time: {time_cov:.4f} seconds.")
    
    # --- Gram Matrix Approach ---
    # The Gram matrix method yields up to N principal components (N=n_samples)
    components_gram, explained_variance_gram, time_gram = gram_matrix_pca_components(X_centered)
    
    print(f"Gram Matrix Approach Time: {time_gram:.4f} seconds.")

    # Visualize results
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    fig.suptitle("PCA Analysis of Digit Dataset", fontsize=14)

    # Average Digit-1 Template
    digit_1_indices = np.where(y == 1)[0]
    avg_digit_1 = np.mean(X[digit_1_indices], axis=0).reshape(img_dim, img_dim)
    axes[0, 0].imshow(avg_digit_1, cmap='gray')
    axes[0, 0].set_title("Avg. Digit 1")
    axes[0, 0].axis('off')
    
    # Covariance PC Visualization
    for i in range(5):
        pc_image = components_cov[:, i].reshape(img_dim, img_dim)
        axes[0, i+1].imshow(pc_image, cmap='gray')
        axes[0, i+1].set_title(f"Cov PC {i+1}")
        axes[0, i+1].axis('off')

    # Gram PC Visualization
    for i in range(5):
        pc_image = components_gram[:, i].reshape(img_dim, img_dim)
        axes[1, i+1].imshow(pc_image, cmap='gray')
        axes[1, i+1].set_title(f"Gram PC {i+1}")
        axes[1, i+1].axis('off')

    axes[1, 0].axis('off') # Empty slot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # --- Part (b): Efficiency Analysis ---
    print("\n" + "="*50)
    print("Part (b): Computational Efficiency Analysis")
    print("="*50)
    
    D = n_features
    N = n_samples
    
    print(f"Dataset Dimensions: N (samples)={N}, D (features)={D}")
    print("\nComputational Complexity Analysis:")
    print(f"1. Covariance Approach (C = D x D matrix):")
    print(f"   - Covariance Matrix C (D x D): O(N * D^2)")
    print(f"   - Eigendecomposition of C: O(D^3)")
    print(f"   - Total Complexity: O(N * D^2 + D^3) = O({N}*{D}^2 + {D}^3)")
    print(f"   - Measured Time: {time_cov:.4f} seconds")

    print(f"\n2. Gram Matrix Approach (K = N x N matrix):")
    print(f"   - Gram Matrix K (N x N): O(D * N^2)")
    print(f"   - Eigendecomposition of K: O(N^3)")
    print(f"   - Principal Component Calculation (U=X^T@V_K): O(D * N^2)")
    print(f"   - Total Complexity: O(D * N^2 + N^3) = O({D}*{N}^2 + {N}^3)")
    print(f"   - Measured Time: {time_gram:.4f} seconds")


    # --- Part (c): Reconstruction and Fidelity ---
    print("\n" + "="*50)
    print("Part (c): Reconstruction and Fidelity Analysis")
    print("="*50)

    arbitrary_index = 0
    original_image = X[arbitrary_index]
    original_image_centered = original_image - mean_vector
    
    n_values = [1, 2, 5, 10, 20]
    n_cols = len(n_values) + 1 # Original + 5 reconstructions
    fig, axes = plt.subplots(2, n_cols, figsize=(2 * n_cols, 4))
    fig.suptitle(f"Image Reconstruction (Arbitrary Image Index {arbitrary_index})", fontsize=14)

    # Function to calculate MSE
    def mse(a, b):
        # MSE = ||a - b||^2_2
        return np.linalg.norm(a - b, ord=2)**2

    # Plot Original Image
    axes[0, 0].imshow(original_image.reshape(img_dim, img_dim), cmap='gray')
    axes[0, 0].set_title("Original\n(Cov & Gram)")
    axes[0, 0].axis('off')
    axes[1, 0].imshow(original_image.reshape(img_dim, img_dim), cmap='gray')
    axes[1, 0].set_title("Original\n(Cov & Gram)")
    axes[1, 0].axis('off')
    
    
    # Reconstructions for Covariance Approach
    for i, n in enumerate(n_values):
        components_n = components_cov[:, :n] # D x n
        X_reduced = np.dot(original_image_centered, components_n) # (1 x D) @ (D x n) = 1 x n
        X_reconstructed_centered = np.dot(X_reduced, components_n.T) # (1 x n) @ (n x D) = 1 x D
        reconstructed_image = X_reconstructed_centered + mean_vector
        error = mse(original_image, reconstructed_image)
        axes[0, i+1].imshow(reconstructed_image.reshape(img_dim, img_dim), cmap='gray')
        axes[0, i+1].set_title(f"Cov, n={n}\nMSE: {error:.2f}")
        axes[0, i+1].axis('off')
        
    for i, n in enumerate(n_values):
        # 1. Transformation (Projection)
        components_n = components_gram[:, :n]
        X_reduced = np.dot(original_image_centered, components_n)
        
        # 2. Reconstruction (Inverse Transformation)
        X_reconstructed_centered = np.dot(X_reduced, components_n.T)
        reconstructed_image = X_reconstructed_centered + mean_vector
        
        # 3. Fidelity (MSE)
        error = mse(original_image, reconstructed_image)
        
        # 4. Visualization
        axes[1, i+1].imshow(reconstructed_image.reshape(img_dim, img_dim), cmap='gray')
        axes[1, i+1].set_title(f"Gram, n={n}\nMSE: {error:.2f}")
        axes[1, i+1].axis('off')
        
    # Add Row Labels
    axes[0, 0].set_ylabel("Covariance", fontsize=12)
    axes[1, 0].set_ylabel("Gram Matrix", fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    run_pca_analysis()