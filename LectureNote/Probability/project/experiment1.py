import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Model Parameter Configuration
# --------------------------
n_points = 200
K_list = [50, 100, 200, 400, 800, 1000]  # Different K values to test
detect_prob = 0.8  # Detection probability for single search (fixed)
n_simulations = 1000  # Number of simulations for convergence analysis


def single_simulation_success_count(strategy, K, n_points, detect_prob):
    """
    Single simulation: Given strategy and K max searches, return number of searches needed for success
    strategy: 'bayesian' or 'random'
    K: Maximum number of searches
    Returns: number of searches needed for first detection (0 if not found within K searches)
    """
    # Input validation
    if not 0 <= detect_prob <= 1:
        raise ValueError("detect_prob must be between 0 and 1")
    if K <= 0 or n_points <= 0:
        return 0

    # Initial prior probability: uniform distribution
    p = np.ones(n_points) / n_points
    target_point = np.random.choice(n_points)  # Randomly place the target

    for search_count in range(1, K + 1):
        # Step 1: Select the point to search this time
        if strategy == "bayesian":
            # Bayesian strategy: select point with maximum posterior probability
            j = np.argmax(p)
        elif strategy == "random":
            # Random strategy: randomly select a point
            j = np.random.choice(n_points)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Step 2: Check if we found the target
        if j == target_point:
            # Detection attempt
            if np.random.random() < detect_prob:
                return search_count  # Success!

        # Step 3: Bayesian posterior update (both strategies share same update rule)
        detection_this_search = p[j] * detect_prob
        current_undetected = 1 - detection_this_search

        # Update posterior probabilities
        if current_undetected > 1e-12:
            p[j] = (p[j] * (1 - detect_prob)) / current_undetected
            mask = np.arange(n_points) != j
            p[mask] = p[mask] / current_undetected
            p = p / np.sum(p)  # Renormalize

    return K  # Return K if not found within K searches (treat as worst case)


# --------------------------
# Multiple Simulations for Convergence Analysis
# --------------------------
def run_convergence_simulations(strategy, K, n_points, detect_prob, n_simulations):
    """
    Run multiple simulations and track running average of search success counts
    Returns cumulative running averages
    """
    success_counts = []
    running_averages = []

    for i in range(n_simulations):
        # Get success count for this simulation
        count = single_simulation_success_count(strategy, K, n_points, detect_prob)
        success_counts.append(count)

        # Calculate running average
        running_avg = np.mean(success_counts)
        running_averages.append(running_avg)

        if (i + 1) % 100 == 0:
            print(
                f"Strategy: {strategy:8s} | K={K:3d} | Simulation {i+1:4d} | Running Avg: {running_avg:.2f}"
            )

    return running_averages


def generate_figure():
    """
    Put 6 K-values convergence curves into one big figure (2 rows × 3 columns).
    """
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10), sharex=True)

    axes = axes.flatten()  # 方便迭代

    for ax, K in zip(axes, K_list):
        # 运行模拟
        bayes_conv = run_convergence_simulations(
            strategy="bayesian",
            K=K,
            n_points=n_points,
            detect_prob=detect_prob,
            n_simulations=n_simulations,
        )

        random_conv = run_convergence_simulations(
            strategy="random",
            K=K,
            n_points=n_points,
            detect_prob=detect_prob,
            n_simulations=n_simulations,
        )

        x = range(1, n_simulations + 1)

        # 绘制曲线
        ax.plot(x, bayes_conv, linewidth=2, label="Bayesian")
        ax.plot(x, random_conv, linewidth=2, label="Random")

        # 标题
        ax.set_title(f"K={K} | Detection={detect_prob:.0%}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    # 统一 x / y 轴标签
    fig.text(0.5, 0.04, "Simulation Number", ha="center", fontsize=14)
    fig.text(
        0.04,
        0.5,
        "Average Searches To Success",
        va="center",
        rotation="vertical",
        fontsize=14,
    )

    plt.tight_layout(rect=[0.06, 0.06, 1, 1])

    output_path = "/Users/xiyuanyang/Desktop/Dev/TypstNote/LectureNote/Probability/project/images/exp_1.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"2×3 combined figure saved to: {output_path}")


# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    print("Starting Experiment 1: Search Strategy Convergence Analysis")
    print("=" * 60)
    print(f"Configuration: {n_points} points, detection rate {detect_prob:.0%}")
    print(f"K values to test: {K_list}")
    print(f"Simulations per K value: {n_simulations}")
    print("=" * 60)

    generate_figure()

    print(f"\n{'='*60}")
    print("Experiment 1 Complete!")
    print(f"All plots saved to 'images/' directory")
    print("=" * 60)
