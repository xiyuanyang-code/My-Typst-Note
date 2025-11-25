import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# --------------------------
# 1. Global Parameters
# --------------------------
n_points = 200
detect_prob = 0.8
threshold_list = [0.1, 0.5, 0.8]   # Three move thresholds
K_range = list(range(1, 1001))
n_simulations = 1000

save_dir = "/Users/xiyuanyang/Desktop/Dev/TypstNote/LectureNote/Probability/project/images"
os.makedirs(save_dir, exist_ok=True)


# --------------------------
# 2. Single Simulation
# --------------------------
def single_simulation_success_detection(strategy, K, n_points, detect_prob, move_threshold):
    """
    Simulate searching for up to K steps and return 1/0 whether detection succeeds.
    Target can move with probability move_threshold.
    """
    p = np.ones(n_points) / n_points
    target = np.random.choice(n_points)

    for _ in range(K):

        # Strategy pick
        if strategy == "bayesian":
            j = np.argmax(p)
        else:
            j = np.random.choice(n_points)

        # Detection
        if j == target and np.random.random() < detect_prob:
            return 1

        # Posterior update
        detection_j = p[j] * detect_prob
        undetected = 1 - detection_j

        if undetected > 1e-12:
            p[j] = p[j] * (1 - detect_prob) / undetected
            mask = np.arange(n_points) != j
            p[mask] = p[mask] / undetected
            p = p / np.sum(p)

        # Target movement
        if np.random.random() < move_threshold:
            if target == 0:
                target = 1
            elif target == n_points - 1:
                target = n_points - 2
            else:
                target += 1 if np.random.random() < 0.5 else -1

    return 0


# --------------------------
# 3. Compute Success Probability Curve
# --------------------------
def compute_success_curve(strategy, K_range, move_threshold):
    probs = []

    for K in tqdm(K_range, total = len(K_range)):
        count = 0
        for _ in range(n_simulations):
            count += single_simulation_success_detection(
                strategy, K, n_points, detect_prob, move_threshold
            )

        prob = count / n_simulations
        probs.append(prob)

    return probs


# --------------------------
# 4. Plot for Each Threshold
# --------------------------
def generate_all_figures():
    for th in threshold_list:
        print(f"\n=== Running threshold = {th} ===")

        bayes_curve = compute_success_curve("bayesian", K_range, th)
        rand_curve = compute_success_curve("random", K_range, th)

        # Draw figure
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(K_range, bayes_curve, label="Bayesian",
                linewidth=2.5, color="#1E88E5")
        ax.plot(K_range, rand_curve, label="Random",
                linewidth=2.5, color="#FF6B35")

        ax.fill_between(K_range, 0, bayes_curve, alpha=0.15, color="#1E88E5")
        ax.fill_between(K_range, 0, rand_curve, alpha=0.15, color="#FF6B35")

        ax.set_xlabel("Number of Searches (K)", fontsize=14)
        ax.set_ylabel("Success Probability", fontsize=14)
        ax.set_title(f"Success Probability vs K (threshold={th})", fontsize=16)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.25)

        out = os.path.join(save_dir, f"exp_4_threshold_{th}.pdf")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {out}")


# --------------------------
# 5. Run All
# --------------------------
if __name__ == "__main__":
    print("Starting Experiment 2 (3 figures)...")
    generate_all_figures()
    print("Finished.")
