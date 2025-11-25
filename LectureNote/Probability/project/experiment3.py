import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------
# 1. Model Parameter Configuration
# --------------------------
n_points = 200
K_list = [100, 400, 1000]
move_threshold_list = [0.1, 0.5, 0.8]
detect_prob = 0.8
n_simulations = 1000


# --------------------------
# 2. Single Simulation
# --------------------------
def single_simulation_success_count(strategy, K, n_points, detect_prob, move_threshold):
    if not 0 <= detect_prob <= 1:
        raise ValueError("detect_prob must be between 0 and 1")
    if K <= 0 or n_points <= 0:
        return 0

    p = np.ones(n_points) / n_points
    target_point = np.random.choice(n_points)

    for search_count in range(1, K + 1):
        # Select search point
        if strategy == 'bayesian':
            j = np.argmax(p)
        else:
            j = np.random.choice(n_points)

        # Detection
        if j == target_point:
            if np.random.random() < detect_prob:
                return search_count

        # Bayesian posterior update
        detection_this_search = p[j] * detect_prob
        current_undetected = 1 - detection_this_search

        if current_undetected > 1e-12:
            p[j] = (p[j] * (1 - detect_prob)) / current_undetected
            mask = np.arange(n_points) != j
            p[mask] = p[mask] / current_undetected
            p = p / np.sum(p)

        # Target movement
        if np.random.random() < move_threshold:
            if target_point == 0:
                target_point = 1
            elif target_point == n_points - 1:
                target_point = n_points - 2
            else:
                if np.random.random() < 0.5:
                    target_point -= 1
                else:
                    target_point += 1

    return K

# --------------------------
# 3. Multi Simulation Runner
# --------------------------
def run_convergence_simulations(strategy, K, n_points, detect_prob, n_simulations, move_threshold):
    success_counts = []
    running_averages = []

    for _ in range(n_simulations):
        count = single_simulation_success_count(
            strategy, K, n_points, detect_prob, move_threshold
        )
        success_counts.append(count)
        running_averages.append(np.mean(success_counts))

    return running_averages

# --------------------------
# 4. 9-plot Combined Figure
# --------------------------
def generate_9_plots():
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12), sharex=True)

    axes = axes.flatten()

    index = 0
    for move_threshold in move_threshold_list:
        for K in K_list:
            ax = axes[index]
            index += 1

            bayes_conv = run_convergence_simulations(
                'bayesian', K, n_points, detect_prob, n_simulations, move_threshold
            )

            random_conv = run_convergence_simulations(
                'random', K, n_points, detect_prob, n_simulations, move_threshold
            )

            x = range(1, n_simulations + 1)

            ax.plot(x, bayes_conv, linewidth=2, label='Bayesian')
            ax.plot(x, random_conv, linewidth=2, label='Random')

            ax.set_title(f"K={K} | move={move_threshold}", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

    fig.text(0.5, 0.04, 'Simulation Number', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Average Searches To Success', va='center',
             rotation='vertical', fontsize=14)

    plt.tight_layout(rect=[0.06, 0.06, 1, 1])

    output_path = "/Users/xiyuanyang/Desktop/Dev/TypstNote/LectureNote/Probability/project/images/exp_3.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"9-plot figure saved to: {output_path}")

# --------------------------
# 5. Main Execution
# --------------------------
if __name__ == "__main__":
    print("Starting 9-plot experiment...")
    generate_9_plots()
    print("Done.")
