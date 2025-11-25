import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------
# 1. Model Parameter Configuration
# --------------------------
n_points = 200
detect_prob = 0.8  # Detection probability for single search (fixed)
K_range = range(1, 1001)  # K from 1 to 1000
n_simulations = 1000  # Number of simulations for each K value

# Create images directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# --------------------------
# 2. Single Simulation Function for Success Detection
# --------------------------
def single_simulation_success_detection(strategy, K, n_points, detect_prob):
    """
    Single simulation: Given strategy and K searches, return whether target was detected (1/0)
    strategy: 'bayesian' or 'random'
    K: Number of searches
    Returns: 1 if target detected within K searches, 0 if not
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
        if strategy == 'bayesian':
            # Bayesian strategy: select point with maximum posterior probability
            j = np.argmax(p)
        elif strategy == 'random':
            # Random strategy: randomly select a point
            j = np.random.choice(n_points)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Step 2: Check if we found the target
        if j == target_point:
            # Detection attempt
            if np.random.random() < detect_prob:
                return 1  # Success!

        # Step 3: Bayesian posterior update (both strategies share same update rule)
        detection_this_search = p[j] * detect_prob
        current_undetected = 1 - detection_this_search

        # Update posterior probabilities
        if current_undetected > 1e-12:
            p[j] = (p[j] * (1 - detect_prob)) / current_undetected
            mask = np.arange(n_points) != j
            p[mask] = p[mask] / current_undetected
            p = p / np.sum(p)  # Renormalize

    return 0  # Not found within K searches

# --------------------------
# 3. Multiple Simulations for Success Probability
# --------------------------
def calculate_success_probability(strategy, K_range, n_points, detect_prob, n_simulations):
    """
    Calculate success probability for each K value
    Returns array of success probabilities
    """
    success_probs = []

    for K in K_range:
        # Run n_simulations for this K value
        success_count = 0
        for _ in range(n_simulations):
            success = single_simulation_success_detection(strategy, K, n_points, detect_prob)
            success_count += success

        success_prob = success_count / n_simulations
        success_probs.append(success_prob)

        # Print progress for key K values
        if K in [50, 100, 200, 400, 600, 800, 1000]:
            print(f"Strategy: {strategy:8s} | K={K:4d} | Success Probability: {success_prob:.3f}")

    return success_probs

# --------------------------
# 4. Generate Combined 2-Subplot Figure
# --------------------------
def generate_combined_figure():
    """Generate combined figure with 2 subplots showing success probability vs K"""

    print(f"\n{'='*60}")
    print("Running Experiment 2: Search Success Probability Analysis")
    print(f"{'='*60}")

    # Calculate success probabilities for both strategies
    print("\nCalculating Bayesian strategy success probabilities...")
    bayes_probs = calculate_success_probability(
        strategy='bayesian',
        K_range=K_range,
        n_points=n_points,
        detect_prob=detect_prob,
        n_simulations=n_simulations
    )

    print("\nCalculating Random strategy success probabilities...")
    random_probs = calculate_success_probability(
        strategy='random',
        K_range=K_range,
        n_points=n_points,
        detect_prob=detect_prob,
        n_simulations=n_simulations
    )
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot both strategies in one figure
    ax.plot(K_range, bayes_probs, linewidth=2.5, alpha=0.9,
            label='Bayesian', color='#1E88E5')
    ax.plot(K_range, random_probs, linewidth=2.5, alpha=0.9,
            label='Random', color='#FF6B35')

    # Shaded area
    ax.fill_between(K_range, 0, bayes_probs, alpha=0.15, color='#1E88E5')
    ax.fill_between(K_range, 0, random_probs, alpha=0.15, color='#FF6B35')

    ax.set_xlabel('Number of Searches (K)', fontsize=14, fontweight='600')
    ax.set_ylabel('Success Probability', fontsize=14, fontweight='600')
    ax.set_title(f'Success Probability vs K ({n_points} Points, Detection {detect_prob:.0%})',
                 fontsize=16, fontweight='700')

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Horizontal reference lines
    for prob in [0.25, 0.5, 0.75, 0.9]:
        ax.axhline(y=prob, color='#9E9E9E', linestyle='--', alpha=0.2)

    # Legend
    ax.legend(fontsize=12)

    # Save figure
    output_path = "/Users/xiyuanyang/Desktop/Dev/TypstNote/LectureNote/Probability/project/images/exp_2_singleplot.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSingle combined figure saved to: {output_path}")

    return bayes_probs, random_probs

# --------------------------
# 5. Main Execution
# --------------------------
if __name__ == "__main__":
    print("Starting Experiment 2: Search Success Probability Analysis")
    print("="*60)
    print(f"Configuration: {n_points} points, detection rate {detect_prob:.0%}")
    print(f"K range: 1 to {max(K_range)}, simulations per K: {n_simulations}")
    print("="*60)

    bayes_probs, random_probs = generate_combined_figure()

    print(f"\n{'='*60}")
    print("Experiment 2 Complete!")
    print("Combined plot saved to 'images/' directory")
    print("="*60)