import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. Parameters
# --------------------------
N = 200
beta = 0.8
K_range = np.arange(1, 1001)
K_max = K_range[-1]



def optimal_success_probability(K, N, beta):
    q = K // N
    r = K % N
    return 1 - (r * (1 - beta) ** (q + 1) + (N - r) * (1 - beta) ** q) / N


def random_success_probability(K, N, beta):
    return 1 - (1 - beta / N) ** K


P_opt = np.array([optimal_success_probability(K, N, beta) for K in K_range])
P_random = np.array([random_success_probability(K, N, beta) for K in K_range])


# --------------------------
# 4. Maximum distance point
# --------------------------
distance = np.abs(P_opt - P_random)
idx_max = np.argmax(distance)

K_star = K_range[idx_max]
P_star = P_opt[idx_max]
Y_star = P_random[idx_max]

# --------------------------
# 5. Plot (style consistent)
# --------------------------
fig, ax = plt.subplots(figsize=(12, 6))

# Optimal theoretical curve
ax.plot(
    K_range,
    P_opt,
    linewidth=2.5,
    alpha=0.9,
    label="Theoretical Optimal",
    color="#1E88E5",
)
ax.fill_between(K_range, 0, P_opt, alpha=0.15, color="#1E88E5")

ax.plot(
    K_range,
    P_random,
    linewidth=2.5,
    alpha=0.9,
    label="Random Search Policy",
    color="#FF6B35",
)
ax.fill_between(K_range, 0, P_random, alpha=0.15, color="#FF6B35")

# Mark maximum distance points
ax.scatter(K_star, P_star, color="#1E88E5", zorder=5)
ax.scatter(K_star, Y_star, color="#FF6B35", zorder=5)
ax.vlines(K_star, Y_star, P_star, colors="black", linestyles="dotted", linewidth=2)

# Labels and style
ax.set_xlabel("Number of Searches (K)", fontsize=14, fontweight="600")
ax.set_title(
    f"Optimal Success Probability vs K (N={N}, Detection {beta:.0%})",
    fontsize=16,
    fontweight="700",
)

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

# Reference lines
for p in [0.25, 0.5, 0.75, 0.9]:
    ax.axhline(y=p, linestyle="--", alpha=0.2)

ax.legend(fontsize=12)

# --------------------------
# 6. Save
# --------------------------
output_path = "theoretical_vs_linear_N200.pdf"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()
