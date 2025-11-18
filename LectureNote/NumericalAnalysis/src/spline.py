import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 1. Original data points
x = np.array([0.25, 0.30, 0.39, 0.45, 0.53])
y = np.array([0.5, 0.5477, 0.6245, 0.6708, 0.7280])

left_boundary_condition = (1, 1.0000)
# S''(xn) = -5.0
right_boundary_condition = (1, 0.6868)

# bc_type is a tuple that defines left and right boundary conditions
custom_bc_type = (left_boundary_condition, right_boundary_condition)

# 3. Create CubicSpline interpolation function with custom boundary conditions
cs = CubicSpline(x, y, bc_type=custom_bc_type)

# 4. Generate new x values for plotting
x_new = np.linspace(x.min(), x.max(), 200)

# 5. Calculate interpolation results and second derivatives
y_interp = cs(x_new)
# Use the .derivative(nu=2) method of the CubicSpline object to calculate second derivatives
y_second_deriv = cs(x_new, nu=2)

# 6. Visualize the results
fig, ax1 = plt.subplots(figsize=(10, 6))

# --- Plot interpolation curve ---
ax1.set_xlabel("X axis")
ax1.set_ylabel("Y values (Curve)", color="blue")
# Plot original data points
ax1.scatter(x, y, color="red", label="Original Data Points", zorder=5)
# Plot interpolation curve
ax1.plot(
    x_new,
    y_interp,
    color="blue",
    linestyle="-",
    label="Cubic spline with custom second derivative boundaries",
)
ax1.tick_params(axis="y", labelcolor="blue")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle="--", alpha=0.6)

# --- Plot second derivative curve (for boundary condition verification) ---
ax2 = ax1.twinx()  # Create a secondary y-axis sharing the same x-axis
ax2.set_ylabel("Second derivative Y'' values", color="green")
ax2.plot(
    x_new,
    y_second_deriv,
    color="green",
    linestyle=":",
    label="Second derivative $S''(x)",
)
ax2.tick_params(axis="y", labelcolor="green")
ax2.legend(loc="upper right")

# Mark boundary second derivative values
ax2.scatter(
    [x.min(), x.max()],
    [cs(x.min(), nu=2), cs(x.max(), nu=2)],
    color="purple",
    marker="o",
    s=100,
    zorder=6,
    label="Boundaries $S''(x_0)$ and $S''(x_n)",
)
ax2.axhline(0.0, color="gray", linestyle="--")  # Mark Y''=0
ax2.axhline(-5.0, color="brown", linestyle="--")  # Mark Y''=-5.0

plt.title(
    "Cubic Spline Interpolation with Custom Second Derivative Boundary Conditions"
)
plt.savefig("./LectureNote/NumericalAnalysis/images/spline_2.pdf")

# 7. Verify boundary conditions
print("\n--- Boundary Condition Verification ---")
print(
    f"Second derivative S''(x0) at left boundary x={x.min()}: {cs(x.min(), nu=2):.4f}"
)
print(
    f"Second derivative S''(xn) at right boundary x={x.max()}: {cs(x.max(), nu=2):.4f}"
)

coeffs = cs.c  # 系数数组 (4 行, N-1 列)
knots = cs.x  # 节点 (即原始数据点 x)


# --- 函数：打印格式化的表达式 ---
def print_spline_expressions(coeffs, knots, precision=4):
    """
    打印 CubicSpline 的分段多项式表达式。
    形式: Pj(x) = c3 * (x - xj)^3 + c2 * (x - xj)^2 + c1 * (x - xj) + c0
    """
    N_intervals = coeffs.shape[1]

    print(f"\n## 📜 三次样条插值分段表达式 (共 {N_intervals} 段) 📜")
    print("-" * 60)

    for j in range(N_intervals):
        # 系数是倒序存储的：c3, c2, c1, c0
        c3, c2, c1, c0 = coeffs[3, j], coeffs[2, j], coeffs[1, j], coeffs[0, j]

        x_j = knots[j]
        x_j1 = knots[j + 1]

        # 格式化系数，保留指定精度
        c3_fmt = f"{c3:.{precision}f}"
        c2_fmt = f"{c2:.{precision}f}"
        c1_fmt = f"{c1:.{precision}f}"
        c0_fmt = f"{c0:.{precision}f}"

        # 构建表达式字符串
        # 注意：这里 t = (x - x_j)
        expression = f"P_{j}(x) = {c3_fmt}(x - {x_j})^3 "
        # 仅当系数不接近于零时才添加项，使表达式更简洁
        if abs(c2) > 1e-9:
            expression += f"+ {c2_fmt}(x - {x_j})^2 "
        if abs(c1) > 1e-9:
            expression += f"+ {c1_fmt}(x - {x_j}) "
        if abs(c0) > 1e-9:
            expression += f"+ {c0_fmt}"

        print(f"### 区间 {j}: [{x_j}, {x_j1}]")
        print(f"> {expression.replace('+ -', '- ')}")
        print("-" * 60)


# 执行输出
print_spline_expressions(coeffs, knots, precision=5)
