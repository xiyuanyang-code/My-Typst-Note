# simple usage of numpy, scipy and matplotlib
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def quadratic_model(x, a, b, c):
    """
    一个二次函数模型 f(x) = ax^2 + bx + c
    """
    return a * x**2 + b * x + c


def demo():

    x_data = np.linspace(0, 20, num=200)
    y_true = 2 * x_data**2 + 3 * x_data + 5
    np.random.seed(0)
    y_noise = 25 * np.random.normal(size=len(x_data))
    y_data = y_true + y_noise

    popt, pcov = curve_fit(quadratic_model, x_data, y_data)
    # curve_fit 使用线性最小二乘法

    print(f"拟合出的参数 [a, b, c] 为: {popt}")
    print(f"真实参数 [a, b, c] 为: [2, 3, 5]")

    x_fit = np.linspace(0, 20, 500)
    y_fit = quadratic_model(x_fit, *popt)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label="Original Data", color="blue", alpha=0.6)
    plt.plot(x_fit, y_fit, label="Curve", color="red", linewidth=2)
    plt.plot(
        x_data, y_true, label="True Function", color="green", linestyle="--", alpha=0.7
    )

    plt.title("Scipy Demo")
    plt.xlabel("X Value")
    plt.ylabel("Y Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("./images/demo.png")


if __name__ == "__main__":
    demo()
