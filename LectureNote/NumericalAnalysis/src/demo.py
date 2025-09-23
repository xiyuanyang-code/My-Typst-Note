# simple usage of numpy, scipy and matplotlib
import numpy as np
import scipy
import math
import decimal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def quadratic_model(x, a, b, c):
    """
    一个二次函数模型 f(x) = ax^2 + bx + c
    """
    return a * x**2 + b * x + c


def demo_1():

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


def demo_2():
    # using decimal for 10-nary operations
    # much slower than binary operations in CPU, but has much more accuracy!
    print("0.1 + 0.2 == 0.3? ", (0.1 + 0.2 == 0.3))
    print(0.1 + 0.2)

    # using another method...
    print(float(decimal.Decimal("0.1") + decimal.Decimal("0.2")))

    print("Why? 0.1 is ...")
    float_number_1 = 0.1
    decimal_representation_1 = decimal.Decimal.from_float(float_number_1)
    print(decimal_representation_1)
    print(float(decimal_representation_1))

    print("Why? 0.2 is ...")
    float_number_2 = 0.2
    decimal_representation_2 = decimal.Decimal.from_float(float_number_2)
    print(decimal_representation_2)
    print(float(decimal_representation_2))

    print("Plusing result: ")
    decimal_result = decimal_representation_2 + decimal_representation_1
    print(decimal_result)
    print(float(decimal_result))


def get_significant_figure(ref: str, est: str) -> int:
    """计算实数估计值 est 相对于实数参考值 ref 的有效数字位数

    Args:
    ref (str): 实数参考值的字符串形式
    est (str): 实数估计值的字符串形式

    Returns:
    n (int): 有效数字位数
    """
    try:
        ref_val = float(ref)
        est_val = float(est)
    except ValueError:
        raise ValueError("输入必须是有效的实数字符串。")

    if ref_val == est_val:
        return 15
    if ref_val == 0:
        return 0

    error = abs(ref_val - est_val)

    if ref_val != 0:
        ref_magnitude = math.floor(math.log10(abs(ref_val)))
    else:
        return 0

    if error != 0:
        error_magnitude = math.floor(math.log10(error))
    else:
        return 15

    sig_fig = int(ref_magnitude - error_magnitude)
    last_sig_fig_magnitude = 10**error_magnitude
    if error < 0.5 * last_sig_fig_magnitude:
        sig_fig += 1

    return max(0, sig_fig)


if __name__ == "__main__":
    # demo_1()
    # demo_2()
    print(get_significant_figure(2.2530, 2.3000))
    # assert get_significant_figure("2.2530", "2.3000") == 2
