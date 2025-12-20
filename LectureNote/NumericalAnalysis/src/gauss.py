"""
高斯消元法和列主元素消去法求解线性方程组
保留四位小数并显示中间过程
"""

import numpy as np
from typing import Tuple
import logging


class GaussianElimination:
    """高斯消元法求解线性方程组"""

    def __init__(self, precision: int = 4):
        """
        初始化高斯消元法求解器

        Args:
            precision: 小数精度，默认4位
        """
        self.precision = precision
        self.logger = logging.getLogger(__name__)

    def round_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """将矩阵元素四舍五入到指定精度"""
        return np.round(matrix, self.precision)

    def print_augmented_matrix(self, A: np.ndarray, b: np.ndarray, step: str = "") -> None:
        """打印增广矩阵"""
        print(f"\n{step}")
        print("增广矩阵 [A|b]:")
        for i in range(len(A)):
            row_elements = "  ".join(f"{A[i, j]:8.4f}" for j in range(len(A)))
            print(f"[{row_elements} | {b[i]:8.4f}]")
        print()

    def forward_elimination(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        基本高斯消元法的前向消元过程

        Args:
            A: 系数矩阵
            b: 右端向量

        Returns:
            消元后的上三角矩阵和对应的右端向量
        """
        n = len(A)
        A = A.copy().astype(float)
        b = b.copy().astype(float)

        print("=" * 60)
        print("开始高斯消元法求解")
        print("=" * 60)

        self.print_augmented_matrix(A, b, "初始矩阵:")

        for k in range(n - 1):
            print(f"第 {k + 1} 步消元：")
            print("-" * 40)

            # 如果主元为0，交换行
            if abs(A[k, k]) == 0:
                for i in range(k + 1, n):
                    if abs(A[i, k]) > 0:
                        A[[k, i]] = A[[i, k]]
                        b[[k, i]] = b[[i, k]]
                        print(f"  交换第 {k + 1} 行和第 {i + 1} 行")
                        break

            pivot = A[k, k]
            print(f"  主元 A[{k + 1},{k + 1}] = {pivot:.4f}")

            for i in range(k + 1, n):
                if abs(A[i, k]) > 0:
                    multiplier = round(A[i, k] / pivot, self.precision)
                    print(f"  第 {i + 1} 行 = 第 {i + 1} 行 - {multiplier:.4f} × 第 {k + 1} 行")

                    for j in range(k, n):
                        product = round(multiplier * A[k, j], self.precision)
                        A[i, j] = round(A[i, j] - product, self.precision)

                    product_b = round(multiplier * b[k], self.precision)
                    b[i] = round(b[i] - product_b, self.precision)

            self.print_augmented_matrix(A, b, f"第 {k + 1} 步消元后：")

        return A, b

    def partial_pivot_elimination(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        列主元素消去法的前向消元过程

        Args:
            A: 系数矩阵
            b: 右端向量

        Returns:
            消元后的上三角矩阵和对应的右端向量
        """
        n = len(A)
        A = A.copy().astype(float)
        b = b.copy().astype(float)

        print("=" * 60)
        print("开始列主元素消去法求解")
        print("=" * 60)

        self.print_augmented_matrix(A, b, "初始矩阵:")

        for k in range(n - 1):
            print(f"第 {k + 1} 步消元：")
            print("-" * 40)

            # 寻找列主元
            max_row = k
            max_val = abs(A[k, k])

            for i in range(k + 1, n):
                if abs(A[i, k]) > max_val:
                    max_val = abs(A[i, k])
                    max_row = i

            # 如果需要，交换行
            if max_row != k:
                A[[k, max_row]] = A[[max_row, k]]
                b[[k, max_row]] = b[[max_row, k]]
                print(f"  列主元：第 {max_row + 1} 行的 {A[k, k]:.4f} 是第 {k + 1} 列的最大值")
                print(f"  交换第 {k + 1} 行和第 {max_row + 1} 行")
            else:
                print(f"  列主元：第 {k + 1} 行的 {A[k, k]:.4f} 是第 {k + 1} 列的最大值")

            pivot = A[k, k]
            print(f"  主元 A[{k + 1},{k + 1}] = {pivot:.4f}")

            for i in range(k + 1, n):
                if abs(A[i, k]) > 1e-10:  # 只对非零元素进行消元
                    # 严格模拟手算：乘子也必须四舍五入到四位小数
                    multiplier = round(A[i, k] / pivot, self.precision)
                    print(f"  第 {i + 1} 行 = 第 {i + 1} 行 - {multiplier:.4f} × 第 {k + 1} 行")

                    # 模拟手算的分步计算：每个元素更新后立即四舍五入
                    for j in range(k, n):
                        # 先计算乘法并四舍五入，再进行减法并四舍五入
                        product = round(multiplier * A[k, j], self.precision)
                        A[i, j] = round(A[i, j] - product, self.precision)

                    # 同样处理向量 b
                    product_b = round(multiplier * b[k], self.precision)
                    b[i] = round(b[i] - product_b, self.precision)

            self.print_augmented_matrix(A, b, f"第 {k + 1} 步消元后：")

        return A, b

    def back_substitution(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        回代求解

        Args:
            A: 上三角矩阵
            b: 右端向量

        Returns:
            解向量
        """
        n = len(A)
        x = np.zeros(n)

        print("=" * 60)
        print("开始回代求解")
        print("=" * 60)

        for i in range(n - 1, -1, -1):
            print(f"\n求解 x[{i + 1}]:")

            # 模拟手算的累加过程：每项计算后立即四舍五入
            sum_ax = 0
            for j in range(i + 1, n):
                term = round(A[i, j] * x[j], self.precision)
                sum_ax = round(sum_ax + term, self.precision)

            # 分步计算除法，模拟手算过程
            numerator = round(b[i] - sum_ax, self.precision)
            x[i] = round(numerator / A[i, i], self.precision)

            if i < n - 1:
                ax_terms = " + ".join([f"{A[i, j]:.4f}×{x[j]:.4f}" for j in range(i + 1, n)])
                print(f"  {A[i, i]:.4f}×x[{i + 1}] + ({ax_terms}) = {b[i]:.4f}")
                print(f"  {A[i, i]:.4f}×x[{i + 1}] = {b[i]:.4f} - ({sum_ax:.4f})")
            else:
                print(f"  {A[i, i]:.4f}×x[{i + 1}] = {b[i]:.4f}")

            print(f"  x[{i + 1}] = {x[i]:.4f}")

        return x

    def solve(self, A: np.ndarray, b: np.ndarray, method: str = "gaussian") -> np.ndarray:
        """
        求解线性方程组

        Args:
            A: 系数矩阵
            b: 右端向量
            method: 求解方法 ("gaussian" 或 "partial_pivot")

        Returns:
            解向量
        """
        A = A.astype(float)
        b = b.astype(float)

        # 前向消元
        if method == "gaussian":
            A_upper, b_upper = self.forward_elimination(A, b)
        elif method == "partial_pivot":
            A_upper, b_upper = self.partial_pivot_elimination(A, b)
        else:
            raise ValueError("method 必须是 'gaussian' 或 'partial_pivot'")

        # 回代求解
        x = self.back_substitution(A_upper, b_upper)

        return x

    def verify_solution(self, A: np.ndarray, b: np.ndarray, x: np.ndarray) -> None:
        """验证解的正确性"""
        print("\n" + "=" * 60)
        print("验证解的正确性")
        print("=" * 60)

        print("\n验证 Ax = b:")
        for i in range(len(b)):
            ax_result = sum(A[i, j] * x[j] for j in range(len(x)))
            ax_result = round(ax_result, self.precision)
            print(f"第 {i + 1} 个方程: {ax_result:.4f} ≈ {b[i]:.4f} (误差: {abs(ax_result - b[i]):.4f})")

        # 计算残差
        residual = A @ x - b
        norm_residual = np.linalg.norm(residual)
        print(f"\n残差范数: ||Ax - b||₂ = {norm_residual:.6f}")


def main():
    """主函数：示例 4×4 线性方程组"""

    # 示例 4×4 线性方程组 Ax = b
    A = np.array([
        [0.4096, 0.1234, 0.3678, 0.2943],
        [0.2246, 0.3872, 0.4015, 0.1129],
        [0.3645, 0.1920, 0.3781, 0.0643],
        [0.1784, 0.4002, 0.2786, 0.3927]
    ])

    b = np.array([0.4043, 0.1550, 0.4240, -0.2557])

    print("=" * 60)
    print("待求解的线性方程组:")
    print("=" * 60)

    for i in range(4):
        terms = []
        for j in range(4):
            if A[i, j] >= 0:
                terms.append(f"{A[i, j]}x{j+1}")
            else:
                terms.append(f"{A[i, j]}x{j+1}")
        equation = " + ".join(terms).replace("+ -", "- ")
        print(f"  {equation} = {b[i]}")

    solver = GaussianElimination(precision=4)

    print("\n\n" + "↔" * 30)
    print("方法一：基本高斯消元法")
    print("↔" * 30)

    try:
        x_gaussian = solver.solve(A, b, method="gaussian")
        solver.verify_solution(A, b, x_gaussian)
    except Exception as e:
        print(f"基本高斯消元法求解失败: {e}")

    print("\n\n" + "↔" * 30)
    print("方法二：列主元素消去法")
    print("↔" * 30)

    try:
        x_pivot = solver.solve(A, b, method="partial_pivot")
        solver.verify_solution(A, b, x_pivot)
    except Exception as e:
        print(f"列主元素消去法求解失败: {e}")


if __name__ == "__main__":
    main()