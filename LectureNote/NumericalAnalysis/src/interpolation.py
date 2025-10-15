import numpy as np
import matplotlib as mlp
mlp.use("Agg")
import matplotlib.pyplot as plt
import time
from tqdm import trange
from typing import List, Tuple, Callable, Dict, Any, Union
from scipy.interpolate import lagrange, KroghInterpolator


class InterpolationSolver:
    """
    A class for solving polynomial interpolation problems.

    This solver uses various methods to find the coefficients of the unique
    polynomial that passes through a given set of points. It supports
    pluggable methods and includes time measurement for the solution process.
    """

    def __init__(self, methods: Dict[str, Callable] = None):
        """
        Initializes the Interpolation Solver.

        The default method provided is 'vandermonde' (using the Vandermonde matrix).

        Args:
            methods: A dictionary where keys are the method names (str) and
                     values are the corresponding solving functions (Callable).
                     The signature of a solving function should be:
                     f(points: List[Tuple[float, float]]) -> np.ndarray.
                     Custom methods will be merged with the default ones.
        """
        # Default method: Vandermonde matrix solution
        # * will add more solvers in the future
        self.methods: Dict[str, Callable] = {
            "vandermonde": self._solve_vandermonde,
            "lagrange": self._solve_lagrange,
            "lagrange_fast": self._solve_lagrange_fast,
        }

        if methods:
            self.methods.update(methods)

        self.last_result: Union[np.ndarray, None] = None
        self.last_method: Union[str, None] = None
        self.last_time: Union[float, None] = None

    def _solve_vandermonde(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Solves for the polynomial coefficients using the Vandermonde matrix method.

        For n+1 data points, the method solves the linear system V * a = y,
        where V is the Vandermonde matrix and a is the vector of coefficients.

        Args:
            points: A list of (x, y) coordinate tuples. Must contain at least
                    one point.

        Returns:
            np.ndarray: The array of polynomial coefficients, ordered from
                        highest degree to lowest degree:
                        [a_n, a_{n-1}, ..., a_1, a_0]
                        for the polynomial P(x) = a_n * x^n + ... + a_0.
                        Returns an empty array if no points are given.

        Raises:
            ValueError: If the linear system is singular (e.g., duplicate x-values
                        or ill-conditioned data), preventing a unique solution.
        """
        n_points = len(points)
        if n_points == 0:
            return np.array([])

        n_degree = n_points - 1  # Degree of the polynomial

        # Separate x and y coordinates
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        V = np.vander(x, n_points)

        # Solve the linear system V * a = y
        try:
            coefficients = np.linalg.solve(V, y)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                f"Failed to solve the linear system. The matrix "
                f"might be singular (e.g., duplicate x-values). Error: {e}"
            )

        return coefficients

    def _solve_lagrange_fast(self, points: List[Tuple[float, float]]) -> np.ndarray:
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        coeff = lagrange(x, y).coef
        return coeff

    def _solve_lagrange(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        使用拉格朗日插值法展开并求和，以获得标准多项式系数。

        P(x) = sum_{j=0}^{n} y_j * l_j(x)，其中 l_j(x) 是拉格朗日基多项式。

        Args:
            points: 包含 (x, y) 坐标点的列表。

        Returns:
            np.ndarray: 多项式系数数组，顺序为从高次到低次：
                        [a_n, a_{n-1}, ..., a_1, a_0]。
        """
        n_points = len(points)
        if n_points == 0:
            return np.array([])

        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])

        # 初始化最终的多项式系数为零。系数顺序: [a_n, ..., a_0]
        final_coeffs = np.zeros(n_points)

        # 迭代计算每个基多项式 l_j(x) 的贡献
        for j in range(n_points):
            x_j = x[j]
            y_j = y[j]

            # compute pi (x - x_j) (k != j)
            roots = np.delete(x, j)

            # 使用 np.poly() 计算多项式 N_j(x) 的系数 (即 x - r1)(x - r2)...
            # 返回的系数顺序是 [a_m, a_{m-1}, ..., a_0]
            numerator_coeffs = np.poly(roots)

            # 计算分母 D_j = product_{k != j} (x_j - x_k)
            denominator = np.prod(x_j - roots)

            if np.isclose(denominator, 0):
                raise ValueError("Error, repeated x value is found.")

            # l_j(x) 的系数 = N_j(x) 的系数 / D_j
            l_j_coeffs = numerator_coeffs / denominator
            term_coeffs = l_j_coeffs * y_j
            final_coeffs = np.polyadd(final_coeffs, term_coeffs)

        return final_coeffs

    def _format_polynomial(self, coefficients: np.ndarray) -> str:
        """
        Formats the polynomial coefficients into a readable string representation.

        Args:
            coefficients: The array of polynomial coefficients
                          [a_n, a_{n-1}, ..., a_0].

        Returns:
            str: The string representation of the polynomial, e.g., "1.0x^3 + 2.0x + 1.0".
        """
        if len(coefficients) == 0:
            return "0"

        terms = []
        n_degree = len(coefficients) - 1

        for i, a in enumerate(coefficients):
            degree = n_degree - i

            if np.isclose(a, 0):
                continue

            sign = " + " if a > 0 else " - "
            if not terms:
                sign = "" if a > 0 else "-"

            abs_a = abs(a)
            coeff_str = f"{abs_a:.6f}"
            if np.isclose(abs_a, 1) and degree != 0:
                coeff_str = ""

            if degree == 0:
                # Constant term
                term_str = f"{sign}{abs_a:.6f}"
            elif degree == 1:
                # x term
                term_str = f"{sign}{coeff_str}x"
            else:
                # x^k term
                term_str = f"{sign}{coeff_str}x^{degree}"

            terms.append(term_str)

        return "".join(terms).strip() or "0"

    def solve(
        self, points: List[Tuple[float, float]], method: str = "vandermonde"
    ) -> Dict[str, Any]:
        """
        The core function to solve the polynomial interpolation using the specified method.

        The function measures the time taken by the chosen solving method.

        Args:
            points: A list of (x, y) coordinate tuples for interpolation.
            method: The name of the interpolation method (str) to use, which
                    must exist as a key in `self.methods`. Defaults to "vandermonde".

        Returns:
            Dict[str, Any]: A dictionary containing the solution results:
                            - 'coefficients': np.ndarray of the polynomial coefficients.
                            - 'method': The name of the method used.
                            - 'time_s': The elapsed time for the calculation in seconds.
                            - 'polynomial_str': A readable string of the polynomial P(x).

        Raises:
            ValueError: If the specified `method` is not recognized.
            ValueError: If the underlying solver function fails (e.g., singular matrix).
        """
        if method not in self.methods:
            raise ValueError(
                f"Unknown interpolation method: '{method}'. "
                f"Available methods: {list(self.methods.keys())}"
            )

        solver_func = self.methods[method]

        if not points:
            return {
                "coefficients": np.array([]),
                "method": method,
                "time_s": 0.0,
                "polynomial_str": "0",
            }

        start_time = time.time()
        coefficients = solver_func(points)
        end_time = time.time()
        elapsed_time = end_time - start_time
        poly_str = self._format_polynomial(coefficients)

        # Store results in the cache
        self.last_result = coefficients
        self.last_method = method
        self.last_time = elapsed_time

        return {
            "coefficients": coefficients,
            "method": method,
            "time_s": elapsed_time,
            "polynomial_str": poly_str,
        }


if __name__ == "__main__":
    solver = InterpolationSolver()
    NUM_POINTS = 50
    NUM_RUNS = 100
    TEST_METHODS = ["vandermonde", "lagrange", "lagrange_fast"]
    results_by_method = {method: [] for method in TEST_METHODS}

    print(f"--- Efficiency Test ---\nPoints N: {NUM_POINTS}, Repeated Nums: {NUM_RUNS}")

    for run in trange(NUM_RUNS):
        points_data = [
            (np.random.random() * 100, np.random.random() * 100)
            for _ in range(NUM_POINTS)
        ]

        for method_name in TEST_METHODS:
            try:
                result = solver.solve(points_data, method=method_name)
                results_by_method[method_name].append(result["time_s"])

            except np.linalg.LinAlgError:
                print(
                    f"Warning: {method_name} failed at run {run+1} due to singular matrix."
                )
                continue
            except ValueError as e:
                print(f"Warning: {method_name} failed at run {run+1} with error: {e}")
                continue

    print("\n--- Efficiency Test End ---")

    vandermonde_times = results_by_method["vandermonde"]
    lagrange_times = results_by_method["lagrange"]
    lagrange_fast_times = results_by_method["lagrange_fast"]

    # 打印统计摘要
    print("\n--- Cost Time Count ---")
    for method, times in results_by_method.items():
        if times:
            print(
                f"  {method.ljust(15)}: Mean = {np.mean(times):.6f}, Median = {np.median(times):.6f}, Min = {np.min(times):.6f}, Max = {np.max(times):.6f} (N={len(times)})"
            )
        else:
            print(f"  {method.ljust(15)}: No data recorded.")

    # 绘制箱线图
    data_to_plot = [vandermonde_times, lagrange_times, lagrange_fast_times]
    labels = TEST_METHODS

    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, vert=True)

    plt.title(
        f"Polynomial (N={NUM_POINTS}, {NUM_RUNS} repetitions)", fontsize=14
    )
    plt.ylabel("Eecutions times", fontsize=12)
    plt.xlabel("Method", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("./images/interpolation.pdf")
