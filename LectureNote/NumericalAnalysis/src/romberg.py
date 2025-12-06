"""Numerical integration methods using Romberg algorithm for exponential functions.

This module provides implementations of Romberg integration algorithm specifically
designed for calculating integrals of exponential functions with detailed
iteration table output.
"""

import math
from typing import Callable, List, Tuple


class ExponentialFunction:
    """Represents an exponential function for integration testing."""

    def __init__(self, coefficient: float = -1.0):
        """Initialize exponential function f(x) = e^(coefficient * x).

        Args:
            coefficient: The coefficient in the exponent (default: -1.0 for e^(-x)).
        """
        self.coefficient = coefficient

    def __call__(self, x: float) -> float:
        """Evaluate exponential function at given point.

        Args:
            x: Point at which to evaluate the function.

        Returns:
            Value of e^(coefficient * x) at point x.
        """
        return math.exp(self.coefficient * x)

    def exact_integral(self, a: float, b: float) -> float:
        """Calculate exact integral of e^(coefficient * x) from a to b.

        Args:
            a: Lower bound of integration.
            b: Upper bound of integration.

        Returns:
            Exact value of the definite integral.
        """
        if self.coefficient == 0:
            return b - a
        return (math.exp(self.coefficient * b) - math.exp(self.coefficient * a)) / self.coefficient

    def get_name(self) -> str:
        """Get string representation of the function.

        Returns:
            String representation of the exponential function.
        """
        if self.coefficient == -1:
            return "f(x) = e^(-x)"
        elif self.coefficient == 1:
            return "f(x) = e^(x)"
        elif self.coefficient == 0:
            return "f(x) = 1"
        else:
            return f"f(x) = e^({self.coefficient}*x)"


class RombergIntegrator:
    """Implements Romberg integration algorithm with detailed output."""

    def __init__(self, tolerance: float = 1e-5, max_iterations: int = 15):
        """Initialize Romberg integrator.

        Args:
            tolerance: Convergence tolerance for stopping criteria.
            max_iterations: Maximum number of iterations.
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def integrate_with_table(self, f: Callable[[float], float], a: float, b: float) -> Tuple[float, List[List[float]]]:
        """Calculate integral using Romberg integration and return complete table.

        Args:
            f: Function to integrate.
            a: Lower bound of integration.
            b: Upper bound of integration.

        Returns:
            Tuple containing (integral_result, romberg_table).
            romberg_table is a list of lists where romberg_table[i][j] represents
            the result at iteration i and extrapolation level j.
        """
        # Initialize Romberg matrix
        romberg_matrix = [[0.0] * self.max_iterations for _ in range(self.max_iterations)]
        iterations_used = 0

        # First step: Initial trapezoidal value T1 (n=1, only two endpoints)
        romberg_matrix[0][0] = 0.5 * (b - a) * (f(a) + f(b))
        iterations_used = 1

        # Iterative computation
        for i in range(1, self.max_iterations):
            # Calculate trapezoidal value T_{2^i} with halved step size
            h = (b - a) / (2 ** i)
            sum_f = 0.0

            # Accumulate function values at new odd-numbered points
            for k in range(1, 2 ** (i - 1) + 1):
                x = a + (2 * k - 1) * h
                sum_f += f(x)

            # Trapezoidal rule recursive formula
            romberg_matrix[i][0] = 0.5 * romberg_matrix[i - 1][0] + h * sum_f

            # Richardson extrapolation for higher accuracy
            for j in range(1, i + 1):
                romberg_matrix[i][j] = romberg_matrix[i][j - 1] + (
                    romberg_matrix[i][j - 1] - romberg_matrix[i - 1][j - 1]
                ) / (4 ** j - 1)

            iterations_used = i + 1

            # Check convergence using diagonal elements
            if abs(romberg_matrix[i][i] - romberg_matrix[i - 1][i - 1]) < self.tolerance:
                print(f"\nConvergence achieved after {i + 1} iterations (error < {self.tolerance})")
                break

        # Extract valid portion of the table
        romberg_table = []
        for i in range(iterations_used):
            row = romberg_matrix[i][:i + 1]  # Row i has i+1 valid values
            romberg_table.append(row)

        return romberg_matrix[iterations_used - 1][iterations_used - 1], romberg_table


class TableFormatter:
    """Handles formatting and printing of Romberg integration tables."""

    @staticmethod
    def get_column_headers(table_width: int) -> List[str]:
        """Generate column headers for Romberg table.

        Args:
            table_width: Number of columns in the table.

        Returns:
            List of column headers.
        """
        headers = []
        for j in range(table_width):
            if j == 0:
                headers.append("T")  # Trapezoidal
            elif j == 1:
                headers.append("S")  # Simpson (1 extrapolation)
            elif j == 2:
                headers.append("C")  # Cotes (2 extrapolations)
            else:
                headers.append(f"R{j - 2}")  # Romberg (3+ extrapolations)
        return headers

    @staticmethod
    def print_romberg_table(table: List[List[float]]) -> None:
        """Print formatted Romberg integration table.

        Args:
            table: Romberg table to print, where table[i][j] represents
                   the result at iteration i and extrapolation level j.
        """
        print("\n=== Romberg Integration Table (T-Table) ===")

        # Generate headers
        headers = TableFormatter.get_column_headers(len(table[-1]))

        # Print header row
        header_str = f"{'Iteration':<10}" + "\t".join([f"{h:<14}" for h in headers])
        print(header_str)
        print("-" * (10 + 15 * len(headers)))

        # Print data rows
        for i, row in enumerate(table):
            # Row title: T_{2^i} (2^0=1, 2^1=2, 2^2=4, ...)
            row_title = f"T_{2**i}"
            # Format row data
            row_data = "\t".join([f"{val:.8f}" for val in row])
            print(f"{row_title:<10}" + row_data)


class ErrorFunctionCalculator:
    """Calculates error function using numerical integration."""

    def __init__(self, tolerance: float = 1e-5):
        """Initialize error function calculator.

        Args:
            tolerance: Integration tolerance.
        """
        self.integrator = RombergIntegrator(tolerance=tolerance)
        self.exponential_func = ExponentialFunction(coefficient=-1.0)

    def calculate_error_function_component(self, a: float = 0.0, b: float = 1.0) -> Tuple[float, List[List[float]]]:
        """Calculate the core component of error function: ∫₀¹ e^(-x) dx.

        Args:
            a: Lower bound of integration.
            b: Upper bound of integration.

        Returns:
            Tuple containing (integral_result, romberg_table).
        """
        return self.integrator.integrate_with_table(self.exponential_func, a, b)

    def calculate_scaled_result(self, integral_core: float) -> float:
        """Calculate (2/√π) * ∫₀¹ e^(-x) dx.

        Args:
            integral_core: Result of the core integral.

        Returns:
            Scaled final result.
        """
        factor = 2 / math.sqrt(math.pi)
        return factor * integral_core

    def calculate_exact_values(self, a: float = 0.0, b: float = 1.0) -> Tuple[float, float]:
        """Calculate exact values for comparison.

        Args:
            a: Lower bound of integration.
            b: Upper bound of integration.

        Returns:
            Tuple containing (exact_core_integral, exact_scaled_result).
        """
        exact_core = self.exponential_func.exact_integral(a, b)
        exact_scaled = self.calculate_scaled_result(exact_core)
        return exact_core, exact_scaled


def main() -> None:
    """Main function to demonstrate Romberg integration for error function calculation."""
    # Initialize calculator
    calculator = ErrorFunctionCalculator(tolerance=1e-5)

    # Set integration bounds
    a, b = 0.0, 1.0

    # Calculate core integral ∫₀¹ e^(-x) dx
    integral_core, romberg_table = calculator.calculate_error_function_component(a, b)

    # Print Romberg table
    TableFormatter.print_romberg_table(romberg_table)

    # Calculate scaled result: (2/√π) * ∫₀¹ e^(-x) dx
    integral_final = calculator.calculate_scaled_result(integral_core)
    factor = 2 / math.sqrt(math.pi)

    # Display results
    print("\n=== Final Calculation Results ===")
    print(f"∫₀¹ e^(-x) dx = {integral_core:.8f}")
    print(f"2/√π = {factor:.8f}")
    print(f"(2/√π) * ∫₀¹ e^(-x) dx = {integral_final:.8f}")

    # Compare with exact values
    exact_core, exact_final = calculator.calculate_exact_values(a, b)

    print("\n=== Exact Value Comparison ===")
    print(f"Exact ∫₀¹ e^(-x) dx = {exact_core:.8f}")
    print(f"Exact target integral = {exact_final:.8f}")
    print(f"Computation error = {abs(integral_final - exact_final):.8f}")


if __name__ == "__main__":
    main()