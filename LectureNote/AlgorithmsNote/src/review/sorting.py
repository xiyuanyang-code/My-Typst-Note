"""
Sorting Algorithms Practice Framework.

This module provides function signatures and testing infrastructure
for implementing and testing various sorting algorithms.
"""

import random
import time
from typing import List, Callable, Tuple


def generate_random_array(size: int, min_val: int = 0, max_val: int = 100) -> list:
    """
    Generate a random integer array.

    Args:
        size: Size of the array to generate
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)

    Returns:
        List of random integers
    """
    return [random.randint(min_val, max_val) for _ in range(size)]


def get_reverse_length(element: str):
    length = len(element)
    return length**2 - 8 * length


# =============================================================================
# Sorting Algorithm Interfaces (To Be Implemented)
# =============================================================================


def merge_sort(arr: List[int]) -> List[int]:
    """
    Sort an array using merge sort algorithm.

    Time Complexity: O(n log n)
    Space Complexity: O(n)

    Args:
        arr: Input array to be sorted

    Returns:
        Sorted array in ascending order
    """
    if len(arr) == 1 or len(arr) == 0:
        return arr
    length = len(arr)
    pivot_index = length // 2
    left_part = merge_sort(arr[:pivot_index])
    right_part = merge_sort(arr[pivot_index:])

    # merge to lists
    i = 0
    j = 0
    new_sorted_array = []
    while i < len(left_part) and j < len(right_part):
        current_left = left_part[i]
        current_right = right_part[j]
        if current_left <= current_right:
            new_sorted_array.append(current_left)
            i += 1
        else:
            new_sorted_array.append(current_right)
            j += 1
    new_sorted_array.extend(left_part[i:])
    new_sorted_array.extend(right_part[j:])
    return new_sorted_array


def quick_sort(arr: List[int]) -> List[int]:
    def quick_sort_(low: int, high: int):
        if low == high:
            return
        if low < high:
            # do partition
            pivot_index = partition(low, high)
            quick_sort_(low, pivot_index)
            quick_sort_(pivot_index + 1, high)

    def partition(left: int, right: int):
        pivot_idx = random.randint(left, right)
        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
        i = left - 1
        for j in range(left, right):
            if arr[j] < arr[right]:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]

        arr[i + 1], arr[right] = arr[right], arr[i + 1]
        return i + 1

    quick_sort_(0, len(arr) - 1)
    return arr


def direct_access_array_sort(arr: List[int]) -> List[int]:
    """
    Sort an array using direct access array sort.

    This algorithm assumes integer keys within a known range.
    Uses direct indexing to place elements in their correct positions.

    Time Complexity: O(n + k) where k is the range of keys
    Space Complexity: O(n + k)

    Args:
        arr: Input array to be sorted (non-negative integers)

    Returns:
        Sorted array in ascending order
    """
    if not arr:
        return []
    max_values = max(arr)
    new_arrays = []
    bucket = [0] * (max_values + 1)
    for value in arr:
        bucket[value] += 1

    for bucket_index, bucket_value in enumerate(bucket):
        new_arrays.extend([bucket_index] * bucket_value)

    return new_arrays


def counting_sort(arr: List[int]) -> List[int]:
    """
    Sort an array using counting sort algorithm.

    This is a stable sorting algorithm for integers within a specific range.
    Counts occurrences of each unique element and uses arithmetic to
    determine position of each element in output.

    Time Complexity: O(n + k) where k is the range of keys
    Space Complexity: O(n + k)

    Args:
        arr: Input array to be sorted (non-negative integers)

    Returns:
        Sorted array in ascending order
    """
    if not arr:
        return []
    max_values = max(arr)
    new_arrays = []
    bucket = [0] * (max_values + 1)
    for value in arr:
        bucket[value] += 1

    for bucket_index, bucket_value in enumerate(bucket):
        new_arrays.extend([bucket_index] * bucket_value)

    return new_arrays


def radix_sort(arr: List[int]) -> List[int]:
    """
    Sort an array using radix sort algorithm.

    Sorts integers by processing individual digits, from least significant
    to most significant. Typically uses counting sort as a subroutine.

    Time Complexity: O(d * (n + k)) where d is number of digits
    Space Complexity: O(n + k)

    Args:
        arr: Input array to be sorted (non-negative integers)

    Returns:
        Sorted array in ascending order
    """
    if not arr:
        return arr

    max_value = max(arr)
    current_exp = 1
    while max_value // current_exp != 0:
        arr = counting_sort_single_digit(arr, current_exp)
        current_exp *= 10
    return arr


def counting_sort_single_digit(arr: List[int], exp: int):
    n = len(arr)
    output = [0] * n
    buckets = [0] * 10

    for i in range(n):
        # simple counting sort
        index = (arr[i] // exp) % 10
        buckets[index] += 1

    for i in range(1, 10):
        buckets[i] += buckets[i - 1]

    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[buckets[index] - 1] = arr[i]
        buckets[index] -= 1

    for i in range(n):
        arr[i] = output[i]

    return arr


# =============================================================================
# Sorting Algorithms Evaluation
# =============================================================================


def use_builtin_python():
    arrays = generate_random_array(size=30)
    print(sorted(arrays))

    # 可以在 key 这里传入一个函数（可以是 lamda 函数 也可以是内置或者自定义函数）
    # * 关键在根据这个函数的返回值进行排序 因此理论上也可以直接传入一个变量
    words = ["banana", "App", "cher"]
    print(sorted(words, key=str.lower))
    print(sorted(words, key=len))
    print(sorted(words, key=lambda x: len(x)))
    print(sorted(words, key=get_reverse_length))
    print(sorted(words, key=lambda x: -get_reverse_length(x)))


class SortEvaluator:
    """
    Comprehensive test framework for evaluating sorting algorithms.
    """

    def __init__(self):
        """Initialize the evaluator with all sorting algorithms to test."""
        self.algorithms = {
            "Merge Sort": merge_sort,
            "Direct Access Array Sort": direct_access_array_sort,
            "Counting Sort": counting_sort,
            "Radix Sort": radix_sort,
            "Quick Sort": quick_sort,
        }

    def verify_sorted(self, arr: List[int]) -> bool:
        """
        Verify if an array is sorted in ascending order.

        Args:
            arr: Array to verify

        Returns:
            True if sorted, False otherwise
        """
        return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))

    def test_algorithm(
        self, name: str, algorithm: Callable, test_cases: List[List[int]]
    ) -> dict:
        """
        Test a single sorting algorithm.

        Args:
            name: Name of the algorithm
            algorithm: Sorting function to test
            test_cases: List of test arrays

        Returns:
            Dictionary with test results
        """
        results = {
            "name": name,
            "total": len(test_cases),
            "passed": 0,
            "failed": 0,
            "errors": [],
        }

        for i, test_case in enumerate(test_cases):
            try:
                original = test_case.copy()
                sorted_result = algorithm(test_case)

                if not self.verify_sorted(sorted_result):
                    results["failed"] += 1
                    results["errors"].append(
                        f"Test case {i + 1}: Result not sorted. "
                        f"Input: {original[:10]}{'...' if len(original) > 10 else ''}, "
                        f"Output: {sorted_result[:10]}{'...' if len(sorted_result) > 10 else ''}"
                    )
                elif sorted(sorted_result) != sorted_result:
                    results["failed"] += 1
                    results["errors"].append(
                        f"Test case {i + 1}: Incorrect sorting. "
                        f"Expected: {sorted(original)[:10]}{'...' if len(original) > 10 else ''}, "
                        f"Got: {sorted_result[:10]}{'...' if len(sorted_result) > 10 else ''}"
                    )
                else:
                    results["passed"] += 1

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Test case {i + 1}: Exception - {str(e)}")

        return results

    def run_all_tests(self, test_cases: List[List[int]]) -> None:
        """
        Run tests for all implemented sorting algorithms.

        Args:
            test_cases: List of test arrays
        """
        print("=" * 70)
        print("Sorting Algorithm Evaluation Results")
        print("=" * 70)
        print(f"Total test cases: {len(test_cases)}\n")

        all_passed = True

        for name, algorithm in self.algorithms.items():
            results = self.test_algorithm(name, algorithm, test_cases)

            status = "PASS" if results["failed"] == 0 else "FAIL"
            print(f"{name:30} [{status}]")
            print(f"  Passed: {results['passed']}/{results['total']}")

            if results["errors"]:
                all_passed = False
                print("  Errors:")
                for error in results["errors"][:3]:  # Show first 3 errors
                    print(f"    - {error}")
                if len(results["errors"]) > 3:
                    print(f"    ... and {len(results['errors']) - 3} more errors")
            print()

        print("=" * 70)
        if all_passed:
            print("All tests passed!")
        else:
            print("Some tests failed. Please check the errors above.")
        print("=" * 70)

    def generate_test_cases(self) -> List[List[int]]:
        """
        Generate a comprehensive set of test cases.

        Returns:
            List of test arrays
        """
        test_cases = []

        # Edge cases
        test_cases.append([])  # Empty array
        test_cases.append([1])  # Single element
        test_cases.append([2, 1])  # Two elements

        # Small arrays
        test_cases.append(generate_random_array(10, 0, 50))
        test_cases.append(generate_random_array(20, 0, 100))

        # Already sorted
        test_cases.append(list(range(10)))

        # Reverse sorted
        test_cases.append(list(range(10, 0, -1)))

        # Arrays with duplicates
        test_cases.append([5, 3, 5, 1, 3, 5, 2, 1])

        # Larger arrays
        test_cases.append(generate_random_array(100, 0, 1000))
        test_cases.append(generate_random_array(500, 0, 10000))

        return test_cases

    def run_performance_test(
        self, algorithm: Callable, algorithm_name: str = None
    ) -> None:
        """
        Run a performance test on a single algorithm.

        Args:
            algorithm: Sorting function to test
            algorithm_name: Optional name for the algorithm
        """
        if algorithm_name is None:
            algorithm_name = algorithm.__name__

        test_sizes = [100, 1000, 10000, 50000]

        print(f"\n{algorithm_name} Performance Test:")
        print("-" * 50)
        print(f"{'Size':<15} {'Time (s)':<15}")

        for size in test_sizes:
            arr = generate_random_array(size, 0, 100000)

            start_time = time.time()
            try:
                result = algorithm(arr)
                end_time = time.time()
                elapsed = end_time - start_time

                # Verify correctness
                if self.verify_sorted(result):
                    print(f"{size:<15} {elapsed:<15.6f}")
                else:
                    print(f"{size:<15} Failed (incorrect sorting)")
            except Exception as e:
                print(f"{size:<15} Error: {str(e)}")


if __name__ == "__main__":
    # Demonstrate built-in sorting
    print("=== Built-in Python Sorting Demo ===")
    use_builtin_python()

    # Run comprehensive tests
    print("\n\n=== Running Sorting Algorithm Tests ===")
    evaluator = SortEvaluator()
    test_cases = evaluator.generate_test_cases()

    print("\nGenerated test cases:")
    for i, tc in enumerate(test_cases[:5], 1):
        print(f"  Case {i}: size={len(tc)}")
    print(f"  ... and {len(test_cases) - 5} more cases\n")

    # Run correctness tests
    evaluator.run_all_tests(test_cases)

    # Example: Run performance test on a single algorithm (uncomment to use)
    print("\n\n=== Performance Testing ===")
    evaluator.run_performance_test(merge_sort, "Merge Sort")
    evaluator.run_performance_test(quick_sort, "Quick Sort")
    evaluator.run_performance_test(counting_sort, "Counting Sort")
    evaluator.run_performance_test(radix_sort, "Radix Sort")
