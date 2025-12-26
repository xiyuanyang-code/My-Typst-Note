from typing import List
import random


def quick_select(array: List[int], k: int):
    """
    Find the k-th smallest element in an unsorted array using Quick Select algorithm.

    Args:
        array: List of integers to search in.
        k: The position (1-indexed) of the element to find.

    Returns:
        The k-th smallest element, or None if k is out of range.
    """
    length = len(array)
    if length < k:
        return None
    if length == 1:
        return array[0]

    # select pivot randomly
    random_pivot = random.randint(0, length - 1)
    pivot_value = array[random_pivot]
    left_array = []
    right_array = []
    same_value_count = 0
    for value in array:
        if value < pivot_value:
            left_array.append(value)
        elif value > pivot_value:
            right_array.append(value)
        else:
            same_value_count += 1

    if len(left_array) >= k:
        return quick_select(left_array, k)
    elif len(left_array) < k and len(left_array) + same_value_count >= k:
        return pivot_value
    else:
        return quick_select(right_array, k - len(left_array) - same_value_count)


class QuickSelectTester:
    """Test suite for Quick Select algorithm."""

    @staticmethod
    def test_basic_functionality():
        """Test basic functionality with simple arrays."""
        test_cases = [
            ([7, 1, 5, 2, 2, 9], 1, 1),
            ([7, 1, 5, 2, 2, 9], 2, 2),
            ([7, 1, 5, 2, 2, 9], 3, 2),
            ([7, 1, 5, 2, 2, 9], 4, 5),
            ([7, 1, 5, 2, 2, 9], 5, 7),
            ([7, 1, 5, 2, 2, 9], 6, 9),
        ]

        for arr, k, expected in test_cases:
            result = quick_select(arr, k)
            assert result == expected, f"Failed: array={arr}, k={k}, expected={expected}, got={result}"
        print("Basic functionality tests passed!")

    @staticmethod
    def test_edge_cases():
        """Test edge cases including empty array and single element."""
        # Single element
        assert quick_select([42], 1) == 42, "Failed: single element"

        # k out of range
        assert quick_select([1, 2, 3], 5) is None, "Failed: k out of range"
        assert quick_select([], 1) is None, "Failed: empty array"

        print("Edge case tests passed!")

    @staticmethod
    def test_duplicates():
        """Test arrays with many duplicate elements."""
        test_cases = [
            ([5, 5, 5, 5, 5], 3, 5),
            ([1, 2, 2, 2, 3], 2, 2),
            ([1, 2, 2, 2, 3], 4, 2),
            ([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], 5, 3),
        ]

        for arr, k, expected in test_cases:
            result = quick_select(arr, k)
            assert result == expected, f"Failed: array={arr}, k={k}, expected={expected}, got={result}"
        print("Duplicate elements tests passed!")

    @staticmethod
    def test_correctness_verification():
        """Verify correctness by comparing with sorted array."""
        random.seed(42)  # For reproducibility

        for _ in range(100):
            # Generate random array
            arr = [random.randint(1, 100) for _ in range(random.randint(1, 50))]
            k = random.randint(1, len(arr))

            # Get result from quick_select
            result = quick_select(arr, k)

            # Get expected result from sorted array
            expected = sorted(arr)[k - 1]

            assert result == expected, f"Failed: arr={arr}, k={k}, expected={expected}, got={result}"

        print("Correctness verification tests passed!")

    @staticmethod
    def test_large_array():
        """Test with larger arrays for performance verification."""
        arr = list(range(1000, 0, -1))  # Reverse sorted array

        # Test various k values
        test_cases = [
            (1, 1),
            (500, 500),
            (1000, 1000),
            (100, 100),
        ]

        for k, expected in test_cases:
            result = quick_select(arr, k)
            assert result == expected, f"Failed: k={k}, expected={expected}, got={result}"

        print("Large array tests passed!")

    @staticmethod
    def run_all_tests():
        """Run all test suites."""
        print("=" * 50)
        print("Running Quick Select Test Suite")
        print("=" * 50)

        QuickSelectTester.test_basic_functionality()
        QuickSelectTester.test_edge_cases()
        QuickSelectTester.test_duplicates()
        QuickSelectTester.test_correctness_verification()
        QuickSelectTester.test_large_array()

        print("=" * 50)
        print("All tests passed successfully!")
        print("=" * 50)


if __name__ == "__main__":
    # Run comprehensive test suite
    QuickSelectTester.run_all_tests()