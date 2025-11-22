import unittest
import sys
import os
import importlib.util

def load_solution(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = 'solution' if os.environ.get('USE_SOLUTION') else 'problem'
    solution_path = os.path.join(current_dir, '..', target_dir, file_name)
    spec = importlib.util.spec_from_file_location("solution_module", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Solution()

class TestCoinChange(unittest.TestCase):
    def setUp(self):
        self.solution = load_solution('1_coin_change.py')

    def test_example_1(self):
        coins = [1, 2, 5]
        amount = 11
        self.assertEqual(self.solution.coinChange(coins, amount), 3)

    def test_example_2(self):
        coins = [2]
        amount = 3
        self.assertEqual(self.solution.coinChange(coins, amount), -1)

    def test_example_3(self):
        coins = [1]
        amount = 0
        self.assertEqual(self.solution.coinChange(coins, amount), 0)

    def test_large_coins(self):
        coins = [186, 419, 83, 408]
        amount = 6249
        # 6249 = 419 * 14 + 83 * 4 + ... actually DP finds optimal
        # Just checking it runs and returns valid result (>=0 or -1)
        # Expected result from LeetCode for this input is 20
        self.assertEqual(self.solution.coinChange(coins, amount), 20)

    def test_impossible_combination(self):
        coins = [5, 7]
        amount = 3
        self.assertEqual(self.solution.coinChange(coins, amount), -1)

    def test_single_coin_match(self):
        coins = [5]
        amount = 5
        self.assertEqual(self.solution.coinChange(coins, amount), 1)

    def test_single_coin_multiple(self):
        coins = [5]
        amount = 10
        self.assertEqual(self.solution.coinChange(coins, amount), 2)

    def test_large_input(self):
        # Test with maximum constraints: amount = 10000
        coins = [1, 2, 5]
        amount = 10000
        # 10000 / 5 = 2000 coins
        self.assertEqual(self.solution.coinChange(coins, amount), 2000)

if __name__ == '__main__':
    unittest.main()
