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

class TestWaterAndJug(unittest.TestCase):
    def setUp(self):
        self.solution = load_solution('2_water_and_jug_problem.py')

    def test_example_1(self):
        self.assertTrue(self.solution.canMeasureWater(3, 5, 4))

    def test_example_2(self):
        self.assertFalse(self.solution.canMeasureWater(2, 6, 5))

    def test_example_3(self):
        self.assertTrue(self.solution.canMeasureWater(1, 2, 3))

    def test_target_zero(self):
        self.assertTrue(self.solution.canMeasureWater(3, 5, 0))

    def test_target_too_large(self):
        self.assertFalse(self.solution.canMeasureWater(3, 5, 9))

    def test_zero_capacity(self):
        self.assertTrue(self.solution.canMeasureWater(0, 0, 0))
        self.assertFalse(self.solution.canMeasureWater(0, 2, 1)) # 1 not divisible by gcd(0,2)=2
        self.assertTrue(self.solution.canMeasureWater(0, 2, 2))

    def test_coprime(self):
        self.assertTrue(self.solution.canMeasureWater(3, 7, 5)) # gcd(3,7)=1, 5%1==0

    def test_non_coprime(self):
        self.assertTrue(self.solution.canMeasureWater(4, 6, 2)) # gcd(4,6)=2, 2%2==0
        self.assertFalse(self.solution.canMeasureWater(4, 6, 3)) # 3%2!=0

    def test_large_input(self):
        # Test with maximum constraints: x, y, target <= 1000
        # x = 1000, y = 999 (coprime), target = 1
        self.assertTrue(self.solution.canMeasureWater(1000, 999, 1))
        # x = 1000, y = 1000, target = 500 (fail)
        self.assertFalse(self.solution.canMeasureWater(1000, 1000, 500))

if __name__ == '__main__':
    unittest.main()
