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

class TestMaxCandies(unittest.TestCase):
    def setUp(self):
        self.solution = load_solution('3_maximum_candies_you_can_get_from_boxes.py')

    def test_example_1(self):
        status = [1,0,1,0]
        candies = [7,5,4,100]
        keys = [[],[],[1],[]]
        containedBoxes = [[1,2],[3],[],[]]
        initialBoxes = [0]
        self.assertEqual(self.solution.maxCandies(status, candies, keys, containedBoxes, initialBoxes), 16)

    def test_example_2(self):
        status = [1,0,0,0,0,0]
        candies = [1,1,1,1,1,1]
        keys = [[1,2,3,4,5],[],[],[],[],[]]
        containedBoxes = [[1,2,3,4,5],[],[],[],[],[]]
        initialBoxes = [0]
        self.assertEqual(self.solution.maxCandies(status, candies, keys, containedBoxes, initialBoxes), 6)

    def test_example_3(self):
        status = [1,1,1]
        candies = [100,1,100]
        keys = [[],[0,2],[]]
        containedBoxes = [[],[],[]]
        initialBoxes = [1]
        self.assertEqual(self.solution.maxCandies(status, candies, keys, containedBoxes, initialBoxes), 1)

    def test_example_4(self):
        status = [1]
        candies = [100]
        keys = [[]]
        containedBoxes = [[]]
        initialBoxes = []
        self.assertEqual(self.solution.maxCandies(status, candies, keys, containedBoxes, initialBoxes), 0)

    def test_example_5(self):
        status = [1,1,1]
        candies = [2,3,2]
        keys = [[],[],[]]
        containedBoxes = [[],[],[]]
        initialBoxes = [2,1,0]
        self.assertEqual(self.solution.maxCandies(status, candies, keys, containedBoxes, initialBoxes), 7)

    def test_nested_keys(self):
        # Box 0 has key to 1. Box 1 has key to 2.
        # Box 0 contains 1. Box 1 contains 2.
        # All initially closed except 0.
        status = [1, 0, 0]
        candies = [1, 2, 3]
        keys = [[1], [2], []]
        containedBoxes = [[1], [2], []]
        initialBoxes = [0]
        self.assertEqual(self.solution.maxCandies(status, candies, keys, containedBoxes, initialBoxes), 6)

    def test_key_before_box(self):
        # Box 0 has key to 1.
        # Box 0 contains 2.
        # Box 2 contains 1.
        # So we get key 1, then find box 2, then find box 1. Box 1 should open.
        status = [1, 0, 1]
        candies = [10, 20, 30]
        keys = [[1], [], []]
        containedBoxes = [[2], [], [1]]
        initialBoxes = [0]
        self.assertEqual(self.solution.maxCandies(status, candies, keys, containedBoxes, initialBoxes), 60)

    def test_large_chain(self):
        # 1000 boxes. Box i has key to i+1. All closed except 0.
        # Each box has 1 candy.
        n = 1000
        status = [0] * n
        status[0] = 1
        candies = [1] * n
        keys = [[i+1] if i < n-1 else [] for i in range(n)]
        containedBoxes = [[i+1] if i < n-1 else [] for i in range(n)]
        initialBoxes = [0]
        
        # We should be able to open all boxes
        self.assertEqual(self.solution.maxCandies(status, candies, keys, containedBoxes, initialBoxes), n)

if __name__ == '__main__':
    unittest.main()
