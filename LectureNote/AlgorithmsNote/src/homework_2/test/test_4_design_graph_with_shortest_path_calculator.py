import unittest
import sys
import os
import importlib.util

def load_graph_class(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = 'solution' if os.environ.get('USE_SOLUTION') else 'problem'
    solution_path = os.path.join(current_dir, '..', target_dir, file_name)
    spec = importlib.util.spec_from_file_location("solution_module", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Graph

class TestGraph(unittest.TestCase):
    def setUp(self):
        self.Graph = load_graph_class('4_design_graph_with_shortest_path_calculator.py')

    def test_example_1(self):
        g = self.Graph(4, [[0, 2, 5], [0, 1, 2], [1, 2, 1], [3, 0, 3]])
        self.assertEqual(g.shortestPath(3, 2), 6)
        self.assertEqual(g.shortestPath(0, 3), -1)
        g.addEdge([1, 3, 4])
        self.assertEqual(g.shortestPath(0, 3), 6)

    def test_disconnected(self):
        g = self.Graph(3, [])
        self.assertEqual(g.shortestPath(0, 2), -1)
        g.addEdge([0, 1, 1])
        self.assertEqual(g.shortestPath(0, 2), -1)
        g.addEdge([1, 2, 1])
        self.assertEqual(g.shortestPath(0, 2), 2)

    def test_multiple_paths(self):
        # 0 -> 1 -> 3 (cost 10)
        # 0 -> 2 -> 3 (cost 5)
        g = self.Graph(4, [[0, 1, 5], [1, 3, 5], [0, 2, 2], [2, 3, 3]])
        self.assertEqual(g.shortestPath(0, 3), 5)
        
        # Add shortcut 0 -> 3 (cost 1)
        g.addEdge([0, 3, 1])
        self.assertEqual(g.shortestPath(0, 3), 1)

    def test_cycle(self):
        # 0 -> 1 -> 0 cycle
        g = self.Graph(2, [[0, 1, 1], [1, 0, 1]])
        self.assertEqual(g.shortestPath(0, 1), 1)
        self.assertEqual(g.shortestPath(1, 0), 1)

    def test_large_graph(self):
        # 100 nodes in a line: 0 -> 1 -> 2 ... -> 99
        # Cost 1 for each edge.
        n = 100
        edges = [[i, i+1, 1] for i in range(n-1)]
        g = self.Graph(n, edges)
        
        # Path from 0 to 99 should be 99
        self.assertEqual(g.shortestPath(0, 99), 99)
        
        # Add a shortcut 0 -> 99 with cost 50
        g.addEdge([0, 99, 50])
        self.assertEqual(g.shortestPath(0, 99), 50)

if __name__ == '__main__':
    unittest.main()
