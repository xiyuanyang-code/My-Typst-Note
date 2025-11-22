from typing import List
import heapq


class Graph(object):

    def __init__(self, n: int, edges: List[List[int]]):
        """
        :type n: int
        :type edges: List[List[int]]
        """
        self.nodes = n
        self.edges = [[] for _ in range(n)]
        for edge in edges:
            self.addEdge(edge=edge)

    def addEdge(self, edge: List[int]):
        """
        :type edge: List[int]
        :rtype: None
        """
        u, v, w = edge
        self.edges[u].append((v, w))

    def shortestPath(self, node1: int, node2: int):
        """
        :type node1: int
        :type node2: int
        :rtype: int
        """
        # using dijkstra
        dist = [float("inf")] * self.nodes
        dist[node1] = 0

        # initialize heaps
        heap = [(0, node1)]
        while heap:
            d, u = heapq.heappop(heap)
            # do relaxations
            if u == node2:
                return d
            
            for v, w in self.edges[u]:
                new_d = d + w
                if new_d < dist[v]:
                    dist[v] = new_d
                    heapq.heappush(heap, (new_d, v))
        return -1
        



# Your Graph object will be instantiated and called as such:
# obj = Graph(n, edges)
# obj.addEdge(edge)
# param_2 = obj.shortestPath(node1,node2)
