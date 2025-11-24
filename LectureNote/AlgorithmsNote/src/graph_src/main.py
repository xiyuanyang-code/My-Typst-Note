from typing import List, Tuple
import heapq
from collections import deque


class Graph:
    def __init__(self, n: int, edges: List[List[int]]):
        self.n = n
        self.edges = [[] for _ in range(n)]
        for u, v, w in edges:
            self.add_edge(u, v, w)

    def add_edge(self, u: int, v: int, w: int):
        self.edges[u].append((v, w))

    # ----------------------------------------------------------------------
    # 1. Dijkstra（无负权图）
    # ----------------------------------------------------------------------
    def dijkstra(self, src: int, dst: int) -> int:
        dist = [float("inf")] * self.n
        dist[src] = 0
        heap = [(0, src)]

        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                return d
            if d > dist[u]:
                continue
            for v, w in self.edges[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return -1

    # ----------------------------------------------------------------------
    # 2. DAG 最短路径（拓扑排序 + DP）
    # 要求：图必须是 DAG
    # ----------------------------------------------------------------------
    def shortest_path_dag(self, src: int, dst: int) -> int:
        indeg = [0] * self.n
        for u in range(self.n):
            for v, _ in self.edges[u]:
                indeg[v] += 1

        # Kahn 拓扑排序
        q = deque([i for i in range(self.n) if indeg[i] == 0])
        topo = []

        while q:
            u = q.popleft()
            topo.append(u)
            for v, _ in self.edges[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        # 路径 DP
        dist = [float("inf")] * self.n
        dist[src] = 0
        for u in topo:
            for v, w in self.edges[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        return dist[dst] if dist[dst] < float("inf") else -1

    # ----------------------------------------------------------------------
    # 3. Bellman–Ford（支持负权边，可检测负环）
    # 返回：
    #   dist 或 -1（无路径）
    #   如果存在负权环，抛出异常
    # ----------------------------------------------------------------------
    def bellman_ford(self, src: int, dst: int) -> int:
        dist = [float("inf")] * self.n
        dist[src] = 0

        edges_flat = []
        for u in range(self.n):
            for v, w in self.edges[u]:
                edges_flat.append((u, v, w))

        # V-1 次松弛
        for _ in range(self.n - 1):
            updated = False
            for u, v, w in edges_flat:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    updated = True
            if not updated:
                break

        # 再检查一次是否有负环
        for u, v, w in edges_flat:
            if dist[u] + w < dist[v]:
                raise ValueError("Negative cycle detected")

        return dist[dst] if dist[dst] < float("inf") else -1


# ----------------------------------------------------------------------
# 测试样例
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 图 1：用于 Dijkstra
    edges1 = [
        [0, 1, 2],
        [0, 2, 4],
        [1, 2, 1],
        [1, 3, 7],
        [2, 3, 3]
    ]
    g1 = Graph(4, edges1)
    print("Dijkstra 0->3:", g1.dijkstra(0, 3))   # 最短路应为 2+1+3 = 6

    # 图 2：DAG 图
    edges2 = [
        [0, 1, 1],
        [0, 2, 2],
        [1, 3, 3],
        [2, 3, 1]
    ]
    g2 = Graph(4, edges2)
    print("DAG shortest 0->3:", g2.shortest_path_dag(0, 3))  # 最短路应为 2+1 = 3

    # 图 3：含负权但无负环（Bellman-Ford）
    edges3 = [
        [0, 1, 1],
        [1, 2, -2],
        [0, 2, 4]
    ]
    g3 = Graph(3, edges3)
    print("Bellman-Ford 0->2:", g3.bellman_ford(0, 2))  # 1 + (-2) = -1

    # 图 4：带负环（测试负环检测）
    edges4 = [
        [0, 1, 1],
        [1, 2, -1],
        [2, 1, -1]  # 1->2->1 构成负环
    ]
    g4 = Graph(3, edges4)
    try:
        print(g4.bellman_ford(0, 2))
    except ValueError as e:
        print("Bellman-Ford detect:", e)
