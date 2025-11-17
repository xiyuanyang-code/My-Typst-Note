from collections import deque
from typing import List, Dict, Set, Optional


class Graph:
    """
    图的类定义，支持有向图和无向图
    """
    def __init__(self, vertices: int, directed: bool = False):
        """
        初始化图
        :param vertices: 顶点数量
        :param directed: 是否为有向图，默认为无向图
        """
        self.vertices = vertices
        self.directed = directed
        # 使用邻接表表示图
        self.adj_list: Dict[int, List[int]] = {i: [] for i in range(vertices)}
    
    def add_edge(self, u: int, v: int) -> None:
        """
        添加边
        :param u: 起始顶点
        :param v: 目标顶点
        """
        self.adj_list[u].append(v)
        if not self.directed:
            self.adj_list[v].append(u)
    
    def get_neighbors(self, vertex: int) -> List[int]:
        """
        获取顶点的邻居
        :param vertex: 指定顶点
        :return: 邻居顶点列表
        """
        return self.adj_list[vertex]
    
    def print_graph(self) -> None:
        """
        打印图的邻接表表示
        """
        for vertex in self.adj_list:
            print(f"顶点 {vertex}: {self.adj_list[vertex]}")


def bfs(graph: Graph, start_vertex: int) -> List[int]:
    """
    广度优先搜索 (BFS) 算法
    :param graph: 图对象
    :param start_vertex: 起始顶点
    :return: BFS 遍历结果列表
    """
    visited: Set[int] = set()
    queue = deque([start_vertex])
    result = []
    
    visited.add(start_vertex)
    
    while queue:
        current_vertex = queue.popleft()
        result.append(current_vertex)
        
        for neighbor in graph.get_neighbors(current_vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result


def dfs_recursive(graph: Graph, vertex: int, visited: Set[int], result: List[int]) -> None:
    """
    深度优先搜索 (DFS) 算法的递归实现
    :param graph: 图对象
    :param vertex: 当前顶点
    :param visited: 已访问顶点集合
    :param result: DFS 遍历结果列表
    """
    visited.add(vertex)
    result.append(vertex)
    
    for neighbor in graph.get_neighbors(vertex):
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited, result)


def dfs(graph: Graph, start_vertex: int) -> List[int]:
    """
    深度优先搜索 (DFS) 算法的入口函数
    :param graph: 图对象
    :param start_vertex: 起始顶点
    :return: DFS 遍历结果列表
    """
    visited: Set[int] = set()
    result = []
    dfs_recursive(graph, start_vertex, visited, result)
    return result


def dfs_iterative(graph: Graph, start_vertex: int) -> List[int]:
    """
    深度优先搜索 (DFS) 算法的迭代实现
    :param graph: 图对象
    :param start_vertex: 起始顶点
    :return: DFS 遍历结果列表
    """
    visited: Set[int] = set()
    stack = [start_vertex]
    result = []
    
    while stack:
        current_vertex = stack.pop()
        
        if current_vertex not in visited:
            visited.add(current_vertex)
            result.append(current_vertex)
            
            # 将邻居节点添加到栈中（逆序添加以保持顺序一致性）
            for neighbor in reversed(graph.get_neighbors(current_vertex)):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return result


# 示例用法
if __name__ == "__main__":
    # 创建一个包含 5 个顶点的无向图
    g = Graph(5, directed=False)
    
    # 添加边
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    g.add_edge(3, 4)
    
    print("图的邻接表表示:")
    g.print_graph()
    
    print(f"\n从顶点 0 开始的 BFS 遍历结果: {bfs(g, 0)}")
    print(f"从顶点 0 开始的 DFS 遍历结果: {dfs(g, 0)}")
    print(f"从顶点 0 开始的 DFS 迭代遍历结果: {dfs_iterative(g, 0)}")