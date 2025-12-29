from typing import Dict
from collections import deque


def bfs(graph: Dict, s: str, t: str):
    visited = set()
    parents = {}
    queue = deque([s])
    visited.add(s)
    parents[s] = None

    while queue:
        # pop element
        current_node = queue.popleft()
        visited.add(current_node)
        if current_node == t:
            # we find the path, now getting the final results
            final_path = []
            index = current_node
            while index:
                final_path.append(index)
                index = parents[index]
            final_path.reverse()
            return final_path, len(final_path) - 1

        neighbors = graph[current_node]
        for neighbor in neighbors:
            if neighbor not in visited:
                parents[neighbor] = current_node
                queue.append(neighbor)

    # we find no path
    return None, -1


def dfs(graph: Dict, s: str, t: str):
    visited = set()

    def dfs_recursive(current: str):
        if current == t:
            return [current]
        if current in visited:
            return None

        visited.add(current)

        # do for all neighbors
        for neighbor in graph[current]:
            if neighbor not in visited:
                appended_path = dfs_recursive(current=neighbor)
                if appended_path:
                    return [current] + appended_path
        return None

    final_length = dfs_recursive(current=s)
    if final_length:
        return final_length, len(final_length) - 1
    else:
        return None, -1


def connected_components(graph: dict):
    """
    找出所有连通分量

    参数:
        graph: 无向图，邻接表表示 {vertex: [neighbors]}

    返回:
        连通分量列表，每个分量是一个顶点列表

    时间复杂度: O(|V| + |E|)
    """
    pass


def topological_sort(graph: dict) -> list:
    """
    对有向无环图(DAG)进行拓扑排序

    参数:
        graph: 有向无环图，邻接表表示 {vertex: [neighbors]}

    返回:
        拓扑排序后的顶点列表，如果图包含环则返回None

    时间复杂度: O(|V| + |E|)
    """
    # TODO: 实现Kahn算法或DFS-based拓扑排序
    # 1. 计算所有顶点的入度
    # 2. 将入度为0的顶点加入队列
    # 3. 处理队列中的顶点并更新邻居的入度
    # 4. 检测是否存在环（处理顶点数是否等于总顶点数）
    pass


def has_cycle(graph: dict) -> bool:
    """
    检测有向图中是否存在环

    参数:
        graph: 有向图，邻接表表示 {vertex: [neighbors]}

    返回:
        如果存在环返回True，否则返回False

    时间复杂度: O(|V| + |E|)
    """
    # TODO: 实现基于DFS的环检测
    # 1. 维护visited集合和recursion_stack集合
    # 2. 对每个未访问顶点进行DFS
    # 3. 如果遇到在递归栈中的顶点，则存在环
    # 4. 或者通过拓扑排序检测（排序后顶点数是否等于总顶点数）
    pass


def dag_shortest_paths(graph: dict, weights: dict, start: str) -> tuple:
    """
    计算有向无环图(DAG)中的单源最短路径

    参数:
        graph: 有向无环图，邻接表表示 {vertex: [neighbors]}
        weights: 边权重字典 {(u, v): weight}
        start: 源顶点

    返回:
        (dist, parents) 元组
        - dist: 字典，dist[v]是从start到v的最短距离
        - parents: 字典，parents[v]是v的前驱顶点

    时间复杂度: O(|V| + |E|)
    """
    # TODO: 实现DAG松弛算法
    # 1. 对图进行拓扑排序
    # 2. 按照拓扑排序的顺序初始化距离
    # 3. 按拓扑顺序松弛所有边
    # 4. 返回距离和前驱数组
    pass


def bellman_ford(graph: dict, weights: dict, start):
    """
    实现Bellman-Ford算法

    参数:
        graph: 有向图，邻接表表示 {vertex: [neighbors]}
        weights: 边权重字典 {(u, v): weight}
        start: 源顶点

    返回:
        (dist, parents, has_negative_cycle) 元组
        - dist: 字典，dist[v]是从start到v的最短距离
        - parents: 字典，parents[v]是v的前驱顶点
        - has_negative_cycle: 布尔值，是否存在负权环

    时间复杂度: O(|V| * |E|)
    """
    pass


def dijkstra(graph: dict, weights: dict, start):
    """
    实现Dijkstra算法

    参数:
        graph: 有向图，邻接表表示 {vertex: [neighbors]}
        weights: 边权重字典 {(u, v): weight}，要求权重非负
        start: 源顶点

    返回:
        (dist, parents) 元组
        - dist: 字典，dist[v]是从start到v的最短距离
        - parents: 字典，parents[v]是v的前驱顶点

    时间复杂度: O((|V| + |E|) log |V|)
    """
    pass


def johnson(graph: dict, weights: dict) -> tuple:
    """
    计算有向图中所有顶点对之间的最短路径（Johnson算法）

    参数:
        graph: 有向图，邻接表表示 {vertex: [neighbors]}
        weights: 边权重字典 {(u, v): weight}，允许负权边但不能有负权环

    返回:
        (dist_matrix, has_negative_cycle) 元组
        - dist_matrix: 字典，dist_matrix[u][v]是从u到v的最短距离
        - has_negative_cycle: 布尔值，是否存在负权环

    时间复杂度: O(|V| * |E| + |V|^2 log |V|)
    """
    # TODO: 实现Johnson算法
    # 1. 添加虚拟顶点s，连接到所有其他顶点（边权重为0）
    # 2. 运行Bellman-Ford算法计算h值（从s到各点的最短距离）
    # 3. 如果检测到负权环，返回错误
    # 4. 重新赋权：w'(u, v) = w(u, v) + h[u] - h[v]
    # 5. 对每个顶点运行Dijkstra算法
    # 6. 恢复原始权重：d(u, v) = d'(u, v) - h[u] + h[v]
    pass


if __name__ == "__main__":
    # 测试1: 最短路径查找
    print("\n=== 测试1: 最短路径查找 ===")
    graph1 = {
        "Alice": ["Bob", "Charlie"],
        "Bob": ["Alice", "David"],
        "Charlie": ["Alice", "Eve"],
        "David": ["Bob"],
        "Eve": ["Charlie"],
        "Frank": ["Grace"],
        "Grace": ["Frank"],
        "Henry": [],
    }

    # 测试路径存在
    path1, length1 = bfs(graph1, "Alice", "Eve")
    print(f"Alice到Eve的最短路径: {path1}")
    print(f"路径长度: {length1}")
    assert path1 == ["Alice", "Charlie", "Eve"], f"路径不正确: {path1}"
    assert length1 == 2, f"期望长度2，实际{length1}"

    # 测试路径不存在
    path2, length2 = bfs(graph1, "Alice", "Frank")
    print(f"Alice到Frank的最短路径: {path2}")
    print(f"路径长度: {length2}")
    assert path2 is None, f"期望None，实际{path2}"
    assert length2 == -1, f"期望-1，实际{length2}"

    # 测试相同顶点
    path3, length3 = bfs(graph1, "Alice", "Alice")
    print(f"Alice到Alice的最短路径: {path3}")
    print(f"路径长度: {length3}")
    assert path3 == ["Alice"], f"期望['Alice']，实际{path3}"
    assert length3 == 0, f"期望0，实际{length3}"

    # 测试路径存在
    path1, length1 = dfs(graph1, "Alice", "Eve")
    print(f"Alice到Eve的最短路径: {path1}")
    print(f"路径长度: {length1}")
    assert path1 == ["Alice", "Charlie", "Eve"], f"路径不正确: {path1}"
    assert length1 == 2, f"期望长度2，实际{length1}"

    # 测试路径不存在
    path2, length2 = dfs(graph1, "Alice", "Frank")
    print(f"Alice到Frank的最短路径: {path2}")
    print(f"路径长度: {length2}")
    assert path2 is None, f"期望None，实际{path2}"
    assert length2 == -1, f"期望-1，实际{length2}"

    # 测试相同顶点
    path3, length3 = dfs(graph1, "Alice", "Alice")
    print(f"Alice到Alice的最短路径: {path3}")
    print(f"路径长度: {length3}")
    assert path3 == ["Alice"], f"期望['Alice']，实际{path3}"
    assert length3 == 0, f"期望0，实际{length3}"

    print("✓ 最短路径测试通过")
