from typing import Dict, Set
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
    visited = set()
    connected_components_list = []
    for node, neighbors in graph.items():
        if node in visited:
            continue
        else:
            # start a new round
            new_connected_part = []
            queue = deque([node])
            while queue:
                current_node = queue.popleft()
                new_connected_part.append(current_node)
                visited.add(current_node)
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            connected_components_list.append(new_connected_part)
    return connected_components_list


def topological_sort_dfs(graph: dict):
    """
    对有向图进行拓扑排序（DFS 实现）

    参数:
        graph: 有向图，邻接表表示 {vertex: [dependencies]}
               例如 {'A': ['B', 'C']} 表示 A 依赖于 B 和 C（B 和 C 是 A 的先修）

    返回:
        拓扑排序列表（先修在前），如果存在环则返回 None

    时间复杂度: O(|V| + |E|)
    """
    if not graph:
        return []

    visited = set()  # 已完全处理
    rec_stack = set()  # 递归栈：正在访问路径
    result = []  # 后序结果（逆序即为拓扑序）

    def dfs(vertex):
        visited.add(vertex)
        rec_stack.add(vertex)

        for dependency in graph.get(vertex, []):
            if dependency not in visited:
                if dfs(dependency):
                    return True
            elif dependency in rec_stack:
                # detecting loops
                return True

        rec_stack.remove(vertex)
        result.append(vertex)
        return False

    for vertex in list(graph.keys()):
        if vertex not in visited:
            if dfs(vertex):
                return None

    return result


def topological_sort_bfs(graph: dict) -> list:
    """
    对有向无环图(DAG)进行拓扑排序

    参数:
        graph: 有向无环图，邻接表表示 {vertex: [neighbors]}

    返回:
        拓扑排序后的顶点列表，如果图包含环则返回None

    时间复杂度: O(|V| + |E|)
    """
    in_degree = {}
    result = []
    queue = deque([])

    # counting all the in_degree of the models
    for node, neighbors in graph.items():
        if node not in in_degree:
            in_degree[node] = 0
        for neighbor in neighbors:
            if neighbor not in in_degree:
                in_degree[neighbor] = 0
            in_degree[neighbor] += 1

    for node, neighbors in graph.items():
        if in_degree[node] == 0:
            queue.append(node)

    while queue:
        current_node = queue.popleft()
        result.append(current_node)
        for neighbor in graph[current_node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # cyclic detections
    if len(result) != len(graph):
        return None

    return result


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
    # * remember the core principles of dag relaxations
    # $$dist(v) = \max_{(u, v) \in E} \{ dist(u) + w(u, v) \}$$
    topological_sort_result = topological_sort_bfs(graph=graph)
    dist = {}
    prev = {}
    # initialize dist
    for node, _ in graph.items():
        if node == start:
            dist[node] = 0
            prev[node] = None
        else:
            dist[node] = float("inf")

    if not topological_sort_result:
        print("This graph contains cyclic, skipped")

    for node in topological_sort_result:
        if dist[node] != float("inf"):
            # it is reachable
            # do relaxations
            for neighbor in graph[node]:
                if weights[(node, neighbor)] + dist[node] < dist[neighbor]:
                    dist[neighbor] = weights[(node, neighbor)] + dist[node]
                    prev[neighbor] = node 
                
    return dist, prev


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


def test_bfs():
    """测试广度优先搜索算法"""
    print("\n=== 测试 BFS ===")
    graph = {
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
    path, length = bfs(graph, "Alice", "Eve")
    print(f"Alice到Eve的最短路径: {path}")
    print(f"路径长度: {length}")
    assert path == ["Alice", "Charlie", "Eve"], f"路径不正确: {path}"
    assert length == 2, f"期望长度2，实际{length}"

    # 测试路径不存在
    path, length = bfs(graph, "Alice", "Frank")
    print(f"Alice到Frank的最短路径: {path}")
    print(f"路径长度: {length}")
    assert path is None, f"期望None，实际{path}"
    assert length == -1, f"期望-1，实际{length}"

    # 测试相同顶点
    path, length = bfs(graph, "Alice", "Alice")
    print(f"Alice到Alice的最短路径: {path}")
    print(f"路径长度: {length}")
    assert path == ["Alice"], f"期望['Alice']，实际{path}"
    assert length == 0, f"期望0，实际{length}"

    print("✓ BFS测试通过")


def test_dfs():
    """测试深度优先搜索算法"""
    print("\n=== 测试 DFS ===")
    graph = {
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
    path, length = dfs(graph, "Alice", "Eve")
    print(f"Alice到Eve的路径: {path}")
    print(f"路径长度: {length}")
    assert path == ["Alice", "Charlie", "Eve"], f"路径不正确: {path}"
    assert length == 2, f"期望长度2，实际{length}"

    # 测试路径不存在
    path, length = dfs(graph, "Alice", "Frank")
    print(f"Alice到Frank的路径: {path}")
    print(f"路径长度: {length}")
    assert path is None, f"期望None，实际{path}"
    assert length == -1, f"期望-1，实际{length}"

    # 测试相同顶点
    path, length = dfs(graph, "Alice", "Alice")
    print(f"Alice到Alice的路径: {path}")
    print(f"路径长度: {length}")
    assert path == ["Alice"], f"期望['Alice']，实际{path}"
    assert length == 0, f"期望0，实际{length}"

    print("✓ DFS测试通过")


def test_connected_components():
    """测试连通分量算法"""
    print("\n=== 测试连通分量 ===")
    graph = {
        "A": ["B", "C"],
        "B": ["A", "C"],
        "C": ["A", "B"],
        "D": ["E"],
        "E": ["D"],
        "F": ["G"],
        "G": ["F"],
        "H": [],
    }

    components = connected_components(graph)
    print(f"连通分量数量: {len(components)}")
    for i, component in enumerate(components, 1):
        print(f"  分量{i}: {sorted(component)}")

    # 验证连通分量数量
    assert len(components) == 4, f"期望4个连通分量，实际{len(components)}"

    # 验证每个分量的顶点
    component_sets = [set(c) for c in components]
    assert {"A", "B", "C"} in component_sets, "缺少分量 {A, B, C}"
    assert {"D", "E"} in component_sets, "缺少分量 {D, E}"
    assert {"F", "G"} in component_sets, "缺少分量 {F, G}"
    assert {"H"} in component_sets, "缺少分量 {H}"

    print("✓ 连通分量测试通过")


def test_topological_sort():
    """测试拓扑排序算法（待实现）"""
    print("\n=== 测试拓扑排序 ===")

    # 示例图（DAG）
    graph = {
        "A": ["C", "D"],
        "B": ["C", "D"],
        "C": ["E"],
        "D": ["E"],
        "E": [],
    }
    result = topological_sort_bfs(graph)
    print(f"拓扑排序结果: {result}")
    assert result is not None, "该图是DAG，应该有拓扑排序"
    assert len(result) == 5, f"期望5个顶点，实际{len(result)}"
    assert set(result) == set(graph.keys()), "拓扑排序应包含所有顶点"


def test_has_cycle():
    """测试环检测算法（待实现）"""
    print("\n=== 测试环检测 ===")

    # 有环图
    graph_with_cycle = {
        "A": ["B"],
        "B": ["C"],
        "C": ["A"],
    }

    # 无环图
    graph_without_cycle = {
        "A": ["B", "C"],
        "B": ["C"],
        "C": [],
    }

    assert topological_sort_bfs(graph_with_cycle) == None, "该图包含环"
    assert topological_sort_bfs(graph_without_cycle) != None, "该图不包含环"


def test_dag_shortest_paths():
    """测试DAG最短路径算法（待实现）"""
    print("\n=== 测试 DAG 最短路径 ===")

    graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": [],
    }

    weights = {
        ("A", "B"): 3,
        ("A", "C"): 6,
        ("B", "D"): 4,
        ("C", "D"): 2,
    }

    dist, parents = dag_shortest_paths(graph, weights, "A")
    print(f"最短距离: {dist}")
    print(f"前驱节点: {parents}")
    assert dist["D"] == 7, f"期望A到D的最短距离为7，实际{dist['D']}"


def test_bellman_ford():
    """测试Bellman-Ford算法（待实现）"""
    print("\n=== 测试 Bellman-Ford ===")
    print("⚠ 该函数尚未实现，跳过测试")

    graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": [],
    }

    weights = {
        ("A", "B"): 3,
        ("A", "C"): 6,
        ("B", "D"): 4,
        ("C", "D"): 2,
    }

    # TODO: 当函数实现后，取消以下注释
    # dist, parents, has_neg_cycle = bellman_ford(graph, weights, "A")
    # print(f"最短距离: {dist}")
    # print(f"前驱节点: {parents}")
    # print(f"存在负权环: {has_neg_cycle}")
    # assert has_neg_cycle == False, "该图不包含负权环"
    # assert dist["D"] == 7, f"期望A到D的最短距离为7，实际{dist['D']}"


def test_dijkstra():
    """测试Dijkstra算法（待实现）"""
    print("\n=== 测试 Dijkstra ===")
    print("⚠ 该函数尚未实现，跳过测试")

    graph = {
        "A": ["B", "C"],
        "B": ["A", "C", "D"],
        "C": ["A", "B", "D"],
        "D": ["B", "C"],
    }

    weights = {
        ("A", "B"): 1,
        ("A", "C"): 4,
        ("B", "C"): 2,
        ("B", "D"): 5,
        ("C", "D"): 1,
        ("B", "A"): 1,
        ("C", "A"): 4,
        ("C", "B"): 2,
        ("D", "B"): 5,
        ("D", "C"): 1,
    }

    # TODO: 当函数实现后，取消以下注释
    # dist, parents = dijkstra(graph, weights, "A")
    # print(f"最短距离: {dist}")
    # print(f"前驱节点: {parents}")
    # assert dist["D"] == 4, f"期望A到D的最短距离为4，实际{dist['D']}"


def test_johnson():
    """测试Johnson算法（待实现）"""
    print("\n=== 测试 Johnson ===")
    print("⚠ 该函数尚未实现，跳过测试")

    graph = {
        "A": ["B"],
        "B": ["C"],
        "C": [],
    }

    weights = {
        ("A", "B"): 3,
        ("B", "C"): 4,
    }

    # TODO: 当函数实现后，取消以下注释
    # dist_matrix, has_neg_cycle = johnson(graph, weights)
    # print(f"距离矩阵: {dist_matrix}")
    # print(f"存在负权环: {has_neg_cycle}")
    # assert has_neg_cycle == False, "该图不包含负权环"
    # assert dist_matrix["A"]["C"] == 7, f"期望A到C的最短距离为7"


if __name__ == "__main__":
    print("=" * 50)
    print("图算法测试套件")
    print("=" * 50)

    # 运行所有测试
    test_bfs()
    test_dfs()
    test_connected_components()
    test_topological_sort()
    test_has_cycle()
    test_dag_shortest_paths()
    test_bellman_ford()
    test_dijkstra()
    test_johnson()

    print("\n" + "=" * 50)
    print("所有测试完成！")
    print("=" * 50)
