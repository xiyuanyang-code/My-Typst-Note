from collections import deque
from typing import List, Dict, Set, Optional


def shortest_path(graph: dict, s: str, t: str):
    visited: Set[str] = set()
    parents: Dict = {}
    distances: Dict = {}

    # initialize
    queue = deque([s])
    parents[s] = None
    distances[s] = 0
    visited.add(s)

    # adding and popping queue
    while queue:
        current_vertex = queue.popleft()
        # checking whether the current index is the target index
        if current_vertex == t:
            final_result = []
            index = current_vertex
            while index:
                final_result.append(index)
                index = parents[index]
            final_result.reverse()
            return (final_result, len(final_result) - 1)

        neighbors = graph.get(current_vertex, [])
        for neighbor in neighbors:
            if neighbor not in visited:
                distances[neighbor] = distances[current_vertex] + 1
                parents[neighbor] = current_vertex
                visited.add(neighbor)
                queue.append(neighbor)
    return (None, -1)


def connected_components(graph: dict):
    """
    找出所有连通分量

    参数:
        graph: 无向图，邻接表表示 {vertex: [neighbors]}

    返回:
        连通分量列表，每个分量是一个顶点列表

    时间复杂度: O(|V| + |E|)
    """
    visited: Set[str] = set()
    components = []
    for vertex, neighbors in graph.items():
        if vertex not in visited:
            # running bfs
            results = []
            queue = deque([vertex])
            results.append(vertex)
            visited.add(vertex)

            while queue:
                current_index = queue.popleft()
                # run neighbors
                neighbors = graph.get(current_index, [])
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
                        visited.add(neighbor)
                        results.append(neighbor)
            components.append(results)
    return components


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("问题4-1：社交网络中的最短路径与连通分量")
    print("=" * 60)

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
    path1, length1 = shortest_path(graph1, "Alice", "Eve")
    print(f"Alice到Eve的最短路径: {path1}")
    print(f"路径长度: {length1}")
    assert path1 == ["Alice", "Charlie", "Eve"], f"路径不正确: {path1}"
    assert length1 == 2, f"期望长度2，实际{length1}"

    # 测试路径不存在
    path2, length2 = shortest_path(graph1, "Alice", "Frank")
    print(f"Alice到Frank的最短路径: {path2}")
    print(f"路径长度: {length2}")
    assert path2 is None, f"期望None，实际{path2}"
    assert length2 == -1, f"期望-1，实际{length2}"

    # 测试相同顶点
    path3, length3 = shortest_path(graph1, "Alice", "Alice")
    print(f"Alice到Alice的最短路径: {path3}")
    print(f"路径长度: {length3}")
    assert path3 == ["Alice"], f"期望['Alice']，实际{path3}"
    assert length3 == 0, f"期望0，实际{length3}"

    print("✓ 最短路径测试通过")

    # 测试2: 连通分量分析
    print("\n=== 测试2: 连通分量分析 ===")
    components = connected_components(graph1)
    print(f"连通分量数量: {len(components)}")
    print("连通分量:")
    for i, comp in enumerate(components, 1):
        print(f"  分量{i}: {sorted(comp)}")

    # 验证结果
    assert len(components) == 3, f"期望3个连通分量，实际{len(components)}"
    all_vertices = set()
    for comp in components:
        all_vertices.update(comp)
    assert all_vertices == set(graph1.keys()), "顶点集合不匹配"

    print("✓ 连通分量测试通过")

    # 测试3: 边界情况
    print("\n=== 测试3: 边界情况 ===")
    # 空图
    empty_graph = {}
    path_empty, length_empty = shortest_path(empty_graph, "A", "B")
    assert path_empty is None and length_empty == -1
    assert connected_components(empty_graph) == []

    # 单顶点图
    single_graph = {"A": []}
    path_single, length_single = shortest_path(single_graph, "A", "A")
    assert path_single == ["A"] and length_single == 0
    components_single = connected_components(single_graph)
    assert len(components_single) == 1 and components_single[0] == ["A"]

    print("✓ 边界测试通过")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
