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
