"""
AI1804 算法设计与分析 - 第5次课上练习
问题5-2：Dijkstra算法实现

请完成Dijkstra算法的实现
"""

import heapq


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
    # 步骤1：初始化
    # 提示：初始化所有顶点的距离和前驱信息，源点距离设为0，创建visited集合
    dist = {}
    parents = {}
    visited = set()
    for node, neighbors in graph.items():
        if node == start:
            dist[node] = 0
            parents[node] = None
        else:
            dist[node] = float("inf")
            parents[node] = None

    # 步骤2：使用优先队列（最小堆）
    # 提示：使用heapq模块维护未处理的顶点，按距离排序
    pq = []
    pq.append((0, start))
    heapq.heapify(pq)

    # 步骤3：主循环
    # 提示：每次取出距离最小的未处理顶点，更新其邻居的距离
    # 注意：需要检测负权边并抛出异常，避免重复处理已确定的顶点
    while pq:
        # TODO: 从优先队列取出距离最小的顶点，检查是否已处理，更新邻居距离
        current_distance, current_node = heapq.heappop(pq)
        if current_node in visited:
            continue
        visited.add(current_node)
        if current_distance > dist[current_node]:
            continue
        for neighbor in graph[current_node]:
            neighbor_weight = weights[(current_node, neighbor)]
            if neighbor_weight < 0:
                raise ValueError("Error, Negative Weights")
            if dist[current_node] + neighbor_weight < dist[neighbor]:
                # update neighbors
                dist[neighbor] = dist[current_node] + neighbor_weight
                parents[neighbor] = current_node
                heapq.heappush(pq, (dist[neighbor], neighbor))
        # 提示：
        # - 如果顶点已处理，跳过
        # - 标记顶点为已处理
        # - 遍历邻居，检测负权边，进行松弛操作，更新队列
        pass

    # 步骤4：返回结果
    # 提示：返回距离字典和前驱字典
    return dist, parents


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
    # it is a greedy algorithms
    dist = {}
    parents = {}
    visited = set()

    pq = []
    heapq.heapify(pq)
    # initialize
    for node, neighbors in graph.items():
        if node == start:
            dist[node] = 0
            parents[node] = None
        else:
            dist[node] = float("inf")
            parents[node] = None

    pq.append((0, start))
    
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        visited.add(current_node)

        for neighbor in graph[current_node]:
            neighbor_weight = weights[(current_node, neighbor)]
            if neighbor_weight < 0:
                raise ValueError("Error, Negative Weights")
            if neighbor not in visited:
                if dist[current_node] + neighbor_weight < dist[neighbor]:
                    # do relaxations
                    dist[neighbor] = dist[current_node] + neighbor_weight
                    parents[neighbor] = current_node
                    heapq.heappush(pq, (dist[neighbor], neighbor))
    return dist, parents


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("问题5-2：Dijkstra算法实现")
    print("=" * 60)

    # 测试1: 基本功能
    print("\n=== 测试1: 基本功能 ===")
    graph1 = {"A": ["B", "C"], "B": ["C", "D"], "C": ["D"], "D": []}
    weights1 = {
        ("A", "B"): 4,
        ("A", "C"): 2,
        ("B", "C"): 1,
        ("B", "D"): 5,
        ("C", "D"): 3,
    }

    dist_dijkstra, parents_dijkstra = dijkstra(graph1, weights1, "A")
    print(f"距离: {dist_dijkstra}")
    print(f"前驱: {parents_dijkstra}")

    assert dist_dijkstra["A"] == 0, f"期望0，实际{dist_dijkstra['A']}"
    assert dist_dijkstra["B"] == 4, f"期望4，实际{dist_dijkstra['B']}"
    assert dist_dijkstra["C"] == 2, f"期望2，实际{dist_dijkstra['C']}"
    assert dist_dijkstra["D"] == 5, f"期望5，实际{dist_dijkstra['D']}"

    print("✓ 基本功能测试通过")

    # 测试2: 负权边检测
    print("\n=== 测试2: 负权边检测 ===")
    graph2 = {"A": ["B", "C"], "B": ["D"], "C": ["B"], "D": []}
    weights2 = {("A", "B"): 1, ("A", "C"): 4, ("C", "B"): -2, ("B", "D"): 3}  # 负权边

    try:
        dijkstra(graph2, weights2, "A")
        print("✗ 应该检测到负权边并抛出异常")
        assert False, "应该抛出异常"
    except (ValueError, AssertionError) as e:
        print(f"✓ 正确检测到负权边: {e}")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
