"""
AI1804 算法设计与分析 - 第5次课上练习
问题5-4：网络延迟时间

请完成网络延迟时间函数的实现
注意：可以直接调用问题5-2中实现的dijkstra函数
"""

# 导入问题5-2中实现的dijkstra函数
# 如果dijkstra函数在同一个文件中或已导入，可以直接使用
# 这里假设dijkstra函数已经实现，签名如下：
# def dijkstra(graph, weights, start) -> (dist, parents)

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


def network_delay_time(times, n, k):
    """
    计算网络延迟时间

    参数:
        times: 边列表，每个元素是 (u, v, w) 元组
        n: 节点数量（节点编号从1到n）
        k: 源节点

    返回:
        所有节点收到信号的时间，如果无法使所有节点收到信号则返回-1

    提示：可以调用dijkstra函数计算从k到所有节点的最短距离
    """
    graph = {}
    weights = {}
    for u, v, w in times:
        if u not in graph:
            graph[u] = []

        if v not in graph:
            graph[v] = []

        graph[u].append(v)
        weights[(u, v)] = w

    dist, parent = dijkstra(graph=graph, weights=weights, start=k)
    max_value = dist[1]
    for key, value in dist.items():
        if value == float("inf"):
            return -1
        max_value = max(value, max_value)

    return max_value


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("问题5-4：网络延迟时间")
    print("=" * 60)

    # 测试1: 基本功能
    print("\n=== 测试1: 基本功能 ===")
    times1 = [[2, 1, 1], [2, 3, 1], [3, 4, 1]]
    result1 = network_delay_time(times1, 4, 2)
    print(f"结果: {result1}")
    assert result1 == 2, f"期望2，实际{result1}"
    print("✓ 基本功能测试通过")

    # 测试2: 不可达节点
    print("\n=== 测试2: 不可达节点 ===")
    times2 = [[1, 2, 1]]
    result2 = network_delay_time(times2, 2, 2)
    print(f"结果: {result2}")
    assert result2 == -1, f"期望-1，实际{result2}"
    print("✓ 不可达节点测试通过")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
