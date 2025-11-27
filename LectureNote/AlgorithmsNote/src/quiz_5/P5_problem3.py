"""
AI1804 算法设计与分析 - 第5次课上练习
问题5-3：矩阵中的最短路径

请完成矩阵最短路径函数的实现
"""

from typing import List


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
    # 步骤1：初始化
    # 提示：初始化所有顶点的距离和前驱信息，源点距离设为0
    dist = {}
    parents = {}
    for node, neighbors in graph.items():
        if node == start:
            dist[node] = 0
            parents[start] = None
        else:
            dist[node] = float("inf")
            parents[node] = None

    # 步骤2：进行|V|-1轮松弛
    # 提示：每轮遍历所有边进行松弛操作，记录前驱顶点以便路径重构
    for i in range(len(graph) - 1):
        updated = False

        for (u, v), weight in weights.items():
            if dist[u] + weight < dist[v]:
                updated = True
                dist[v] = dist[u] + weight
                parents[v] = u
        if not updated:
            break

    # 步骤3：检测负权环
    # 提示：再进行一轮松弛，如果仍有距离被更新，说明存在负权环
    has_negative_cycle = False
    for (u, v), weight in weights.items():
        if dist[u] + weight < dist[v]:
            has_negative_cycle = True
            break

    # 步骤4：返回结果
    # 提示：返回距离字典、前驱字典和负权环标志
    return dist, parents, has_negative_cycle


def add_neighbour(node: int, m: int, n: int, graph: dict, weights: dict, grid):
    if node not in graph:
        graph[node] = []

    i = node // n
    j = node % n

    if i - 1 >= 0:
        new_index = (i - 1) * n + j
        graph[node].append(new_index)
        weights[(node, new_index)] = grid[i - 1][j]
        if new_index not in graph:
            graph[new_index] = []
    if j - 1 >= 0:
        new_index = i * n + j - 1
        graph[node].append(new_index)
        weights[(node, new_index)] = grid[i][j - 1]
        if new_index not in graph:
            graph[new_index] = []
    if i + 1 < m:
        new_index = (i + 1) * n + j
        graph[node].append(new_index)
        weights[(node, new_index)] = grid[i + 1][j]
        if new_index not in graph:
            graph[new_index] = []
    if j + 1 < n:
        new_index = i * n + j + 1
        graph[node].append(new_index)
        weights[(node, new_index)] = grid[i][j + 1]
        if new_index not in graph:
            graph[new_index] = []
    return graph


def min_path_sum_dp(grid):
    """
    计算矩阵中从左上角到右下角的最短路径权重

    参数:
        grid: 二维列表，grid[i][j]表示位置(i,j)的权重（可能为负）

    返回:
        从(0,0)到(m-1,n-1)的最短路径权重
        如果不存在路径或存在负权环，返回-1
    """
    m = len(grid)
    n = len(grid[0])
    dp = [[float("inf")] * n for _ in range(m)]
    dp[0][0] = 0
    for i in range(m):
        for j in range(n):
            if i - 1 >= 0:
                dp[i][j] = min(dp[i][j], dp[i - 1][j] + grid[i][j])
            if j - 1 >= 0:
                dp[i][j] = min(dp[i][j], dp[i][j - 1] + grid[i][j])

    return dp[m - 1][n - 1]


def min_path_sum(grid: List[List[int]]):
    """
    计算矩阵中从左上角到右下角的最短路径权重

    参数:
        grid: 二维列表，grid[i][j]表示位置(i,j)的权重（可能为负）

    返回:
        从(0,0)到(m-1,n-1)的最短路径权重
        如果不存在路径或存在负权环，返回-1
    """
    # build the graph
    m = len(grid)
    n = len(grid[0])
    graphs = {}
    weights = {}

    # for index (i,j): transfer to node in the graph i*n + j
    for i in range(m):
        for j in range(n):
            weight = grid[i][j]
            add_neighbour(i * n + j, m, n, graphs, weights, grid)

    dist, parents, has_negative_cycle = bellman_ford(
        graph=graphs, weights=weights, start=0
    )
    if has_negative_cycle:
        return -1

    return dist[(m - 1) * n + (n - 1)]


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("问题5-3：矩阵中的最短路径")
    print("=" * 60)

    # 测试1: 基本功能
    print("\n=== 测试1: 基本功能 ===")
    grid1 = [[1, 3, 1], [1, 5, 1], [4, 2, 1]]
    result1 = min_path_sum(grid1)
    print(f"结果: {result1}")
    # 路径: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2)
    # 权重: 3 + 1 + 1 + 1 = 6（不包括起点(0,0)）
    assert result1 == 6, f"期望6，实际{result1}"
    print("✓ 基本功能测试通过")

    # 测试2: 简单情况
    print("\n=== 测试2: 简单情况 ===")
    grid2 = [[1, 2, 3], [4, 5, 6]]
    result2 = min_path_sum(grid2)
    print(f"结果: {result2}")
    # 路径: (0,0) -> (0,1) -> (0,2) -> (1,2)
    # 权重: 2 + 3 + 6 = 11（不包括起点(0,0)）
    assert result2 == 11, f"期望11，实际{result2}"
    print("✓ 简单情况测试通过")

    # 测试3: 单行单列
    print("\n=== 测试3: 单行单列 ===")
    grid3 = [[1, 2, 3, 4]]
    result3 = min_path_sum(grid3)
    print(f"结果: {result3}")
    # 路径: (0,0) -> (0,1) -> (0,2) -> (0,3)
    # 权重: 2 + 3 + 4 = 9（不包括起点(0,0)）
    assert result3 == 9, f"期望9，实际{result3}"
    print("✓ 单行单列测试通过")

    # 测试4: 包含负值
    print("\n=== 测试4: 包含负值 ===")
    grid4 = [[1, -1, 3], [2, -2, 1], [1, 1, 1]]
    result4 = min_path_sum(grid4)
    print(f"结果: {result4}")
    # 路径: (0,0) -> (0,1) -> (1,1) -> (1,2) -> (2,2)
    # 权重: -1 + (-2) + 1 + 1 = -1（不包括起点(0,0)）
    assert result4 == -1, f"期望-1，实际{result4}"
    print("✓ 包含负值测试通过")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
