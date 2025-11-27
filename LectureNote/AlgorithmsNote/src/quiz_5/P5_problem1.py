"""
AI1804 算法设计与分析 - 第5次课上练习
问题5-1：Bellman-Ford算法实现

请完成Bellman-Ford算法和路径重构函数的实现
"""


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


def reconstruct_path(parents, start, target):
    """
    从parents字典重构从start到target的路径

    参数:
        parents: 前驱字典
        start: 起始顶点
        target: 目标顶点

    返回:
        路径列表，如果不存在路径返回None
    """
    # 提示：从target开始沿着parents字典回溯到start，然后反转路径
    # 注意：需要验证路径的有效性（起点是否正确）
    path = []
    current = target

    while current is not None:
        # 提示：将当前顶点加入path，然后移动到前驱顶点
        path.append(current)
        current = parents[current]
        pass

    # 提示：反转path，检查path[0]是否等于start
    path.reverse()
    if path[0] == start:
        return path
    else:
        return None


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("问题5-1：Bellman-Ford算法实现")
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

    dist1, parents1, has_cycle1 = bellman_ford(graph1, weights1, "A")
    print(f"距离: {dist1}")
    print(f"前驱: {parents1}")
    print(f"负权环: {has_cycle1}")

    assert dist1["A"] == 0, f"期望0，实际{dist1['A']}"
    assert dist1["B"] == 4, f"期望4，实际{dist1['B']}"
    assert dist1["C"] == 2, f"期望2，实际{dist1['C']}"
    assert dist1["D"] == 5, f"期望5，实际{dist1['D']}"
    assert has_cycle1 == False, "不应该有负权环"

    # 测试路径重构
    path1 = reconstruct_path(parents1, "A", "D")
    print(f"A到D的路径: {path1}")
    assert path1 == ["A", "C", "D"], f"路径不正确: {path1}"

    print("✓ 基本功能测试通过")

    # 测试2: 负权边（无环）
    print("\n=== 测试2: 负权边（无环） ===")
    graph2 = {"A": ["B", "C"], "B": ["D"], "C": ["B"], "D": []}
    weights2 = {("A", "B"): 1, ("A", "C"): 4, ("C", "B"): -2, ("B", "D"): 3}  # 负权边

    dist2, parents2, has_cycle2 = bellman_ford(graph2, weights2, "A")
    print(f"距离: {dist2}")
    print(f"负权环: {has_cycle2}")

    # A->B: 1, A->C->B: 4+(-2)=2, 所以B的最短距离是1
    assert dist2["B"] == 1, f"期望1，实际{dist2['B']}"
    assert dist2["D"] == 4, f"期望4，实际{dist2['D']}"  # A->B->D: 1+3=4
    assert has_cycle2 == False, "不应该有负权环"

    print("✓ 负权边测试通过")

    # 测试3: 负权环检测
    print("\n=== 测试3: 负权环检测 ===")
    graph3 = {"A": ["B"], "B": ["C"], "C": ["A"]}
    weights3 = {("A", "B"): 1, ("B", "C"): 1, ("C", "A"): -3}  # 形成负权环

    dist3, parents3, has_cycle3 = bellman_ford(graph3, weights3, "A")
    print(f"负权环: {has_cycle3}")
    assert has_cycle3 == True, "应该检测到负权环"

    print("✓ 负权环检测测试通过")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
