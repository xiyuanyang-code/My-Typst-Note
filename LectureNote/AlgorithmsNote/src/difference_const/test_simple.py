"""
简单测试：验证 Bellman-Ford 算法实现
"""
import math


def bellman_ford(nodes, edges, source):
    """Bellman-Ford 算法"""
    dist = {v: math.inf for v in nodes}
    dist[source] = 0.0
    n = len(nodes)

    # 进行 n-1 次迭代
    for i in range(n - 1):
        updated = False
        for (u, v, w) in edges:
            if dist[u] != math.inf and dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            print(f"在第 {i+1} 次迭代后收敛")
            break

    # 检测负环
    for (u, v, w) in edges:
        if dist[u] != math.inf and dist[v] > dist[u] + w:
            print(f"检测到负环：节点 {u} 到节点 {v}，当前距离 {dist[v]} > {dist[u]} + {w}")
            return dist, True  # 存在负环

    return dist, False


# 简单测试案例：x1 - x0 <= 1, x2 - x0 <= 2, x2 - x1 <= 1
# 应该有解: x0=0, x1=1, x2=2
print("=== 简单测试案例 ===")
nodes = [0, 1, 2]
edges = [
    (0, 1, 1),  # x1 - x0 <= 1  => 边 0->1, 权重 1
    (0, 2, 2),  # x2 - x0 <= 2  => 边 0->2, 权重 2
    (1, 2, 1),  # x2 - x1 <= 1  => 边 1->2, 权重 1
]

dist, has_neg_cycle = bellman_ford(nodes, edges, 0)
if not has_neg_cycle:
    print("✅ 简单测试通过，解为：")
    for node in nodes:
        print(f"x{node} <= {dist[node]}")
else:
    print("❌ 简单测试失败")

# 现在测试原始问题
print("\n=== 原始差分约束问题 ===")
nodes = [0, 1, 2, 3, 4]  # 0 是虚拟源点
s = 0

locations = {1: "A", 2: "B", 3: "C", 4: "D"}

# 旅行时间 (分钟)
travel = {
    (1, 2): 25, (1, 3): 35, (1, 4): 50,
    (2, 1): 25, (2, 3): 20, (2, 4): 45,
    (3, 1): 35, (3, 2): 20, (3, 4): 30,
    (4, 1): 50, (4, 2): 45, (4, 3): 30,
}

# 可行的时间窗
time_windows = {
    1: (30, 150),
    2: (20, 180),
    3: (80, 220),
    4: (10, 260),
}

# 构建边
edges = []

# 旅行约束：x_j >= x_i + t => x_i - x_j <= -t => 边 j->i 权重 -t
for (i, j), t in travel.items():
    edges.append((j, i, -t))
    print(f"添加旅行约束边: {j} -> {i}, 权重 {-t} (对应 x_{i} - x_{j} <= {-t})")

# 时间窗约束：加入虚拟源点
for i, (a, b) in time_windows.items():
    # 上界: x_i <= b => x_i - x_s <= b => 边 s->i 权重 b
    edges.append((s, i, b))
    print(f"添加上界约束边: {s} -> {i}, 权重 {b} (对应 x_{i} - x_s <= {b})")
    # 下界: x_i >= a => x_s - x_i <= -a => 边 i->s 权重 -a
    edges.append((i, s, -a))
    print(f"添加下界约束边: {i} -> {s}, 权重 {-a} (对应 x_s - x_i <= {-a})")

print(f"\n总边数: {len(edges)}")

dist, has_neg_cycle = bellman_ford(nodes, edges, s)

if has_neg_cycle:
    print("❌ 系统无解（存在负权环）")
else:
    print("✅ 系统可行。以下是一组满足约束的到达时间（分钟）:")
    for i in sorted(locations):
        print(f"  {locations[i]}: x_{i} = {dist[i]:.1f}  (允许区间 {time_windows[i]})")

    # 检查每条旅行约束松弛情况
    print("\n旅行约束检查 (x_j >= x_i + t)：")
    for (i, j), t in travel.items():
        lhs = dist[j]
        rhs = dist[i] + t
        slack = lhs - rhs
        print(f"  {locations[i]} -> {locations[j]}: "
              f"{lhs:.1f} ≥ {rhs:.1f}  (slack={slack:+.1f})")