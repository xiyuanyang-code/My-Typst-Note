from typing import List
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
        pass

    return dist, parents


def build_graphs(n: int, edges: List[List[int]]):
    # all nodes
    weights = {}
    graphs = {}
    for u, v, w in edges:
        # current node
        if u not in graphs:
            graphs[u] = []
        if v not in graphs:
            graphs[v] = []
        graphs[u].append(v)
        weights[(u, v)] = w

        # upper level graph
        if u + n not in graphs:
            graphs[u + n] = []
        if v + n not in graphs:
            graphs[v + n] = []
        graphs[u + n].append(v + n)
        weights[(u + n, v + n)] = w

        # backward
        graphs[v].append(u+n)
        weights[(v, u+n)] = 2 * w

    return weights, graphs


def get_min_path(n: int, edges: List[List[int]]):
    weights, graphs = build_graphs(n, edges)
    dist, parents = dijkstra(graph=graphs, weights=weights, start=0)
    candidate = []
    if dist[n - 1] != -1:
        candidate.append(dist[n - 1])

    if dist[2 * n - 1] != -1:
        candidate.append(dist[2 * n - 1])

    if not candidate:
        return -1
    else:
        return min(candidate)
    

if __name__ == "__main__":
    # print(get_min_path(4,[[0,1,3],[3,1,1],[2,3,4],[0,2,2]]))
    import json
    n = int(input())
    edges = json.loads(input())
    print(get_min_path(n, edges))
