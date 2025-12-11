from typing import List, Dict


def build_graph(n: int, flights: List[List[int]]):
    graph = {}
    weights = {}

    for flight in flights:
        u = flight[0]
        v = flight[1]
        w = flight[2]

        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)

        weights[(u, v)] = w

    return graph, weights


def min_path_within_k(n: int, flights: List[List[int]], src: int, dst: int, k: int):
    graph, weights = build_graph(n, flights)
    dist = {}
    parents = {}
    for node, neighbors in graph.items():
        if node == src:
            dist[node] = 0
            parents[src] = None
        else:
            dist[node] = float("inf")
            parents[node] = None

    for i in range(min(len(graph) - 1, k+1)):
        new_dist = dist.copy()
        for (u, v), w in weights.items():
            if dist[u] + w < new_dist[v]:
                new_dist[v] = dist[u] + w
        dist = new_dist


    if dist[dst] == float("inf"):
        return -1
    else:
        return dist[dst]


if __name__ == "__main__":
    import json
    # print(min_path_within_k(4,[[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]],0,3,1))
    n = int(input())
    flights = json.loads(input())
    src = int(input())
    dst = int(input())
    k = int(input())

    print(min_path_within_k(n=n, flights=flights, src=src, dst=dst, k=k))