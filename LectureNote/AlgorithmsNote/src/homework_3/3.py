from typing import List
import heapq


def get_minimal_path(start: List[int], target: List[int], specialRoads: List[List[int]]):
    # build graphs
    if start == target:
        return 0
    
    vertexes = []
    vertexes.append((start[0], start[1]))
    vertexes.append((target[0], target[1]))
    for x1i, y1i, x2i, y2i, costi in specialRoads:
        if (x1i, y1i) not in vertexes:
            vertexes.append((x1i, y1i))

        if (x2i, y2i) not in vertexes:
            vertexes.append((x2i, y2i))

    vertexes = list(set(vertexes))
    graph = {}
    weights = {}
    for vertex in vertexes:
        graph[vertex] = []
    
    for x1i, y1i, x2i, y2i, costi in specialRoads:
        graph[(x1i, y1i)].append((x2i, y2i))
        weights[((x1i, y1i),(x2i, y2i))] = costi

    for u in vertexes:
        for v in vertexes:
            if u == v:
                continue
            if v not in graph[u]:
                graph[u].append(v)
            if (u,v) not in weights:
                weights[(u,v)] = abs(u[0]-v[0]) + abs(u[1] - v[1])
            else:
                weights[(u,v)] = min( abs(u[0]-v[0]) + abs(u[1] - v[1]),weights[(u,v)] )

    n = len(vertexes)


    # run dijkstra
    dist = {}
    for vertex in vertexes:
        if vertex == (start[0], start[1]):
            dist[vertex] = 0
        else:
            dist[vertex] = float("inf")

    heap = [(0, (start[0], start[1]))]

    while heap:
        d, u = heapq.heappop(heap)
        print(f"Popping {u}")
        if u == (target[0], target[1]):
            return d
        if d > dist[u]:
            continue
        for v in graph[u]:
            w = weights[(u,v)]
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist[(target[0], target[1])]



if __name__  == "__main__":
    import json
    start = json.loads(input())
    target = json.loads(input())
    special_roads = json.loads(input())
    print(get_minimal_path(start=start, target=target, specialRoads=special_roads))

