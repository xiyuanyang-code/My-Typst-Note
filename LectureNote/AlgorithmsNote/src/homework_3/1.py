from typing import List
from collections import deque, defaultdict

def check_sequence(nums: List[int], sequences: List[List[int]]) -> bool:
    n = len(nums)
    
    # 构建图
    graph = defaultdict(list)   # key -> list of neighbors
    in_degree = {num: 0 for num in nums}  # 每个节点的入度
    
    for seq in sequences:
        for i in range(len(seq) - 1):
            u, v = seq[i], seq[i + 1]
            if v not in graph[u]:
                graph[u].append(v)
                in_degree[v] += 1
    
    # 拓扑排序
    queue = deque([num for num in nums if in_degree[num] == 0])
    count = 0
    
    while queue:
        if len(queue) > 1:  # 多个入度为0的节点，序列不唯一
            return False
        current = queue.popleft()
        count += 1
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # 判断是否覆盖所有节点
    return count == n



if __name__ == "__main__":
    assert False == check_sequence([1, 2, 3], [[1, 2], [1, 3]])
    assert True == check_sequence([1, 2, 3], [[1, 2], [1, 3], [1, 2, 3]])
    assert True == check_sequence([1, 2, 3, 4, 5], [[1, 2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3], [1], [4], [5]])
    print("All test passed")