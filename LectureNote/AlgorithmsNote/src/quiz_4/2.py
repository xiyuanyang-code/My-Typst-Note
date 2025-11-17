"""
AI1804 算法设计与分析 - 第4次课上练习
问题4-2：课程依赖关系与拓扑排序

请完成以下两个任务的实现
"""

from collections import deque
from typing import Set


def dfs_recursive(graph: dict, vertex, visited: Set, result: list):
    visited.add(vertex)
    result.append(vertex)
    neighbors = graph.get(vertex, [])
    for neighbor in neighbors:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited, result)


def dfs(graph: dict, starting_index):
    visited: Set = set()
    result = []
    dfs_recursive(graph, starting_index, visited, result)
    return result

def topological_sort_dfs(graph: dict):
    """
    对有向图进行拓扑排序（DFS 实现）

    参数:
        graph: 有向图，邻接表表示 {vertex: [dependencies]}
               例如 {'A': ['B', 'C']} 表示 A 依赖于 B 和 C（B 和 C 是 A 的先修）

    返回:
        拓扑排序列表（先修在前），如果存在环则返回 None

    时间复杂度: O(|V| + |E|)
    """
    if not graph:
        return []

    visited = set()      # 已完全处理
    rec_stack = set()    # 递归栈：正在访问路径
    result = []          # 后序结果（逆序即为拓扑序）

    def dfs(vertex):
        visited.add(vertex)
        rec_stack.add(vertex)

        for dependency in graph.get(vertex, []):
            if dependency not in visited:
                if dfs(dependency):
                    return True
            elif dependency in rec_stack:
                # detecting loops
                return True 

        rec_stack.remove(vertex)
        result.append(vertex)
        return False

    for vertex in list(graph.keys()):
        if vertex not in visited:
            if dfs(vertex):
                return None
            
    return result

def topological_sort_bfs(graph: dict):
    """
    对有向图进行拓扑排序

    参数:
        graph: 有向图，邻接表表示 {vertex: [dependencies]}
               例如 {'A': ['B', 'C']} 表示A依赖于B和C（B和C是A的先修）

    返回:
        拓扑排序列表，如果存在环则返回None

    时间复杂度: O(|V| + |E|)
    """
    # counting the indegrees
    in_degree = {}
    neighbors = {}
    result = []
    queue = deque([])
    for vertex, dependencies in graph.items():
        in_degree[vertex] = len(dependencies)
        if in_degree[vertex] == 0:
            queue.append(vertex)

    # build neighbors
    for vertex, dependencies in graph.items():
        for depency in dependencies:
            if depency not in neighbors:
                neighbors[depency] = [vertex]
            else:
                neighbors[depency].append(vertex)

    while queue:
        current_item = queue.popleft()
        result.append(current_item)
        neighbor_courses = neighbors.get(current_item, [])
        for course in neighbor_courses:
            in_degree[course] -= 1
            if in_degree[course] == 0:
                queue.append(course)
    
    return result if len(result) == len(graph) else None


# ========== 任务2：课程学习计划 ==========


def course_plan(prerequisites: dict, target_courses: list):
    """
    找出学习目标课程所需的所有课程及其拓扑排序

    参数:
        prerequisites: 课程依赖关系 {course: [prerequisite_courses]}
        target_courses: 目标课程列表

    返回:
        包含所有必需课程的拓扑排序列表，如果存在环则返回None

    时间复杂度: O(|V| + |E|)
    """
    sub_graph = dict()
    queue = deque([])
    for target_course in target_courses:
        queue.append(target_course)

    while queue:
        current_course = queue.popleft()
        sub_graph[current_course] = prerequisites[current_course]
        for neighbour in prerequisites[current_course]:
            if neighbour not in sub_graph:
                queue.append(neighbour)

    return topological_sort(sub_graph)


# ========== 测试代码 ==========

if __name__ == "__main__":
    for i in range(2):
        if i == 0:
            print("\n\nUsing BFS Versions of Topological Sort")
            topological_sort = topological_sort_bfs
        elif i == 1:
            print("\n\nUsing DFS Versions of Topological Sort")
            topological_sort = topological_sort_dfs

        print("=" * 60)
        print("问题4-2：课程依赖关系与拓扑排序")
        print("=" * 60)

        # 测试1: 拓扑排序（无环）
        print("\n=== 测试1: 拓扑排序（无环） ===")
        prerequisites1 = {
            "数据结构": ["程序设计基础"],
            "算法设计": ["数据结构", "离散数学"],
            "操作系统": ["数据结构", "计算机组成原理"],
            "编译原理": ["数据结构", "算法设计"],
            "数据库": ["数据结构"],
            "程序设计基础": [],
            "离散数学": [],
            "计算机组成原理": [],
        }

        result1 = topological_sort(prerequisites1)
        print(f"拓扑排序结果: {result1}")

        # 验证：检查每条边的顺序是否正确
        if result1:
            pos = {course: i for i, course in enumerate(result1)}
            for course, deps in prerequisites1.items():
                for dep in deps:
                    assert pos[dep] < pos[course], f"错误：{dep}应该在{course}之前"
            print("✓ 拓扑排序验证通过")
        else:
            print("✗ 拓扑排序失败（检测到环）")

        # 测试2: 检测环
        print("\n=== 测试2: 检测环 ===")
        prerequisites2 = {"A": ["B"], "B": ["C"], "C": ["A"]}  # 形成环：A -> B -> C -> A

        result2 = topological_sort(prerequisites2)
        print(f"拓扑排序结果: {result2}")
        assert result2 is None, "应该检测到环"
        print("✓ 环检测测试通过")

        # 测试3: 课程学习计划
        print("\n=== 测试3: 课程学习计划 ===")
        target = ["编译原理", "操作系统"]
        plan = course_plan(prerequisites1, target)
        print(f"学习计划: {plan}")

        # 验证：目标课程应该在计划中
        if plan:
            assert "编译原理" in plan
            assert "操作系统" in plan
            # 验证先修课程也在计划中
            assert "数据结构" in plan
            assert "程序设计基础" in plan
            assert "算法设计" in plan
            assert "计算机组成原理" in plan

            # 验证拓扑顺序
            pos = {course: i for i, course in enumerate(plan)}
            for course in plan:
                if course in prerequisites1:
                    for dep in prerequisites1[course]:
                        if dep in plan:
                            assert pos[dep] < pos[course], f"错误：{dep}应该在{course}之前"

            print("✓ 课程学习计划验证通过")
        else:
            print("✗ 课程学习计划失败（检测到环）")

        # 测试4: 边界情况
        print("\n=== 测试4: 边界情况 ===")
        # 空图
        empty_graph = {}
        assert topological_sort(empty_graph) == []

        # 单顶点图
        single_graph = {"A": []}
        result_single = topological_sort(single_graph)
        assert result_single == ["A"]

        # 无依赖的多个课程
        independent = {"A": [], "B": [], "C": []}
        result_indep = topological_sort(independent)
        assert len(result_indep) == 3
        assert set(result_indep) == {"A", "B", "C"}

        print("✓ 边界测试通过")

        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
