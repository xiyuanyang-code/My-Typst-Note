"""
AI1804 算法设计与分析
第3次课上练习：哈希表与优先队列

问题3-2：K路归并与去重

请在TODO标记处填写代码
"""

import heapq
import random

def merge_k_unsorted_lists_top_n(lists, n):
    if not lists or n <= 0:
        return []

    # 第一阶段：找到每个ID的全局最大优先级 O(N)
    max_priority = {}
    for lst in lists:
        for item_id, priority in lst:
            if item_id not in max_priority or priority > max_priority[item_id]:
                max_priority[item_id] = priority

    negated_lists = []
    # 建队 O(N)
    for list_index, lst in enumerate(lists):
        if lst:
            negated_list = [(-priority, item_id) for item_id, priority in lst]
            heapq.heapify(negated_list)
            negated_lists.append(negated_list)
        else:
            negated_lists.append([])

    # 初始化全局最小堆 O(K log N)
    # 从每个列表中找到第一个是全局最大优先级的元素
    global_heap = []
    heapq.heapify(global_heap)
    for list_index in range(len(lists)):
        while negated_lists[list_index]:
            negated_priority, item_id = heapq.heappop(negated_lists[list_index])
            actual_priority = -negated_priority
            # 只有当这个元素的优先级等于全局最大时才加入堆
            if actual_priority == max_priority[item_id]:
                heapq.heappush(global_heap, (negated_priority, item_id, list_index))
                break

    result = []
    added_ids = set()  # 跟踪已添加的ID，防止重复

    # 寻找新的最大元素 O(n log N)
    while global_heap and len(result) < n:
        negated_priority, item_id, list_index = heapq.heappop(global_heap)
        actual_priority = -negated_priority

        # 确保不重复添加
        if item_id not in added_ids:
            result.append((item_id, actual_priority))
            added_ids.add(item_id)

        # 从同一个列表中补充新的全局最大元素
        while negated_lists[list_index]:
            negated_priority, item_id = heapq.heappop(negated_lists[list_index])
            actual_priority = -negated_priority
            if actual_priority == max_priority[item_id]:
                heapq.heappush(global_heap, (negated_priority, item_id, list_index))
                break

    return result

    # 总时间复杂度：O(N log N)

# ========== 测试代码 ==========

def verify_result(lists, n, result, test_name):
    """
    验证测试结果的正确性.

    Args:
        lists: 输入列表
        n: 请求的元素数量
        result: 函数返回结果
        test_name: 测试名称
    """
    # 构建所有item的最大优先级字典
    max_priority = {}
    for lst in lists:
        for item_id, priority in lst:
            if item_id not in max_priority or priority > max_priority[item_id]:
                max_priority[item_id] = priority

    # 验证结果数量
    if n <= 0:
        expected_len = 0
    else:
        expected_len = min(n, len(max_priority))

    if len(result) != expected_len:
        raise AssertionError(f"{test_name}: 结果长度应为{expected_len}, 得到{len(result)}")

    # 如果结果为空，直接返回
    if len(result) == 0:
        return

    # 验证结果按优先级降序排列
    for i in range(len(result) - 1):
        if result[i][1] < result[i+1][1]:
            raise AssertionError(f"{test_name}: 结果未按降序排列: {result[i]} < {result[i+1]}")

    # 验证每个item的优先级是最大值
    for item_id, priority in result:
        if item_id not in max_priority:
            raise AssertionError(f"{test_name}: 未知item: {item_id}")
        if priority != max_priority[item_id]:
            raise AssertionError(f"{test_name}: {item_id}的优先级应为{max_priority[item_id]}, 得到{priority}")

    # 验证无重复
    result_ids = [item[0] for item in result]
    if len(result_ids) != len(set(result_ids)):
        raise AssertionError(f"{test_name}: 结果存在重复: {result_ids}")


def run_test(test_name, lists, n, expected_check=None):
    """
    运行单个测试用例.

    Args:
        test_name: 测试名称
        lists: 输入列表
        n: 请求的元素数量
        expected_check: 额外验证函数
    """
    try:
        result = merge_k_unsorted_lists_top_n(lists, n)
        verify_result(lists, n, result, test_name)

        if expected_check:
            if not expected_check(result):
                raise AssertionError(f"额外验证失败")

        print(f"  ✓ {test_name}通过")
        return True
    except Exception as e:
        print(f"  ✗ {test_name}失败: {e}")
        return False


def generate_random_test(k, max_list_size, id_pool_size):
    """
    生成随机测试用例.

    Args:
        k: 列表数量
        max_list_size: 每个列表最大元素数
        id_pool_size: ID池大小

    Returns:
        (lists, n) 测试输入
    """
    id_pool = [f"ID_{i}" for i in range(id_pool_size)]

    lists = []
    for _ in range(k):
        list_size = random.randint(0, max_list_size)
        current_list = []
        for _ in range(list_size):
            item_id = random.choice(id_pool)
            priority = random.randint(-100, 1000)
            current_list.append((item_id, priority))
        random.shuffle(current_list)
        lists.append(current_list)

    n = random.randint(0, id_pool_size)
    return lists, n


if __name__ == "__main__":
    print("=" * 60)
    print("问题3-2：K路归并与去重 - 自动化测试套件")
    print("=" * 60)

    passed = 0
    total = 0

    # ========== 边界情况测试 ==========
    print("\n[1/3] 边界情况测试")
    print("-" * 60)

    total += 1
    print(f"\n测试1: 空列表输入")
    if run_test("空列表", [], 5, lambda r: r == []):
        passed += 1

    total += 1
    print(f"\n测试2: n=0")
    if run_test("n=0", [[("A", 10)], [("B", 20)]], 0, lambda r: r == []):
        passed += 1

    total += 1
    print(f"\n测试3: n<0")
    if run_test("n<0", [[("A", 10)], [("B", 20)]], -1, lambda r: r == []):
        passed += 1

    total += 1
    print(f"\n测试4: 所有空列表")
    if run_test("所有空列表", [[], [], []], 5, lambda r: r == []):
        passed += 1

    total += 1
    print(f"\n测试5: 单个列表")
    lists5 = [[("A", 10), ("B", 30), ("C", 20)]]
    if run_test("单个列表", lists5, 2):
        passed += 1

    total += 1
    print(f"\n测试6: 单个元素")
    lists6 = [[("A", 100)]]
    if run_test("单个元素", lists6, 5, lambda r: r == [("A", 100)]):
        passed += 1

    total += 1
    print(f"\n测试7: 相同ID不同优先级（取最大）")
    lists7 = [
        [("X", 10)], [("X", 50)], [("X", 30)], [("X", 100)]
    ]
    if run_test("相同ID取最大", lists7, 1, lambda r: r == [("X", 100)]):
        passed += 1

    total += 1
    print(f"\n测试8: 负数优先级")
    lists8 = [[("A", -10), ("B", 0)], [("C", -5)]]
    if run_test("负数优先级", lists8, 3):
        passed += 1

    total += 1
    print(f"\n测试9: 相同优先级")
    lists9 = [[("A", 50), ("B", 50)], [("C", 50)]]
    if run_test("相同优先级", lists9, 3, lambda r: len(r) == 3):
        passed += 1

    # ========== 基础功能测试 ==========
    print("\n[2/3] 基础功能测试")
    print("-" * 60)

    total += 1
    print(f"\n测试10: K个未排序列表Top-3")
    lists10 = [
        [("A", 100), ("C", 60), ("B", 80)],
        [("E", 50), ("A", 90), ("D", 85)],
        [("F", 40), ("B", 95), ("C", 70)]
    ]
    if run_test("K路归并Top-3", lists10, 3):
        passed += 1

    total += 1
    print(f"\n测试11: 键冲突保留最大值")
    lists11 = [
        [("X", 50), ("Y", 30)],
        [("X", 100), ("Z", 20)],
        [("X", 30), ("W", 40)]
    ]
    if run_test("键冲突", lists11, 2, lambda r: r[0] == ("X", 100)):
        passed += 1

    total += 1
    print(f"\n测试12: n大于总item数")
    lists12 = [[("A", 100)], [("B", 90)]]
    if run_test("n过大", lists12, 10, lambda r: len(r) == 2):
        passed += 1

    total += 1
    print(f"\n测试13: 包含空列表")
    lists13 = [
        [("A", 100), ("B", 80)],
        [],
        [("C", 90), ("A", 70)]
    ]
    if run_test("包含空列表", lists13, 3):
        passed += 1

    # ========== 随机化测试 ==========
    print("\n[3/3] 随机化测试")
    print("-" * 60)

    print("\n测试14-43: 小规模随机数据 (30组)")
    for i in range(30):
        k = random.randint(1, 5)
        lists, n = generate_random_test(k, 10, 20)
        total += 1
        try:
            result = merge_k_unsorted_lists_top_n(lists, n)
            verify_result(lists, n, result, f"随机小规模_{i}")
            passed += 1
            if (i + 1) % 10 == 0:
                print(f"  进度: {i + 1}/30")
        except Exception as e:
            print(f"  ✗ 随机小规模_{i}失败: {e}")

    print("\n测试44-53: 中等规模随机数据 (10组)")
    for i in range(10):
        k = random.randint(5, 20)
        lists, n = generate_random_test(k, 50, 100)
        total += 1
        try:
            result = merge_k_unsorted_lists_top_n(lists, n)
            verify_result(lists, n, result, f"随机中等规模_{i}")
            passed += 1
            if (i + 1) % 5 == 0:
                print(f"  进度: {i + 1}/10")
        except Exception as e:
            print(f"  ✗ 随机中等规模_{i}失败: {e}")

    # ========== 大规模测试 ==========
    print("\n[4/4] 大规模数据测试")
    print("-" * 60)

    total += 1
    print(f"\n测试54: K=10000的大列表")
    k = 10000
    lists54 = [[(f"ID_{i}", random.randint(1, 10000))] for i in range(k)]
    try:
        import time
        start = time.time()
        result = merge_k_unsorted_lists_top_n(lists54, 10)
        elapsed = time.time() - start
        verify_result(lists54, 10, result, "K=10000")
        print(f"  ✓ K=10000的大列表通过 (耗时: {elapsed:.3f}秒)")
        passed += 1
    except Exception as e:
        print(f"  ✗ K=10000的大列表失败: {e}")

    total += 1
    print(f"\n测试55: 大范围优先级")
    lists55 = [[(f"ID_{i}_{j}", random.randint(1, 10**6)) for j in range(10)] for i in range(100)]
    try:
        result = merge_k_unsorted_lists_top_n(lists55, 50)
        verify_result(lists55, 50, result, "大范围优先级")
        print(f"  ✓ 大范围优先级通过")
        passed += 1
    except Exception as e:
        print(f"  ✗ 大范围优先级失败: {e}")

    total += 1
    print(f"\n测试56: 高重复率数据")
    lists56 = [[(f"ID_{j}", random.randint(1, 1000)) for j in range(10)] for _ in range(100)]
    try:
        result = merge_k_unsorted_lists_top_n(lists56, 10)
        verify_result(lists56, 10, result, "高重复率")
        result_ids = [item[0] for item in result]
        assert len(set(result_ids)) == 10, "所有ID应唯一"
        print(f"  ✓ 高重复率数据通过")
        passed += 1
    except Exception as e:
        print(f"  ✗ 高重复率数据失败: {e}")

    # ========== 测试总结 ==========
    print("\n" + "=" * 60)
    print(f"测试总结: {passed}/{total} 通过")
    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print(f"⚠️  {total - passed} 个测试失败")
    print("=" * 60)

