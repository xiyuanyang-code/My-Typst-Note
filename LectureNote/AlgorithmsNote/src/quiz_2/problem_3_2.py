"""
AI1804 算法设计与分析
第3次课上练习：哈希表与优先队列

问题3-2：K路归并与去重

请在TODO标记处填写代码
"""

import heapq

def merge_k_unsorted_lists_top_n(lists, n):
    if not lists or n <= 0:
        return []
    
    negated_lists = []
    # 建队 O（N）
    for list_index, list in enumerate(lists):
        if list:
            negated_list = [(-b,a) for a,b in list]
            heapq.heapify(negated_list)
            negated_lists.append(negated_list)
        else:
            negated_lists.append([])

    # 初始化全局最小堆 O（N log N）
    # * 最坏情况 K 路分布不均匀，因此内部循环+弹出的时间复杂度最坏可能达到 N log N
    global_dict = {}
    global_heap = []
    heapq.heapify(global_heap)
    for list_index in range(len(lists)):
        while negated_lists[list_index]:
            new_element, new_id = heapq.heappop(negated_lists[list_index])
            if new_id not in global_dict or (-new_element) > global_dict[new_id]:
                global_dict[new_id] = -new_element
                heapq.heappush(global_heap, (new_element, new_id, list_index))
                break

    result = []
    # 寻找新的最大元素 O（n log N）
    while global_heap and len(result) < n:
        global_max, global_max_item_id, list_index = heapq.heappop(global_heap)
        result.append((global_max_item_id,-global_max))

        # inserting new elements into it
        if negated_lists[list_index]:
            new_element, new_id = heapq.heappop(negated_lists[list_index])
            if new_id not in global_dict or (-new_element) > global_dict[new_id]:
                global_dict[new_id] = -new_element
                heapq.heappush(global_heap, (new_element, new_id, list_index))
   
    return result

    # 总时间复杂度：O（N log N）

# ========== 测试代码 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("问题3-2：K路归并与去重 - 测试")
    print("=" * 60)
    
    # 测试1: K个未排序列表，获取Top-3
    print("\n=== 测试1: K个未排序列表，获取Top-3 ===")
    lists1 = [
        [("A", 100), ("C", 60), ("B", 80)],      # 未排序
        [("E", 50), ("A", 90), ("D", 85)],       # 未排序
        [("F", 40), ("B", 95), ("C", 70)]        # 未排序
    ]
    result1 = merge_k_unsorted_lists_top_n(lists1, 3)
    print(f"  输出: {result1}")
    print(f"  期望: [('A', 100), ('B', 95), ('D', 85)]")
    
    if (len(result1) == 3 and 
        result1[0] == ("A", 100) and 
        result1[1] == ("B", 95) and 
        result1[2] == ("D", 85)):
        print("  ✓ 测试1通过")
    else:
        print("  ✗ 测试1失败")
    
    # 测试2: 键冲突保留最大值
    print("\n=== 测试2: 键冲突保留最大值 ===")
    lists2 = [
        [("X", 50), ("Y", 30),],
        [("X", 100), ("Z", 20)],  # X出现，更大priority
        [("X", 30), ("W", 40)]     # X再次出现，更小priority
    ]
    result2 = merge_k_unsorted_lists_top_n(lists2, 2)
    print(f"  输出: {result2}")
    print(f"  期望: X的priority应该是100（最大）")
    
    if len(result2) >= 1 and result2[0] == ("X", 100):
        print("  ✓ 测试2通过")
    else:
        print("  ✗ 测试2失败")
    
    # 测试3: n大于总item数
    print("\n=== 测试3: n大于总item数 ===")
    lists3 = [
        [("A", 100)],
        [("B", 90)]
    ]
    result3 = merge_k_unsorted_lists_top_n(lists3, 10)
    print(f"  输出长度: {len(result3)}")
    print(f"  期望: 2（只有2个不同item）")
    
    if len(result3) == 2:
        print("  ✓ 测试3通过")
    else:
        print("  ✗ 测试3失败")
    
    # 测试4: 包含空列表
    print("\n=== 测试4: 包含空列表 ===")
    lists4 = [
        [("A", 100), ("B", 80)],
        [],  # 空列表
        [("C", 90), ("A", 70)]  # A重复，保留100
    ]
    result4 = merge_k_unsorted_lists_top_n(lists4, 3)
    print(f"  输出: {result4}")
    
    if len(result4) == 3 and ("A", 100) in result4:
        print("  ✓ 测试4通过")
    else:
        print("  ✗ 测试4失败")
    
    print("\n" + "=" * 60)

